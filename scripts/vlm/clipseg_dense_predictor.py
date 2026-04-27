from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torchvision import transforms

_CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"
# The remote host resets TLS connections to huggingface.co, so this wrapper uses
# the official CLIPSeg codeload source archive plus the official ownCloud weights
# and caches them under repo-local .cache/clipseg on first use.
_CLIPSEG_SOURCE_HEAD_ZIP_URL = "https://codeload.github.com/timojl/clipseg/zip/refs/heads/master"
_CLIPSEG_SOURCE_HEAD_ZIP_SHA256 = "a4c5c35b9db8740593a224746b389cb600afcaf3ac50a03bd83d490ff5ab70ec"
_CLIPSEG_WEIGHTS_ZIP_URL = "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download"
_CLIPSEG_WEIGHTS_ZIP_SHA256 = "6d2d55042551b8f5d7be2158593c667cb383728bb5d856e204ad2a3fb3ef81a3"
# OpenAI CLIP is required by the official CLIPSeg source. GitHub only exposes a
# branch-head archive URL here, so we pin the currently accepted SHA256 and fail
# loudly if upstream changes the archive until this constant is reviewed.
_OPENAI_CLIP_SOURCE_HEAD_ZIP_URL = "https://codeload.github.com/openai/CLIP/zip/refs/heads/main"
_OPENAI_CLIP_SOURCE_HEAD_ZIP_SHA256 = "7ced2e3a4fb6ea456f130f25fec2f7ccaff40fe47b56fdbf5eefd846b2bec1ee"
_OPENAI_CLIP_VITB16_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
)
_OPENAI_CLIP_VITB16_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
_CLIPSEG_REFINED_WEIGHTS_RELATIVE_PATH = "clipseg_weights/rd64-uni-refined.pth"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256_file(destination) == expected_sha256:
        return destination
    if destination.exists():
        destination.unlink()
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    actual_sha256 = _sha256_file(destination)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for {destination}: expected {expected_sha256}, got {actual_sha256}"
        )
    return destination


def _refresh_extracted_archive(archive_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        top_level = archive.namelist()[0].split("/", 1)[0]
        extracted_dir = extract_root / top_level
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        archive.extractall(extract_root)
    return extracted_dir


def _refresh_refined_weights(archive_path: Path, weights_root: Path, cache_root: Path) -> Path:
    weights_root.mkdir(parents=True, exist_ok=True)
    refined_weights_path = weights_root / "rd64-uni-refined.pth"
    if refined_weights_path.exists():
        refined_weights_path.unlink()
    extracted_parent = cache_root / "clipseg_weights"
    if extracted_parent.exists():
        shutil.rmtree(extracted_parent)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extract(_CLIPSEG_REFINED_WEIGHTS_RELATIVE_PATH, cache_root)
    extracted = cache_root / _CLIPSEG_REFINED_WEIGHTS_RELATIVE_PATH
    extracted.replace(refined_weights_path)
    shutil.rmtree(extracted_parent)
    return refined_weights_path


def _load_clipseg_module(clipseg_source_dir: Path):
    module_name = "clipseg_official_module"
    if module_name in sys.modules:
        del sys.modules[module_name]
    module_path = clipseg_source_dir / "models" / "clipseg.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


class CLIPSegDensePredictor:
    def __init__(self, model_id: str = _CLIPSEG_MODEL_ID, device: str | None = None):
        if model_id != _CLIPSEG_MODEL_ID:
            raise ValueError(f"unsupported model_id: {model_id}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.cache_root = _repo_root() / ".cache" / "clipseg"
        self.source_root = self.cache_root / "src"
        self.weights_root = self.cache_root / "weights"
        self.clip_root = self.cache_root / "openai_clip"

        clipseg_archive = _download_file(
            _CLIPSEG_SOURCE_HEAD_ZIP_URL,
            self.cache_root / "clipseg-master.zip",
            _CLIPSEG_SOURCE_HEAD_ZIP_SHA256,
        )
        clipseg_weights_archive = _download_file(
            _CLIPSEG_WEIGHTS_ZIP_URL,
            self.cache_root / "clipseg_weights.zip",
            _CLIPSEG_WEIGHTS_ZIP_SHA256,
        )
        openai_clip_archive = _download_file(
            _OPENAI_CLIP_SOURCE_HEAD_ZIP_URL,
            self.cache_root / "openai-clip-main.zip",
            _OPENAI_CLIP_SOURCE_HEAD_ZIP_SHA256,
        )
        _download_file(
            _OPENAI_CLIP_VITB16_URL,
            self.clip_root / "ViT-B-16.pt",
            _OPENAI_CLIP_VITB16_SHA256,
        )

        # Always refresh extracted runtime trees from the verified archives so a
        # stale prior extraction cannot silently survive a new cache state.
        clipseg_source_dir = _refresh_extracted_archive(clipseg_archive, self.source_root)
        openai_clip_source_dir = _refresh_extracted_archive(openai_clip_archive, self.source_root)
        refined_weights_path = _refresh_refined_weights(
            clipseg_weights_archive,
            self.weights_root,
            self.cache_root,
        )

        path_str = str(openai_clip_source_dir)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

        import clip

        clipseg_module = _load_clipseg_module(clipseg_source_dir)
        CLIPDensePredT = clipseg_module.CLIPDensePredT

        original_clip_load = clip.load
        local_clip_root = str(self.clip_root)

        def _load_with_local_cache(name, *args, **kwargs):
            kwargs.setdefault("download_root", local_clip_root)
            return original_clip_load(name, *args, **kwargs)

        clip.load = _load_with_local_cache
        try:
            self.model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64, complex_trans_conv=True)
        finally:
            clip.load = original_clip_load

        state_dict = torch.load(refined_weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((352, 352)),
            ]
        )

    @torch.inference_mode()
    def predict_logits(self, image, prompts: Sequence[str]) -> torch.Tensor:
        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompts must be non-empty")

        image = image.convert("RGB")
        width, height = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        batch = tensor.repeat(len(prompts), 1, 1, 1)
        logits = self.model(batch, prompts)[0]
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return logits.squeeze(1)
