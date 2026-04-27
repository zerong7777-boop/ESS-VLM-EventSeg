from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FC_CLIP_ROOT = REPO_ROOT / 'third_party' / 'fc-clip'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fcclip_confidence_utils import (
    sem_seg_to_mask_and_confidence,
    sem_seg_to_top2_soft_maps,
    threshold_mask,
)

DDD17_CLASS_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ('road', 'street', 'sidewalk', 'pavement'),
    ('sky', 'building', 'wall', 'fence'),
    ('pole', 'traffic light', 'traffic sign', 'street pole'),
    ('vegetation', 'tree', 'grass', 'bush'),
    ('person', 'pedestrian', 'rider', 'cyclist'),
    ('car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train', 'van', 'jeep', 'suv', 'trailer', 'tram'),
)



def _deterministic_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    colors: List[Tuple[int, int, int]] = []
    for idx in range(num_classes):
        colors.append(((53 * idx) % 255, (97 * idx + 71) % 255, (193 * idx + 29) % 255))
    return colors



def _build_metadata(dataset_name: str, class_groups: Sequence[Sequence[str]]):
    class_names = [','.join(group) for group in class_groups]
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(dataset_name, lambda: [])
    colors = _deterministic_colors(len(class_names))
    ids = {idx: idx for idx in range(len(class_names))}
    return MetadataCatalog.get(dataset_name).set(
        stuff_classes=class_names,
        thing_classes=class_names,
        stuff_colors=colors,
        thing_colors=colors,
        thing_dataset_id_to_contiguous_id=ids,
        stuff_dataset_id_to_contiguous_id=ids,
    )



def _resolve_repo_path(path_like: str) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())



def _resolve_cfg_opts(opts: Optional[Sequence[str]]) -> list[str]:
    if not opts:
        return []

    resolved = list(opts)
    for index, item in enumerate(resolved[:-1]):
        if item == 'MODEL.WEIGHTS':
            resolved[index + 1] = _resolve_repo_path(resolved[index + 1])
    return resolved


@contextmanager
def _fcclip_repo_context():
    original_cwd = Path.cwd()
    fcclip_path = str(FC_CLIP_ROOT)
    added_path = False
    if fcclip_path not in sys.path:
        sys.path.insert(0, fcclip_path)
        added_path = True
    os.chdir(FC_CLIP_ROOT)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        if added_path:
            try:
                sys.path.remove(fcclip_path)
            except ValueError:
                pass


class FCCLIPDensePredictor:
    def __init__(
        self,
        config_file: str,
        opts: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        class_groups: Sequence[Sequence[str]] = DDD17_CLASS_GROUPS,
        metadata_name: str = 'ddd17_fcclip_teacher',
    ) -> None:
        config_file = _resolve_repo_path(config_file)
        opts = _resolve_cfg_opts(opts)

        with _fcclip_repo_context():
            from fcclip import add_fcclip_config, add_maskformer2_config

            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            add_fcclip_config(cfg)
            cfg.merge_from_file(config_file)
            if opts:
                cfg.merge_from_list(list(opts))
            if device is not None:
                cfg.defrost()
                cfg.MODEL.DEVICE = device
                cfg.freeze()
            self.cfg = cfg
            self.metadata = _build_metadata(metadata_name, class_groups)
            self.predictor = DefaultPredictor(cfg)
            model = self.predictor.model
            if hasattr(model, 'set_metadata'):
                model.set_metadata(self.metadata)
            elif hasattr(self.predictor, 'set_metadata'):
                self.predictor.set_metadata(self.metadata)
            else:
                raise AttributeError('FC-CLIP predictor does not expose set_metadata')

    @torch.inference_mode()
    def predict_sem_seg(self, image_bgr):
        predictions = self.predictor(image_bgr)
        if 'sem_seg' not in predictions:
            raise KeyError('FC-CLIP predictions did not contain sem_seg')
        return predictions['sem_seg']

    @torch.inference_mode()
    def predict_mask_and_confidence(self, image_bgr) -> tuple[np.ndarray, np.ndarray]:
        sem_seg = self.predict_sem_seg(image_bgr)
        return sem_seg_to_mask_and_confidence(sem_seg)

    @torch.inference_mode()
    def predict_top2_soft(self, image_bgr) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sem_seg = self.predict_sem_seg(image_bgr)
        return sem_seg_to_top2_soft_maps(sem_seg)

    @torch.inference_mode()
    def predict_mask(self, image_bgr, threshold: float = 0.5) -> np.ndarray:
        mask, confidence = self.predict_mask_and_confidence(image_bgr)
        return threshold_mask(mask, confidence, threshold=threshold, ignore_label=255)
