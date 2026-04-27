from pathlib import Path

import numpy as np
import pytest
import yaml

from config.settings import Settings
from datasets.ddd17_events_loader import normalize_pseudolabel_mask, resolve_pseudolabel_root


def build_settings_payload(tmp_path, *, include_weight=True):
    dataset_a_root = tmp_path / "cityscapes"
    dataset_b_root = tmp_path / "ddd17"
    pseudolabel_root = tmp_path / "pseudolabels"
    log_root = tmp_path / "logs"

    dataset_a_root.mkdir()
    dataset_b_root.mkdir()
    pseudolabel_root.mkdir()
    log_root.mkdir()

    payload = {
        "dataset": {
            "name_a": "Cityscapes_gray",
            "name_b": "DDD17_events",
            "cityscapes_img": {
                "dataset_path": str(dataset_a_root),
                "shape": [200, 352],
                "random_crop": True,
                "read_two_imgs": False,
                "require_paired_data_train": False,
                "require_paired_data_val": False,
            },
            "DDD17_events": {
                "dataset_path": str(dataset_b_root),
                "split_train": "train",
                "shape": [200, 346],
                "nr_events_data": 20,
                "nr_events_files_per_data": None,
                "fixed_duration": False,
                "delta_t_per_data": 50,
                "require_paired_data_train": True,
                "require_paired_data_val": True,
                "nr_events_window": 32000,
                "event_representation": "voxel_grid",
                "nr_temporal_bins": 5,
                "separate_pol": False,
                "normalize_event": False,
                "pseudolabels_path_b": str(pseudolabel_root),
                "pseudolabel_confidence_path_b": str(pseudolabel_root),
                "use_pseudolabels_train_b": True,
                "pseudolabel_ignore_label_b": 254,
            },
        },
        "task": {"semseg_num_classes": 6},
        "dir": {"log": str(log_root)},
        "model": {
            "model_name": "ess",
            "skip_connect_encoder": True,
            "skip_connect_task": True,
            "skip_connect_task_type": "concat",
            "data_augmentation_train": True,
            "train_on_event_labels": False,
        },
        "optim": {
            "batch_size_a": 1,
            "batch_size_b": 1,
            "lr_front": 1e-5,
            "lr_back": 1e-4,
            "lr_decay": 1,
            "num_epochs": 1,
            "val_epoch_step": 1,
            "weight_task_loss": 1,
            "weight_cycle_pred_loss": 1,
            "weight_cycle_emb_loss": 0.01,
            "weight_cycle_task_loss": 0.01,
            "task_loss": ["dice", "cross_entropy"],
        },
        "checkpoint": {
            "save_checkpoint": False,
            "resume_training": False,
            "load_pretrained_weights": False,
            "resume_file": "",
            "pretrained_file": "",
        },
        "hardware": {"num_cpu_workers": 0, "gpu_device": "cpu"},
    }
    if include_weight:
        payload["optim"]["weight_pseudolabel_loss"] = 0.2
    return payload


def write_settings(tmp_path, payload):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return settings_path


def test_settings_parse_ddd17_vlm_pseudolabel_fields(tmp_path):
    payload = build_settings_payload(tmp_path)

    settings = Settings(str(write_settings(tmp_path, payload)), generate_log=False)

    assert settings.use_pseudolabels_train_b is True
    assert Path(settings.pseudolabels_path_b) == tmp_path / "pseudolabels"
    assert Path(settings.pseudolabel_confidence_path_b) == tmp_path / "pseudolabels"
    assert settings.pseudolabel_ignore_label_b == 254
    assert settings.weight_pseudolabel_loss == 0.2


def test_settings_default_pseudolabel_loss_weight_is_zero(tmp_path):
    payload = build_settings_payload(tmp_path, include_weight=False)

    settings = Settings(str(write_settings(tmp_path, payload)), generate_log=False)

    assert settings.weight_pseudolabel_loss == 0.0


def test_settings_parse_pixel_confidence_fields(tmp_path):
    payload = build_settings_payload(tmp_path)
    payload["dataset"]["DDD17_events"]["max_train_samples"] = 8
    payload["dataset"]["DDD17_events"]["max_valid_samples"] = 4
    payload["optim"]["use_confidence_weighted_pseudolabel_loss"] = True
    payload["optim"]["train_loader_len_source"] = "second"
    payload["optim"]["skip_validation"] = True

    settings = Settings(str(write_settings(tmp_path, payload)), generate_log=False)

    assert settings.use_confidence_weighted_pseudolabel_loss is True
    assert settings.max_train_samples_b == 8
    assert settings.max_val_samples_b == 4
    assert settings.train_loader_len_source == "second"
    assert settings.skip_validation is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["dataset"]["DDD17_events"].update({"pseudolabels_path_b": ""}),
        lambda payload: payload["dataset"]["DDD17_events"].update({"require_paired_data_train": False}),
    ],
)
def test_settings_reject_invalid_pseudolabel_training_config(tmp_path, mutator):
    payload = build_settings_payload(tmp_path)
    mutator(payload)

    with pytest.raises(AssertionError):
        Settings(str(write_settings(tmp_path, payload)), generate_log=False)


def test_settings_reject_missing_confidence_root_when_confidence_weighting_enabled(tmp_path):
    payload = build_settings_payload(tmp_path)
    payload["optim"]["use_confidence_weighted_pseudolabel_loss"] = True
    payload["dataset"]["DDD17_events"]["pseudolabel_confidence_path_b"] = ""

    with pytest.raises(AssertionError):
        Settings(str(write_settings(tmp_path, payload)), generate_log=False)


def test_resolve_pseudolabel_root_accepts_top_level_and_split_specific_paths(tmp_path):
    top_level_root = tmp_path / "pseudolabels"
    split_root = top_level_root / "train"
    split_root.mkdir(parents=True)

    assert resolve_pseudolabel_root(str(top_level_root), "train") == top_level_root
    assert resolve_pseudolabel_root(str(split_root), "train") == top_level_root


def test_resolve_pseudolabel_root_rejects_missing_split_dir(tmp_path):
    pseudolabel_root = tmp_path / "pseudolabels"
    pseudolabel_root.mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_pseudolabel_root(str(pseudolabel_root), "train")


def test_normalize_pseudolabel_mask_maps_invalid_ids_to_ignore():
    mask = np.array([[0, 5, 254], [6, -1, 255]], dtype=np.int64)

    normalized = normalize_pseudolabel_mask(mask, num_classes=6, ignore_label=254)

    assert normalized.dtype == np.int64
    assert normalized.tolist() == [[0, 5, 254], [254, 254, 254]]
