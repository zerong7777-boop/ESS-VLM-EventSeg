from pathlib import Path

import pytest
import yaml

from config.settings import Settings


def build_payload(tmp_path):
    dataset_a_root = tmp_path / "cityscapes"
    dataset_b_root = tmp_path / "ddd17"
    pseudolabel_root = tmp_path / "pseudolabels"
    top2_root = tmp_path / "top2"
    log_root = tmp_path / "logs"
    for path in [dataset_a_root, dataset_b_root, pseudolabel_root, top2_root, log_root]:
        path.mkdir()

    return {
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
                "top2_soft_labels_path_b": str(top2_root),
                "use_pseudolabels_train_b": True,
                "pseudolabel_ignore_label_b": 255,
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
            "weight_pseudolabel_loss": 0.2,
            "use_top2_soft_distillation": True,
            "weight_top2_soft_distillation": 0.1,
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


def write_settings(tmp_path, payload):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return settings_path


def test_settings_parse_top2_soft_fields(tmp_path):
    payload = build_payload(tmp_path)
    settings = Settings(str(write_settings(tmp_path, payload)), generate_log=False)

    assert settings.use_top2_soft_distillation is True
    assert settings.weight_top2_soft_distillation == 0.1
    assert Path(settings.top2_soft_labels_path_b) == tmp_path / "top2"
    assert settings.top2_soft_temperature == 1.0
    assert settings.top2_soft_min_p1 == 0.0
    assert settings.top2_soft_min_margin == 0.0


def test_settings_parse_non_default_top2_soft_temperature_and_gate_fields(tmp_path):
    payload = build_payload(tmp_path)
    payload["optim"]["top2_soft_temperature"] = 1.5
    payload["optim"]["top2_soft_min_p1"] = 0.5
    payload["optim"]["top2_soft_min_margin"] = 0.1

    settings = Settings(str(write_settings(tmp_path, payload)), generate_log=False)

    assert settings.top2_soft_temperature == 1.5
    assert settings.top2_soft_min_p1 == 0.5
    assert settings.top2_soft_min_margin == 0.1


@pytest.mark.parametrize(
    "field,value",
    [("top2_soft_temperature", 0.0), ("top2_soft_min_p1", -0.1), ("top2_soft_min_margin", -0.1)],
)
def test_settings_reject_invalid_top2_soft_temperature_and_gate_fields(tmp_path, field, value):
    payload = build_payload(tmp_path)
    payload["optim"][field] = value

    with pytest.raises(AssertionError):
        Settings(str(write_settings(tmp_path, payload)), generate_log=False)


def test_settings_reject_missing_top2_root_when_enabled(tmp_path):
    payload = build_payload(tmp_path)
    payload["dataset"]["DDD17_events"]["top2_soft_labels_path_b"] = ""

    with pytest.raises(AssertionError):
        Settings(str(write_settings(tmp_path, payload)), generate_log=False)


def test_settings_reject_top2_soft_with_confidence_weighting(tmp_path):
    payload = build_payload(tmp_path)
    payload["dataset"]["DDD17_events"]["pseudolabel_confidence_path_b"] = str(tmp_path / "top2")
    payload["optim"]["use_confidence_weighted_pseudolabel_loss"] = True

    with pytest.raises(AssertionError):
        Settings(str(write_settings(tmp_path, payload)), generate_log=False)
