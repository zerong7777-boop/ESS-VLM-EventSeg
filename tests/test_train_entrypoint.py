import importlib.util
from pathlib import Path


def test_train_module_imports_without_supervised_trainer_side_effect():
    repo_root = Path(__file__).resolve().parents[1]
    train_path = repo_root / "train.py"
    spec = importlib.util.spec_from_file_location("ess_train_module", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
