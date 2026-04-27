# ESS Bootstrap Baseline Run

- Command: `bash scripts/bootstrap/run_ddd17_smoke.sh`
- Launcher: `<PROJECT_ROOT>/scripts/bootstrap/run_ddd17_smoke.sh`
- Config: `<PROJECT_ROOT>/config/settings_DDD17_bootstrap.yaml`
- Stdout log: `<PROJECT_ROOT>/log/bootstrap_ddd17/smoke.stdout.log`
- Current result: `blocked in DDD17 preflight`
- Exit code: `1`
- Train run directory: `not created`
- Checkpoints directory: `not created`

## Current Meaning

The smoke launcher is operational. On the current remote state it reaches:

1. environment probe
2. DDD17 verifier
3. controlled stop before `train.py`

This is the expected result for the current machine because `<PROJECT_ROOT>/data/ddd17_seg` is still missing.

Important boundary:

- This log does **not** prove baseline training has started yet.
- After DDD17 is staged, the next preflight gate in the launcher is the Cityscapes sensor-A check.
- Only after both datasets pass preflight will the launcher invoke `python3 train.py --settings_file config/settings_DDD17_bootstrap.yaml`.

## First Useful Output Lines

```text
[SMOKE] DDD17 bootstrap launcher
[SMOKE] project_root=<PROJECT_ROOT>
[SMOKE] config=config/settings_DDD17_bootstrap.yaml
DDD17 layout check root: <PROJECT_ROOT>/data/ddd17_seg
Scope: DDD17 sensor-B only. This check does not validate Cityscapes sensor-A readiness.
[FAIL] dataset_root: expected <PROJECT_ROOT>/data/ddd17_seg
RESULT: FAIL (1 failing checks)
[SMOKE] exit_code=1
```
