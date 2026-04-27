# Top-2 Soft Best-Checkpoint Optimization

- Last Updated: `2026-04-26`

## Stage Contract

- task family: `ESS + FC-CLIP + top-2 soft distillation`
- target metric: best validation checkpoint mIoU, not final epoch mIoU
- current branch best: `56.7029` best val mIoU from `settings_DDD17_fcclip_top2_soft_ablation.yaml`
- run cap: max `3` complete DDD17 training runs
- launcher default behavior: preflight only; training requires `ESS_RUN_TRAIN=1`

## Baseline Table

| Result | Source | Meaning |
| --- | --- | --- |
| `53.05` | official/resume baseline reference | lower bound reference for DDD17 UDA recovery |
| `56.36` | FC-CLIP hard pseudolabel main | prior VLM pseudolabel branch reference |
| `56.6576` | top-2 soft main best checkpoint | near-threshold best checkpoint |
| `56.7029` | top-2 soft short ablation best checkpoint | current best to beat |

## Optimization Candidates

| Order | Mode | Config | Weight | Temperature | Gate | Purpose |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | `w0015` | `config/settings_DDD17_fcclip_top2_soft_best_w0015.yaml` | `0.015` | `1.0` | off | first bounded best-checkpoint attempt |
| 2 | `w001` | `config/settings_DDD17_fcclip_top2_soft_best_w001.yaml` | `0.01` | `1.0` | off | lower-weight retry if `w0015` does not settle the stage |
| 3 | `temp15` | `config/settings_DDD17_fcclip_top2_soft_best_temp15.yaml` | `0.015` default | `1.5` | off | soften teacher mass if A/B are close or stable |
| 3 alternate | `gate` | `config/settings_DDD17_fcclip_top2_soft_best_gate.yaml` | `0.015` | `1.0` | `p1>=0.50`, margin `>=0.10` | use instead of `temp15` only if noisy or ambiguous pixels look likely |

Run C must be exactly one of `temp15` or `gate`; do not run both in this stage unless the run cap is explicitly revised.

## Non-Goals

- no VPR retry
- no new teacher model
- no online teacher
- no full-distribution distillation
- no trainer/loss/settings/test edits in this worker's scope
- no claim about final-epoch convergence unless a run also improves final metrics

## Preflight Evidence

Pending command:

```bash
cd <PROJECT_ROOT>
bash scripts/vlm/run_ddd17_top2_soft_best_ckpt_optimization.sh w0015
```

Expected behavior: print project root, config path, python path, and training command; exit before training unless `ESS_RUN_TRAIN=1`.

