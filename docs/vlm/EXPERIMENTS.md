# DDD17 ESS VLM Experiment Ledger

## Scope

This file is the canonical experiment ledger for the DDD17-only ESS follow-on stages after the official baseline verification.

The ledger now covers five method states:

1. official ESS baseline
2. ESS plus FC-CLIP offline dense pseudolabel supervision
3. ESS plus FC-CLIP plus class-wise VPR
4. ESS plus FC-CLIP plus top-2 soft distillation main
5. ESS plus FC-CLIP plus top-2 soft distillation short ablation

## Fixed Reference Artifacts

- Remote project root: `<PROJECT_ROOT>`
- Baseline checkpoint: `<PROJECT_ROOT>/weights/official/DDD17_UDA.pt.pt`
- Loader-consumed FC-CLIP hard pseudolabel root: `<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_no_filter_cropped/train`
- Loader-consumed FC-CLIP top-2 soft root: `<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_top2_soft_cropped`

## Experiment Table

| Stage | Run | Config | Extra Signal | Final mIoU | Final Acc | Best mIoU | Best Step | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | official ESS reference | `config/settings_DDD17_official_UDA_resume_eval_officialish.yaml` | `none` | `53.05` | `87.01` | `53.05` | `13` | Fixed anchor for all later comparisons. |
| fcclip-main | FC-CLIP main | `config/settings_DDD17_fcclip_pseudolabel.yaml` | hard FC-CLIP masks | `56.36` | `89.82` | `56.36` | `18` | Current best stable final-run result. Relative to baseline: `+3.31 mIoU`, `+2.81 Acc`. |
| fcclip-ablation | FC-CLIP tiny-weight | `config/settings_DDD17_fcclip_pseudolabel_tiny_weight.yaml` | hard FC-CLIP masks | `55.87` | `89.55` | `55.87` | `18` | Lower hard-mask weight still beats baseline. |
| vpr-main | FC-CLIP plus VPR | `config/settings_DDD17_fcclip_vpr.yaml` | class-wise reliability weighting | `54.45` | `89.60` | `54.78` | `best intermediate` | Above baseline, below FC-CLIP main. |
| top2-main | FC-CLIP plus top-2 soft distillation | `config/settings_DDD17_fcclip_top2_soft.yaml` | hard masks + offline top-2 soft labels | `56.20` | `89.80` | `56.66` | `15` | Near-miss line. Final metric below FC-CLIP main, but best checkpoint nearly reaches the stage threshold. |
| top2-ablation | top-2 soft short ablation | `config/settings_DDD17_fcclip_top2_soft_ablation.yaml` | hard masks + lower-weight top-2 soft labels | `56.09` | `89.67` | `56.70` | `15` | Lower soft weight improves the best intermediate checkpoint and crosses the `56.66` bar, but final convergence is still below FC-CLIP main. |

## Interpretation

### What the ledger proves

- The official ESS baseline is reproducible and fixed at `53.05 / 87.01` on the DDD17 path used in this project.
- FC-CLIP hard pseudolabel supervision is a stable positive method change over the official baseline.
- Class-wise VPR does not beat the FC-CLIP main line.
- Top-2 soft distillation is stronger than VPR and can exceed the `56.66` threshold at an intermediate checkpoint when the soft weight is reduced.
- Top-2 soft distillation does not yet replace the FC-CLIP main line as the best stable final-run recipe.

### Current method ranking

If ranking by stable final-run metric:

1. `ESS + FC-CLIP` final `56.36 / 89.82`
2. `ESS + FC-CLIP + top-2 soft distillation` main final `56.20 / 89.80`
3. `ESS + FC-CLIP + top-2 soft distillation` ablation final `56.09 / 89.67`
4. `ESS + FC-CLIP` tiny-weight final `55.87 / 89.55`
5. `ESS + FC-CLIP + VPR` final `54.45 / 89.60`
6. official ESS baseline `53.05 / 87.01`

If ranking by best observed checkpoint mIoU:

1. `top2-ablation` best `56.70`
2. `top2-main` best `56.66`
3. `fcclip-main` best/final `56.36`
4. `fcclip-ablation` best/final `55.87`
5. `vpr-main` best `54.78`
6. official ESS baseline `53.05`

### Decision bound

For delivery, report, and thesis framing, the recommended stable main method remains:

`ESS + FC-CLIP offline dense pseudolabel supervision`

For innovation framing, top-2 soft distillation should be reported as:

- a completed positive follow-on module
- evidence that richer teacher supervision can improve the best checkpoint
- not yet a stronger final-run replacement without checkpoint-sensitive selection

## Related Documents

- `docs/vlm/FC_CLIP_IMPLEMENTATION.md`
- `docs/vlm/VPR_IMPLEMENTATION.md`
- `docs/ai/TOP2_SOFT_DISTILLATION.md`
- `docs/vlm/STAGE_CONCLUSION.md`
