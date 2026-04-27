# Figure Guide for ESS-VLM-EventSeg

This document explains the README figures for the ESS-VLM-EventSeg GitHub page.

Important metric note: `56.8124` is the best validation checkpoint result at Epoch 15, not the final epoch result. The final epoch result is `56.1059` at Epoch 18.

FC-CLIP should be described as an offline visual-language teacher used to generate supervision artifacts. Do not describe or imply FC-CLIP online inference during ESS training. Do not claim SOTA.

## Required Figures

| Figure | Type | Purpose | Recommended README Position |
| --- | --- | --- | --- |
| `hero_overview.png` | Schematic overview with real metric summary | Gives a 5-10 second overview: DDD17 events go to the ESS student, frames are processed by the FC-CLIP offline teacher, and training uses hard pseudolabel CE plus Top-2 Soft distillation. The result badge says `mIoU 53.05 -> 56.81` and `best checkpoint`. | Top of README, before the project introduction |
| `method_pipeline.png` | Schematic method diagram | Explains the full teacher-student pipeline: DDD17 paired frames/events, FC-CLIP offline teacher outputs, hard pseudolabels, top-2 ids/probabilities, ESS event student, hard CE, and Top-2 Soft distillation CE. | Method section |
| `top2_uncertainty_explainer.png` | Explanatory schematic, not measured pixel data | Explains why Top-2 Soft labels contain richer supervision than one-hot hard labels. The example distribution is illustrative only. | Top-2 Soft distillation subsection |
| `contribution_waterfall.png` | Real metric-backed figure | Shows the contribution path from `ESS baseline: 53.05` to `+ FC-CLIP pseudolabel: +3.31`, then `+ Top-2 Soft best checkpoint: +0.4524`, ending at `56.8124`. VPR is annotated as explored but not selected. | Contribution or ablation summary section |
| `checkpoint_curve.png` | Real metric-backed figure | Shows validation mIoU from Epoch 13 to Epoch 18 and highlights `Epoch 15: 56.8124` as the best validation checkpoint. | Experiment notes or checkpoint selection section |
| `miou_comparison.png` | Real metric-backed figure | Provides the simplest metric comparison: `ESS baseline: 53.05`, `ESS + FC-CLIP: 56.36`, and `ESS + FC-CLIP + Top-2 Soft: 56.8124 best checkpoint`. | Results section, preferably after the result table |

## Suggested README Order

1. `hero_overview.png`
2. Short project introduction
3. Result table
4. `miou_comparison.png`
5. `method_pipeline.png`
6. `top2_uncertainty_explainer.png`
7. `checkpoint_curve.png`
8. `contribution_waterfall.png`

## Metric Facts Used by Result Figures

| Item | mIoU | Acc | Note |
| --- | ---: | ---: | --- |
| ESS official baseline | 53.05 | 87.01 | DDD17 official ESS UDA validation |
| ESS + FC-CLIP dense pseudolabel | 56.36 | 89.82 | Stable VLM pseudolabel branch |
| VPR exploration | 54.45 | 89.60 | Explored, not selected as final method |
| Previous Top-2 Soft best | 56.7029 | 89.5777 | Previous best checkpoint |
| Current Top-2 Soft w0015 best | 56.8124 | 89.5776 | Epoch 15, best validation checkpoint |
| Current Top-2 Soft w0015 final | 56.1059 | 89.6507 | Epoch 18, final epoch |

## Wording Constraints

Use these descriptions consistently:

- `FC-CLIP offline teacher`
- `best validation checkpoint`
- `Top-2 Soft distillation`
- `Validation mIoU on DDD17`

Avoid these unsupported claims:

- Do not call the result SOTA.
- Do not describe FC-CLIP as running online during ESS training.
- Do not present the Top-2 uncertainty example as a real measured pixel distribution.
- Do not present VPR as part of the final improvement path.
