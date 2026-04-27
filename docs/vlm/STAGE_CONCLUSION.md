# DDD17 VLM Follow-On Stage Conclusion

## Stage Boundary

This conclusion covers four consecutive DDD17-only follow-on stages after the official ESS baseline was verified:

1. offline dense VLM pseudolabel feasibility
2. FC-CLIP teacher swap
3. class-wise VPR reliability weighting
4. top-2 soft distillation on top of the FC-CLIP line

The stage family stayed inside the official ESS codebase and did not expand to DSEC or full OpenESS reproduction.

## What Was Completed

### 1. Official ESS Baseline Verification

The project first fixed the reference point using the official DDD17 UDA checkpoint.

Recorded baseline:

- `mIoU = 53.05`
- `Acc = 87.01`

### 2. Teacher-Only Improvement Path

The earlier weak teacher line was replaced with FC-CLIP while keeping the ESS-side integration almost unchanged.

This produced the first stable positive result:

- FC-CLIP main: `56.36 / 89.82`
- FC-CLIP tiny-weight: `55.87 / 89.55`

This remains the best stable final-run result of the whole follow-on stage.

### 3. VPR Innovation Module

A lightweight class-wise reliability-weighted pseudolabel module was added on top of the FC-CLIP line.

Recorded VPR result:

- final: `54.45 / 89.60`
- best observed validation mIoU: `54.78`

This confirms the module runs correctly, but it does not beat the simpler FC-CLIP main line.

### 4. Top-2 Soft Distillation Module

A second follow-on stage added offline top-2 teacher supervision on top of the existing hard FC-CLIP pseudolabel path.

Recorded main result:

- final: `56.1993 / 89.8037`
- best mIoU checkpoint: `56.6576`

Recorded short ablation result with lower soft weight:

- final: `56.0884 / 89.6736`
- best mIoU checkpoint: `56.7029`

This is the strongest innovation module explored in the project after the FC-CLIP teacher swap.

## Final Method Decision

The final recommended stable method for this project stage remains:

`ESS + FC-CLIP offline dense pseudolabel supervision`

Reason:

- it is still the best verified final-run metric
- it improves clearly over the official ESS baseline
- it stays within a controllable engineering scope
- it is easier to explain and reproduce than a checkpoint-sensitive variant

## How To Frame Top-2 Soft Distillation

Top-2 soft distillation should be framed as:

- a completed positive innovation module
- evidence that richer offline teacher supervision is useful on DDD17
- a checkpoint-sensitive gain rather than a stronger stable convergence recipe

This distinction matters:

- the implementation succeeded
- the branch produced the best observed checkpoint of the whole project (`56.7029`)
- the branch did not produce the best final converged run

## Stage Outcome Summary

Stable final-run ranking:

1. `ESS + FC-CLIP` final `56.36 / 89.82`
2. `ESS + FC-CLIP + top-2 soft distillation` main final `56.20 / 89.80`
3. `ESS + FC-CLIP + top-2 soft distillation` ablation final `56.09 / 89.67`
4. `ESS + FC-CLIP + VPR` final `54.45 / 89.60`
5. official ESS baseline `53.05 / 87.01`

Best-checkpoint ranking:

1. `top2-ablation` best `56.7029`
2. `top2-main` best `56.6576`
3. `fcclip-main` best/final `56.36`
4. `vpr-main` best `54.78`
5. official baseline `53.05`

## Recommended External Summary

A precise external summary for the customer or thesis report is:

`On DDD17, we first verified the official ESS baseline, then introduced FC-CLIP offline dense pseudolabel supervision and obtained a stable mIoU gain from 53.05 to 56.36. We further implemented two enhancement modules beyond the FC-CLIP main line: a class-wise VPR branch and an offline top-2 soft-distillation branch. The VPR branch did not exceed the FC-CLIP main result. The top-2 soft-distillation branch produced the best observed checkpoint of the whole project at 56.70 mIoU, but its final converged runs remained slightly below the FC-CLIP main line, so the final recommended stable method remains ESS plus FC-CLIP.`

## Next-Step Judgment

If another stage is opened, the most justified next step is not more class-wise VPR tuning. The strongest next candidate is a stabilization-oriented follow-on for the top-2 soft branch, such as checkpoint selection strategy, early-stopping policy, or a better-calibrated soft-distillation schedule.
