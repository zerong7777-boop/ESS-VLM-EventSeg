# VPR Implementation Note

## Purpose

This document freezes the actual VPR module that was implemented on top of the working `ESS + FC-CLIP` DDD17 line.

The implemented module name is:

`VPR: Vision-Language Pseudolabel Reliability`

In this project, VPR means a lightweight reliability-weighted pseudolabel loss. It is not an OpenESS-style trainer redesign and it is not a second teacher branch.

## Implemented Signal Path

The implemented path is:

`class-wise reliability prior + weighted pseudolabel cross-entropy`

The training flow is:

1. FC-CLIP generates offline pseudolabel masks on DDD17 frames.
2. ESS loads the cropped FC-CLIP pseudolabel masks from the existing training root.
3. A class-wise reliability vector is normalized into class weights.
4. The event-branch pseudolabel CE is weighted pixel-wise according to the target pseudolabel class.
5. The weighted VPR term is added as an extra auxiliary loss.

## Why Pixel-Confidence Export Was Not Chosen

The FC-CLIP wrapper can expose dense semantic scores at inference time through `predict_sem_seg(...)`, but the project had already stabilized around an on-disk hard-mask pseudolabel root:

`<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_no_filter_cropped/train`

At the time of this stage, there was no persisted confidence-map root aligned with the final cropped training masks. Generating and validating a second full confidence-map tree would have increased engineering scope and rerun cost.

To keep the module diagnostic and low-risk, the first VPR implementation used the lower-cost path:

- reuse existing FC-CLIP masks
- reuse measured FC-CLIP per-class reliability diagnostics
- add a separate weighted CE term without disturbing the proven FC-CLIP main setup

## Reliability Source

The class-wise reliability prior came from the FC-CLIP raw-cropped train sanity diagnostics recorded in `docs/vlm/FC_CLIP_IMPLEMENTATION.md`:

- class `0`: `0.6898`
- class `1`: `0.8795`
- class `2`: `0.2027`
- class `3`: `0.5317`
- class `4`: `0.5046`
- class `5`: `0.6551`

These values are treated as relative reliability cues, not as calibrated probabilities.

## Code Paths

### Helper Layer

- `utils/vpr_reliability.py`
- `tests/test_vpr_reliability.py`

Implemented helper responsibilities:

- normalize raw class reliability into a bounded weight range
- build a per-pixel weight map from the pseudolabel target
- keep ignore label pixels at zero weight

### Trainer Integration

- `training/ess_trainer.py`
- `config/settings.py`

The trainer keeps the original pseudolabel path intact and adds a separate VPR branch.

Relevant runtime knobs:

- `use_vpr_pseudo_ce`
- `weight_vpr_pseudo_ce`
- `vpr_class_reliability`
- `vpr_min_class_weight`
- `vpr_max_class_weight`

The VPR configuration sets:

- `weight_pseudolabel_loss: 0.0`
- `use_vpr_pseudo_ce: true`
- `weight_vpr_pseudo_ce: 0.05`

This keeps the VPR experiment isolated from the original FC-CLIP pseudolabel CE line.

### Config And Launcher

- Config: `config/settings_DDD17_fcclip_vpr.yaml`
- Launcher: `scripts/vlm/run_ddd17_fcclip_vpr.sh`

Consumed pseudolabel root:

- `<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_no_filter_cropped/train`

Run artifacts:

- Stdout log: `<PROJECT_ROOT>/log/bootstrap_ddd17/fcclip_vpr.stdout.log`
- Output log dir: `<PROJECT_ROOT>/log/ddd17_fcclip_vpr/20260409-170652`

## Loss Definition

The implemented VPR term is a weighted CE over the event prediction logits:

1. compute unreduced CE against the pseudolabel target
2. map each pseudolabel class to a reliability weight
3. zero out ignore-label pixels
4. compute the normalized weighted mean
5. multiply by `weight_vpr_pseudo_ce`

In code terms, this is intentionally separate from the original `weight_pseudolabel_loss` path.

## Result

Current recorded VPR result:

- final `mIoU = 54.45`
- final `Acc = 89.60`
- best observed validation `mIoU = 54.78`

Reference lines:

- official ESS baseline: `53.05 / 87.01`
- FC-CLIP main: `56.36 / 89.82`
- FC-CLIP tiny-weight: `55.87 / 89.55`

## Conclusion

The VPR module was implemented successfully as a real training branch, but the current class-wise reliability version did not beat the simpler FC-CLIP main line.

Therefore the correct framing is:

- keep VPR as a completed exploratory module
- do not promote it to the main delivered method
- if a future iteration is opened, the next higher-ceiling path is pixel-confidence VPR or a separate OpenESS-lite stage, not repeated small tuning of the current class-wise version
