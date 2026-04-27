# FC-CLIP Teacher Swap Implementation

## Purpose

This document defines the boundary for Task 1 only.

## In Scope

- Swap the frozen teacher from CLIPSeg to FC-CLIP.
- Keep OVSeg as the fallback teacher.
- Preserve the existing pseudolabel pipeline outside teacher selection.

## Out of Scope

- OpenESS-lite trainer redesign.
- Loss-stack redesign unrelated to teacher selection.
- Dataset layout changes.
- New experiment sweeps or broader training refactors.

## Boundary Rule

If a change is not required to select, load, or fall back between teachers, it does not belong in this phase.

## Acceptance Criteria

- FC-CLIP is the primary teacher path.
- OVSeg is the fallback teacher path.
- CLIPSeg is not treated as the primary teacher in this task.
- No OpenESS-lite trainer redesign is introduced here.

## FC-CLIP No-Filter Full-Root Diagnostics

Recorded after full train-split generation under:
`<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_no_filter/train`

- Expected GT count for generated train dirs: `15950`
- Actual pseudolabel count: `15950`
- Generated dirs: `dir0, dir3, dir4, dir6, dir7`
- Mask validation: `not_2d=0`, `invalid_labels=0`, `missing_gt=0`
- Raw generated mask shape vs GT shape: `(260, 346)` vs `(200, 346)` for all masks
- Diagnostic comparison crop: use `pseudolabel[:-60, :]` to match raw DDD17 GT height
- Ignore ratio: `0.0`
- Full-mask label proportions:
  - `0`: `0.2388`
  - `1`: `0.5388`
  - `2`: `0.0043`
  - `3`: `0.0761`
  - `4`: `0.0010`
  - `5`: `0.1410`
  - `255`: `0.0`
- Cropped raw sanity metric vs DDD17 GT:
  - `mIoU = 0.5772`
  - `acc = 0.8778`
- Per-class IoU `0..5`:
  - `0`: `0.6898`
  - `1`: `0.8795`
  - `2`: `0.2027`
  - `3`: `0.5317`
  - `4`: `0.5046`
  - `5`: `0.6551`

Interpretation: FC-CLIP no-filter pseudolabels are materially stronger than the previous CLIPSeg root and are good enough to enter ESS-side training. The `0.5` confidence threshold smoke produced all-ignore masks, so this stage uses the `threshold=0.0` no-filter root.

