# DDD17 VLM Reference Scout

## Candidate Table

| Candidate | Role | Status | Notes |
| --- | --- | --- | --- |
| FC-CLIP | Primary | Selected | Use as the frozen teacher for first-pass mask generation on DDD17 image files under `data/ddd17_seg/data/dir*/imgs/`. |
| OVSeg | Fallback | Standby | Use only if FC-CLIP is blocked or produces unusable output. |
| CLIPSeg | Rejected Primary | Rejected | Keep only as the legacy baseline; do not use it as the primary teacher in this phase. |

## Decision Rules

- Prefer FC-CLIP by default.
- Fall back to OVSeg only when FC-CLIP is blocked by missing weights, incompatible inputs, runtime failure, or persistent degenerate masks.
- Reject CLIPSeg as the primary teacher for this freeze.
- Keep the reference choice fixed for the DDD17-only pseudolabel stage.
- Do not mix candidates within the same generation pass.
- Generate pseudolabels only from the DDD17 image files already aligned with the ESS event dataset root at `data/ddd17_seg/data/dir*`.

## Working Decision

- Generation source: DDD17 image files in `data/ddd17_seg/data/dir*/imgs/img_*.png`, aligned with the event dataset root that also contains `events.dat.t` and `events.dat.xyp`.
- Working candidate: FC-CLIP.
- Fallback candidate: OVSeg.
- Rejected primary: CLIPSeg.
- Output policy: produce semantic pseudolabels for the DDD17 6-class task and mark uncertain pixels as ignore.

## Why This Decision

- The current repo evidence in `docs/vlm/PSEUDOLABEL_IMPLEMENTATION.md` is still CLIPSeg-specific, so CLIPSeg is the legacy teacher and should not remain primary for this phase.
- This task is explicitly a teacher swap, so FC-CLIP becomes the primary teacher and OVSeg is reserved as the fallback path.
- Rejecting CLIPSeg as primary prevents the docs from drifting back to the pre-swap baseline once implementation starts.

## DDD17 6-Class Prompt Set

1. `flat`
2. `background`
3. `object`
4. `vegetation`
5. `human`
6. `vehicle`

## Initial Mapping Rule

- Map each DDD17 image file in `data/ddd17_seg/data/dir*/imgs/` to the 6-class prompt set in the same order as `config/settings.py`.
- Emit one 8-bit PNG mask per aligned image, using the original segmentation mask filename for the corresponding sequence entry.
- Collapse low-confidence or unmatched pixels to ignore label `255`.
- Keep the 6-class label space stable across train and validation pseudolabel generation.

## Confidence Policy

- Use the FC-CLIP per-pixel confidence score, interpreted as a sigmoid probability over the selected prompt.
- Default threshold: `0.5`.
- Pixels below the threshold are not forced into a semantic class and remain `255`.
- If a frame cannot be scored confidently enough to yield a usable mask, keep that region or frame at `255` rather than inventing a class.
