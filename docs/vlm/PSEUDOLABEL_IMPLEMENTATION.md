# DDD17 VLM Pseudolabel Implementation

## Root

- Pseudolabel root: `<PROJECT_ROOT>/data/ddd17_pseudolabels/clipseg`
- Source alignment root: `<PROJECT_ROOT>/data/ddd17_seg/data/dir*/imgs/`

## Split Layout

- `train/`: pseudolabels consumed by training.
- `valid/`: pseudolabels generated only for inspection and debugging.
- Pseudolabel split membership must be derived from the current indexed DDD17 loader partition in `datasets/ddd17_events_loader.py`, which maps the live on-disk sequence list to:
  - `train`: `dir0`, `dir3`, `dir4`, `dir6`, `dir7`
  - `valid`: `dir1`
- The current live dataset directories on disk are `dir0`, `dir1`, `dir3`, `dir4`, `dir5`, `dir6`, and `dir7`.

Concrete contract:

```text
<PROJECT_ROOT>/data/ddd17_pseudolabels/clipseg/
  train/
    dir0/
      segmentation_masks/
        segmentation_00000002.png
        segmentation_00000003.png
        ...
    dir3/
      segmentation_masks/
        ...
    dir4/
      segmentation_masks/
        ...
    dir6/
      segmentation_masks/
        ...
    dir7/
      segmentation_masks/
        ...
  valid/
    dir1/
      segmentation_masks/
        segmentation_00000002.png
        ...
```

Implementation details:

- Each `dir*` folder must match the source ESS sequence directory name.
- Each pseudolabel mask must reuse the original `segmentation_*.png` filename from the source `segmentation_masks/` folder for that sequence.
- Masks are stored as 8-bit PNG files with DDD17 6-class ids `0..5` and ignore label `255`.
- The content is a single-channel semantic mask, not RGB.

## Training Integration

- Use the verified `DDD17_UDA.pt.pt` checkpoint from `<PROJECT_ROOT>/weights/official/DDD17_UDA.pt.pt`.
- Load pseudolabels from the `train` split only during optimization.
- Add one auxiliary CE-style pseudolabel loss as an extra term in the existing ESS task-loss stack (`dice` + `cross_entropy`), masked with ignore label `255`.
- Keep the original ESS losses unchanged and additive.
- Leave the `valid` split out of the training loss path.

## Validation Semantics

- Pseudolabels may be generated for `valid` only for inspection, debugging, or visual QA.
- Official validation metrics must continue to use the real DDD17 valid split labels, not the pseudolabels in `data/ddd17_pseudolabels/clipseg/valid`.
- Do not substitute synthetic labels into the evaluation path.

## Experiment Matrix

| Run | Pseudolabels | Confidence score | Threshold | Aux CE weight | Notes |
| --- | --- | --- | --- | --- | --- |
| baseline | off | n/a | n/a | 0.0 | Existing ESS baseline. |
| main | on | CLIPSeg per-pixel sigmoid probability | 0.5 | 0.2 | Primary CLIPSeg pseudolabel run. |
| no-filter | on | CLIPSeg per-pixel sigmoid probability | disabled | 0.2 | Uses `train_no_filter` pseudolabels. |
| low-weight | on | CLIPSeg per-pixel sigmoid probability | 0.5 | 0.1 | Lower auxiliary pressure. |
| medium-weight | on | CLIPSeg per-pixel sigmoid probability | 0.5 | 0.3 | Mid auxiliary pressure. |

## Trainer Contract

- When `use_pseudolabels_train_b` is `false`, ESS behavior stays unchanged.
- For the DDD17 training batch with pseudolabels enabled, `batch[1][1]` is the pseudolabel tensor and `batch[1][2]` remains the real DDD17 train label tensor.
- Validation still uses the real DDD17 valid labels and its existing validation tuple shape; pseudolabel tensors are not part of the validation path.
- The trainer adds one extra event-side pseudolabel term by calling the existing task-loss path and then scaling that result by `weight_pseudolabel_loss`.
- In the current committed configs, `weight_task_loss = 1`, so this experiment sweep is effectively governed by `weight_pseudolabel_loss` in this phase.

## Smoke Check

- Unmodified `Settings(...)` parsing is currently blocked at the pseudolabel-path assertion in `config/settings.py` because both configured roots are still absent on the remote host:
  - `<PROJECT_ROOT>/data/ddd17_pseudolabels/clipseg/train`
  - `<PROJECT_ROOT>/data/ddd17_pseudolabels/clipseg/train_no_filter`
- Constrained field-resolution check for the main config reported:
  - `use_pseudolabels_train_b=True`
  - `pseudolabels_path_b=<PROJECT_ROOT>/data/ddd17_pseudolabels/clipseg/train`
  - `weight_pseudolabel_loss=0.2`

