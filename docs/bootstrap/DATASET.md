# DDD17 Bootstrap Dataset Notes

## Scope

This bootstrap path is DDD17-first. It is intended to prove that the remote ESS project can recognize a usable DDD17 layout before we spend time on longer baseline runs or VLM extensions.

Important scope note: the current `config/settings_DDD17_bootstrap.yaml` still follows the upstream `ESS` UDA path, so a real `train.py` smoke run needs both:

- DDD17 for sensor B
- Cityscapes for sensor A

Current observed state on `2026-04-02`: `<PROJECT_ROOT>/data/` is still empty, so the dataset is not ready yet.

## Recommended first artifact

Stage the pre-processed DDD17 package referenced by the upstream ESS README first:

- `https://download.ifi.uzh.ch/rpg/ESS/ddd17_seg.tar.gz`

Extract it under:

```text
<PROJECT_ROOT>/data/ddd17_seg
```

Cityscapes is needed before any `train.py` smoke run that uses `config/settings_DDD17_bootstrap.yaml`, because `BaseTrainer.createDataLoaders()` builds sensor A first. This document and the verifier still focus on the DDD17 sensor-B side first so we can stage the smaller, task-specific dataset before the larger image-domain dataset.

Stage Cityscapes separately under:

```text
<PROJECT_ROOT>/data/cityscapes
```

## Expected layout

The ESS DDD17 event loader expects a root that contains `dir*` sequence folders. A bootstrap-ready sequence should look like:

```text
ddd17_seg/
  dir0/
    events.dat.t
    events.dat.xyp
    index/
      index_50ms.npy
    segmentation_masks/
      segmentation_00000001.png
      ...
  dir1/
  ...
```

Notes:

- `segmentation_masks.zip` is a useful sign that labels were copied from EV-SegNet, but the loader consumes extracted `segmentation_masks/*.png`.
- `video.mp4` may exist, but it is not required for the event-label bootstrap path.
- The current bootstrap config uses `delta_t_per_data: 50`, so `index/index_50ms.npy` is treated as required.

## Verifier

Use:

```bash
python3 scripts/bootstrap/verify_ddd17_layout.py
```

The verifier checks:

- dataset root exists
- at least one `dir*` sequence exists
- each discovered sequence has `segmentation_masks/`
- each discovered sequence has at least one `*.png` mask
- each discovered sequence has `events.dat.t`
- each discovered sequence has `events.dat.xyp`
- each discovered sequence has `index/index_50ms.npy`
- optional signs searched: `segmentation_masks.zip`, `video.mp4`

Pass/fail rule:

- `PASS`: at least one discovered sequence is complete and no required top-level check failed
- `FAIL`: missing root, no `dir*` sequences, or any discovered sequence is missing required files

Scope boundary:

- A verifier `PASS` only means the DDD17 sensor-B side is staged correctly.
- A verifier `PASS` does not mean `train.py` can start yet, because `config/settings_DDD17_bootstrap.yaml` also needs a valid Cityscapes root for sensor A.

## Bootstrap config

`config/settings_DDD17_bootstrap.yaml` is a conservative, smoke-oriented config:

- 1 epoch
- batch size 2
- DDD17 event path fixed to the project-local data root
- Cityscapes path declared explicitly because this upstream-compatible bootstrap config still needs sensor A at startup

This config is not the final production setting. It exists to keep the bootstrap path explicit and reproducible.
