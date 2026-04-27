# Top-2 Soft Distillation Working Note

- Last Updated: `2026-04-25`

## Stage Contract

- task family: `ESS + FC-CLIP + top-2 soft distillation`
- acceptance bar: `mIoU >= 56.66`
- budget: `1 full main + 1 short ablation`
- hard path retained: `fcclip_no_filter_cropped`
- soft artifacts: `fcclip_top2_soft_cropped`
- active loader tensor geometry: `200x352`
  - this supersedes the older planning expectation of `200x346`
  - current loader path resizes to width `352` and then crops `60` bottom rows, yielding `(C, 200, 352)`

## Status

- contract: frozen
- top2 export smoke: passed
- layout verifier: passed
- loader smoke: passed
- training smoke: passed
- full main: completed-negative
- short ablation: completed-mixed

## Evidence Log

- 2026-04-25 bounded train export smoke passed.
  - export command: `PYTHONPATH=<PROJECT_ROOT> ./.venv-fcclip/bin/python scripts/vlm/export_ddd17_fcclip_top2_soft.py --dataset-root <PROJECT_ROOT>/data/ddd17_seg/data --output-root <PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_top2_soft_cropped --split train --config-file <PROJECT_ROOT>/third_party/fc-clip/configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml --opts MODEL.WEIGHTS <PROJECT_ROOT>/third_party/fc-clip/checkpoints/fcclip_convnext_large_eval_ade20k.pth MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS <PROJECT_ROOT>/third_party/fc-clip/checkpoints/open_clip_model.safetensors --max-sequences 1 --max-images-per-sequence 4 --overwrite`
  - export result: `WROTE_TOP2_SOFT split=train count=4 output_root=<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_top2_soft_cropped/train`
  - artifact check: `find <PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_top2_soft_cropped/train -type f -name '*.npz' | wc -l` -> `4`
- 2026-04-25 bounded layout verifier passed on the same subset.
  - verifier command: `PYTHONPATH=<PROJECT_ROOT> ./.venv-fcclip/bin/python scripts/vlm/verify_ddd17_fcclip_top2_soft_layout.py --dataset-root <PROJECT_ROOT>/data/ddd17_seg/data --top2-root <PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_top2_soft_cropped --split train --max-sequences 1 --max-images-per-sequence 4`
  - verifier result: `RESULT: PASS`
  - output root: `<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_top2_soft_cropped/train`
- 2026-04-25 bounded loader smoke passed on the exported subset.
  - loader file: `datasets/ddd17_events_loader.py`
  - hard root: `<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_no_filter_cropped`
  - soft root: `<PROJECT_ROOT>/data/ddd17_pseudolabels/fcclip_top2_soft_cropped`
  - smoke masks: `dir0/segmentation_masks/segmentation_00000002.png`, `dir0/segmentation_masks/segmentation_00000003.png`
  - probe result:
    - `DATASET_LEN 2`
    - `BATCH_TENSOR_COUNT 7`
    - `BATCH_ITEM 0 (2, 25, 200, 352) torch.float32`
    - `BATCH_ITEM 1 (2, 200, 352) torch.int64`
    - `BATCH_ITEM 2 (2, 200, 352) torch.int64`
    - `BATCH_ITEM 3 (2, 200, 352) torch.int64`
    - `BATCH_ITEM 4 (2, 200, 352) torch.float32`
    - `BATCH_ITEM 5 (2, 200, 352) torch.float32`
    - `BATCH_ITEM 6 (2, 200, 352) torch.int64`
  - contract note: active loader tensors are `200x352`, not `200x346`
- 2026-04-25 bounded wrapped-path smoke passed.
- 2026-04-25 bounded training smoke passed through the dedicated launcher.
  - launcher command: cd <PROJECT_ROOT> && ESS_RUN_TRAIN=1 bash scripts/vlm/run_ddd17_fcclip_top2_soft.sh smoke
  - smoke log: <PROJECT_ROOT>/log/bootstrap_ddd17/fcclip_top2_soft_smoke.stdout.log
  - train-loop evidence:
    - DDD17Events num of batches:  2 2
    - progress advanced across 2.00/2.00 Batch
    - TrainLoss=2.82 then TrainLoss=2.79
  - one-batch direct probe evidence:
    - semseg_sensor_b_pseudo_loss=2.5813934803009033
    - semseg_sensor_b_pseudo_weighted_loss=0.12906967103481293
    - semseg_sensor_b_top2_soft_loss=1.8515686988830566
    - semseg_sensor_b_top2_soft_weighted_loss=0.09257843345403671
    - FINAL_LOSS 2.8635520935058594
  - wrapper file: `datasets/wrapper_dataloader.py`
  - wrapped sensor-a result:
    - `WRAPPED_A_TENSOR_COUNT 2`
    - `WRAPPED_A_ITEM 0 (2, 3, 4) torch.float32`
    - `WRAPPED_A_ITEM 1 (2,) torch.int64`
  - wrapped sensor-b result:
    - `WRAPPED_B_TENSOR_COUNT 7`
    - `WRAPPED_B_ITEM 0 (2, 25, 200, 352) torch.float32`
    - `WRAPPED_B_ITEM 1 (2, 200, 352) torch.int64`
    - `WRAPPED_B_ITEM 2 (2, 200, 352) torch.int64`
    - `WRAPPED_B_ITEM 3 (2, 200, 352) torch.int64`
    - `WRAPPED_B_ITEM 4 (2, 200, 352) torch.float32`
    - `WRAPPED_B_ITEM 5 (2, 200, 352) torch.float32`
    - `WRAPPED_B_ITEM 6 (2, 200, 352) torch.int64`
  - wrapper impact: `wrapper_dataloader.py` now uses its generic paired-sample path for both paired branches, so the 7-tensor sensor-b train sample carries without fixed-width unpacking

- 2026-04-25 top-2 soft full main completed with a near-threshold best checkpoint.
  - export prerequisite log: <PROJECT_ROOT>/log/bootstrap_ddd17/fcclip_top2_soft_full_export_20260425_071547.log
  - chain log: <PROJECT_ROOT>/log/bootstrap_ddd17/fcclip_top2_soft_main_chain_20260425_071547.log
  - final run dir: <PROJECT_ROOT>/log/ddd17_fcclip_top2_soft/20260425-080809
  - final metrics: mIoU = 56.1993, Acc = 89.8037
  - best mIoU checkpoint: Epoch_15 = 56.6576
  - comparison versus FC-CLIP main 56.36 / 89.82:
    - final delta: -0.1607 mIoU, -0.0163 Acc
    - best-step delta: +0.2976 mIoU
  - acceptance-bar comparison: 56.6576 - 56.66 = -0.0024
  - current conclusion: near miss, not yet a clean promotion over the existing main line

- 2026-04-25 top-2 soft short ablation completed with a stronger best checkpoint but weaker final convergence.
  - run dir: <PROJECT_ROOT>/log/ddd17_fcclip_top2_soft_ablation/20260425-120124
  - final metrics: mIoU = 56.0884, Acc = 89.6736
  - best mIoU checkpoint: Epoch_15 = 56.7029
  - comparison versus top-2 main final 56.1993 / 89.8037:
    - final delta: -0.1109 mIoU, -0.1301 Acc
    - best-step mIoU delta: +0.0453
  - acceptance-bar comparison: 56.7029 - 56.66 = +0.0429
  - current conclusion: lower soft weight improves the best checkpoint, but not the final converged metric
