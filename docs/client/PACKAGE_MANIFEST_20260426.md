# 交付包清单（2026-04-26）

## 包含

- ESS 项目核心代码：
  - `config`
  - `datasets`
  - `evaluation`
  - `models`
  - `scripts`
  - `training`
  - `utils`
  - `train.py`
  - `requirements.txt`
  - `README.md`
- 正式中文交付文档：
  - `README_交付说明_20260426.md`
  - `TOP2_SOFT蒸馏方法说明_20260426.md`
  - `ESS_VLM答辩版_傻瓜式完整说明_20260426.md`
- 官方 ESS 权重：
  - `weights/official/DDD17_UDA.pt.pt`
  - `weights/official/DDD17_Semantic_supervised_events.pt`
  - `weights/official/DDD17_Semantic_supervised_events_frames.pt`
- FC-CLIP 关键权重：
  - `third_party/fc-clip/checkpoints/fcclip_convnext_large_eval_ade20k.pth`
  - `third_party/fc-clip/checkpoints/open_clip_model.safetensors`
- 本轮推荐 checkpoint 与日志：
  - `artifacts/checkpoints/top2_soft_best_w0015/Epoch_15.pt`
  - `artifacts/checkpoints/top2_soft_best_w0015/Epoch_18.pt`
  - `artifacts/checkpoints/top2_soft_best_w0015/events.out.tfevents.*`
  - `artifacts/logs/top2_best_w0015_20260426_210259.stdout.log`

## 不包含

- DDD17 原始数据集
- Cityscapes 原始数据集
- DDD17 伪标签全量数据
- `.git`
- `.cache`
- `.venv*`
- `.wandb`
- `.pytest_cache`
- `.transfer_cache`
- 临时打包目录
- 过程性 `docs/ai` 文档
- 历史中间实验的大量 checkpoint

