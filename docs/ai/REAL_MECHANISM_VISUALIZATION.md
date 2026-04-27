# Real Mechanism Visualization Record

## Purpose

Generate a real data-backed mechanism visualization for the ESS-VLM GitHub README.

## Inputs

- Data root: `/media/jgzn/SSD_lexar/RZ/danzi/ESS/data/ddd17_seg/data`
- FC-CLIP hard pseudo label root: `/media/jgzn/SSD_lexar/RZ/danzi/ESS/data/ddd17_pseudolabels/fcclip_no_filter_cropped/train`
- Top-2 soft root: `/media/jgzn/SSD_lexar/RZ/danzi/ESS/data/ddd17_pseudolabels/fcclip_top2_soft_cropped/train`

## Selected Samples

- `dir0:00001002`
- `dir0:00001010`

## Command

```bash
python3 scripts/vlm/make_real_mechanism_visualization.py \
  --data-root data/ddd17_seg/data \
  --hard-root data/ddd17_pseudolabels/fcclip_no_filter_cropped/train \
  --top2-root data/ddd17_pseudolabels/fcclip_top2_soft_cropped/train \
  --sample dir0:00001002 \
  --sample dir0:00001010 \
  --output log/visualization_export/real_mechanism/real_mechanism_visualization.png
```

## Interpretation

The Top-2 ambiguity heatmap is computed from FC-CLIP `p1-p2` margins. It is not attention. Brighter regions indicate smaller top-1/top-2 margins.

## Output

- GitHub asset: `assets/figures/real_mechanism_visualization.png`
- Size: `581333` bytes
- SHA256: `3C76D0BE5147F89831A4775C5D2CFEF4B9B81C2D42B8FE080ACD31282A6C99C3`

## ESS Prediction Export Status

Agreement overlay source: FC-CLIP hard pseudo label vs DDD17 label.
