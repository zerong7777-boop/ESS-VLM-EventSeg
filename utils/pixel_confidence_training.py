import torch
import torch.nn.functional as f


def confidence_weighted_cross_entropy(logits, labels, confidence_map, ignore_index=255):
    labels = labels.long()
    confidence_map = confidence_map.float().clamp_(0.0, 1.0)
    ce = f.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='none')
    valid_mask = (labels != ignore_index).float()
    pixel_weights = confidence_map * valid_mask
    weight_sum = pixel_weights.sum().clamp_min(1.0)
    return (ce * pixel_weights).sum() / weight_sum
