import torch
import torch.nn.functional as F


def top2_soft_cross_entropy(
    logits,
    top1_id,
    top2_id,
    p1,
    p2,
    valid_mask=None,
    temperature=1.0,
    min_p1=0.0,
    min_margin=0.0,
    eps=1e-8,
):
    log_probs = F.log_softmax(logits, dim=1)
    top1_log_prob = log_probs.gather(1, top1_id.long().unsqueeze(1)).squeeze(1)
    top2_log_prob = log_probs.gather(1, top2_id.long().unsqueeze(1)).squeeze(1)

    p1 = p1.float().clamp(min=0.0)
    p2 = p2.float().clamp(min=0.0)
    if temperature != 1.0:
        teacher_logits = torch.stack([p1.clamp_min(eps).log(), p2.clamp_min(eps).log()], dim=0)
        teacher_probs = torch.softmax(teacher_logits / float(temperature), dim=0)
        w1, w2 = teacher_probs[0], teacher_probs[1]
    else:
        mass = (p1 + p2).clamp_min(eps)
        w1 = p1 / mass
        w2 = p2 / mass
    loss_map = -(w1 * top1_log_prob + w2 * top2_log_prob)

    if valid_mask is None:
        valid_mask = torch.ones_like(loss_map, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()

    gate = torch.ones_like(loss_map, dtype=torch.bool)
    if min_p1 > 0.0:
        gate = gate & (p1 >= float(min_p1))
    if min_margin > 0.0:
        gate = gate & ((p1 - p2) >= float(min_margin))
    valid_mask = valid_mask & gate

    valid_weight = valid_mask.float()
    denom = valid_weight.sum().clamp_min(1.0)
    return (loss_map * valid_weight).sum() / denom
