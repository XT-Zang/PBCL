"""Numerically stable kernel attention shared by GeoSeg decoders."""

import torch
import torch.nn.functional as F


def spatial_linear_attention(query, key, value, eps):
    with torch.autocast(device_type=query.device.type, enabled=False):
        query = F.normalize(query.float(), dim=1, eps=eps).transpose(1, 2)
        key = F.normalize(key.float(), dim=1, eps=eps)
        value = value.float()
        denominator = query.shape[1] + torch.einsum("bnc,bc->bn", query, key.sum(dim=-1) + eps)
        denominator = denominator.abs().clamp_min(eps).copysign(denominator)
        matrix = torch.einsum("bmn,bcn->bmc", key, value)
        numerator = value.sum(dim=-1, keepdim=True) + torch.einsum("bnm,bmc->bcn", query, matrix)
        return numerator / denominator.unsqueeze(1)
