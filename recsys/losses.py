"""损失函数：MSE / BCE / BPR。

统一约定：模型输出 raw score（logit）。
    * MSE：sigmoid(logit) 与 0/1 标签做均方误差
    * BCE：带 logits 的二元交叉熵
    * BPR：pairwise，-log sigmoid(score_pos - score_neg)
"""
from typing import Optional

import torch
import torch.nn.functional as F

POINTWISE_LOSSES = ("MSE", "BCE")
PAIRWISE_LOSSES = ("BPR",)


def mse_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(scores), labels)


def bce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(scores, labels)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


POINTWISE_FN = {"MSE": mse_loss, "BCE": bce_loss}
PAIRWISE_FN = {"BPR": bpr_loss}


def pointwise_loss(name: str, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return POINTWISE_FN[name.upper()](scores, labels)


def pairwise_loss(name: str, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return PAIRWISE_FN[name.upper()](pos_scores, neg_scores)
