"""全量排序评估：Recall@K / HitRatio@K / NDCG@K。

评估协议：
    * 对每个用户对全部候选物品打分并排序（候选集 = 全部 1682 个物品，
      但屏蔽掉该用户在训练集（评估验证集时）/ 训练集+验证集（评估测试集时）中的正样本）
    * ground truth 为该用户验证集 / 测试集中的正样本
    * HitRatio@K：top-K 中命中至少一个正样本记为 1，否则 0，对全部用户求平均
"""
from typing import Dict, List, Sequence

import numpy as np
import torch

from .data import Dataset
from .models import MF


@torch.no_grad()
def score_all_items(model: torch.nn.Module, n_users: int, n_items: int,
                    device: torch.device, chunk: int = 262144) -> np.ndarray:
    """返回 (n_users, n_items) 的打分矩阵。"""
    model.eval()
    scores = np.empty((n_users, n_items), dtype=np.float32)
    users_all = torch.arange(n_users, dtype=torch.long, device=device).repeat_interleave(n_items)
    items_all = torch.arange(n_items, dtype=torch.long, device=device).repeat(n_users)
    total = n_users * n_items
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        out = model(users_all[start:end], items_all[start:end])
        scores.reshape(-1)[start:end] = out.detach().cpu().numpy()
    return scores


def _ndcg(hits_positions: Sequence[int], n_relevant: int, k: int) -> float:
    dcg = sum(1.0 / np.log2(p + 2.0) for p in hits_positions if p < k)
    ideal = sum(1.0 / np.log2(j + 2.0) for j in range(min(n_relevant, k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def evaluate(
    model: torch.nn.Module,
    ds: Dataset,
    split: str = "test",
    ks: Sequence[int] = (5, 10, 20),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    ks = sorted(ks)
    max_k = max(ks)

    scores = score_all_items(model, ds.n_users, ds.n_items, device)

    # 屏蔽用户已交互的正样本（训练集，以及在评估测试集时一并屏蔽验证集）
    for u in range(ds.n_users):
        banned = list(ds.train_pos[u])
        if split == "test":
            banned += ds.val_pos[u]
        if banned:
            scores[u, np.asarray(banned, dtype=np.int64)] = -np.inf

    topk = np.argpartition(-scores, max_k - 1, axis=1)[:, :max_k]
    row = np.take_along_axis(scores, topk, axis=1)
    order = np.argsort(-row, axis=1)
    topk = np.take_along_axis(topk, order, axis=1)[:, :max_k]

    ground_truth: List[List[int]] = ds.test_pos if split == "test" else ds.val_pos

    n_users_eval = 0
    recall = {k: 0.0 for k in ks}
    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    n_hits = {k: 0.0 for k in ks}

    for u, rel in enumerate(ground_truth):
        if not rel:
            continue
        n_users_eval += 1
        rel_set = set(rel)
        rank_list = topk[u]
        hits_at = {k: [] for k in ks}
        for pos, item in enumerate(rank_list, start=1):
            if int(item) in rel_set:
                for k in ks:
                    if pos <= k:
                        hits_at[k].append(pos - 1)
        for k in ks:
            h = len(hits_at[k])
            recall[k] += h / len(rel_set)
            hit[k] += 1.0 if h > 0 else 0.0
            ndcg[k] += _ndcg(hits_at[k], len(rel_set), k)
            n_hits[k] += h

    if n_users_eval == 0:
        raise RuntimeError("no user has ground truth in split {}".format(split))

    metrics: Dict[str, float] = {"n_users_eval": float(n_users_eval)}
    for k in ks:
        metrics["Recall@{}".format(k)] = recall[k] / n_users_eval
        metrics["HitRatio@{}".format(k)] = hit[k] / n_users_eval
        metrics["NDCG@{}".format(k)] = ndcg[k] / n_users_eval
    return metrics
