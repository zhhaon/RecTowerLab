"""数据读取、按时间划分与负采样。

约定（与需求一致）：
    * 仅使用 ml-100k/u.data（用户ID | 物品ID | 评分 | 时间戳）
    * rating > 3 -> 真实正样本，rating < 3 -> 真实负样本，rating == 3 -> 中性，剔除
    * 对每个用户按时间戳先后 8:1:1 划分为训练 / 验证 / 测试
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    n_users: int
    n_items: int
    # 训练集交互（已编码为 0 基索引）：真实正负样本，(user, item, label) 三个等长数组
    train_users: np.ndarray
    train_items: np.ndarray
    train_labels: np.ndarray
    # 训练集中的正样本（用于随机负采样与候选集屏蔽）
    train_pos: List[List[int]]
    # 验证 / 测试的正样本（评估时作为 ground truth）
    val_pos: List[List[int]]
    test_pos: List[List[int]]

    @property
    def n_interactions(self) -> int:
        return int(len(self.train_users))

    def stats(self) -> Dict[str, float]:
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_train": int(len(self.train_users)),
            "n_train_pos": int((self.train_labels == 1).sum()),
            "n_train_neg": int((self.train_labels == 0).sum()),
            "n_val_pos": int(sum(len(x) for x in self.val_pos)),
            "n_test_pos": int(sum(len(x) for x in self.test_pos)),
        }


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(
        str(path),
        sep="\t",
        header=None,
        names=["user", "item", "rating", "timestamp"],
        dtype={"user": np.int64, "item": np.int64, "rating": np.int64, "timestamp": np.int64},
    )
    return df


def build_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    pos_threshold: int = 3,
    neg_threshold: int = 3,
) -> Dataset:
    """按用户、按时间 8:1:1 划分，并转换为正负样本。

    划分基于用户完整的评分序列（不做任何预过滤），按时间先后切分为
    8:1:1 的训练 / 验证 / 测试三段；随后在每段内部把 rating > 3 记为
    正样本、rating < 3 记为负样本，rating == 3 的中性评分直接剔除。
    """
    df = df.sort_values(["user", "timestamp", "item"], kind="mergesort").reset_index(drop=True)
    n_users = int(df["user"].max())
    n_items = int(df["item"].max())

    train_pos: List[List[int]] = [[] for _ in range(n_users)]
    val_pos: List[List[int]] = [[] for _ in range(n_users)]
    test_pos: List[List[int]] = [[] for _ in range(n_users)]

    tr_u, tr_i, tr_y = [], [], []

    for uid, group in df.groupby("user", sort=True):
        u = int(uid) - 1
        items = (group["item"].to_numpy() - 1).astype(np.int64)
        ratings = group["rating"].to_numpy()
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # 保证验证 / 测试集尽量非空（每用户至少 20 条评分，不会退化）
        if n_train + n_val >= n:
            n_val = max(0, n - 1 - n_train)
        seg = [
            (items[:n_train], ratings[:n_train]),
            (items[n_train:n_train + n_val], ratings[n_train:n_train + n_val]),
            (items[n_train + n_val:], ratings[n_train + n_val:]),
        ]
        for (seg_items, seg_ratings), bucket in zip(seg, (0, 1, 2)):
            for it, rt in zip(seg_items, seg_ratings):
                if rt > pos_threshold:
                    label = 1
                elif rt < neg_threshold:
                    label = 0
                else:
                    continue  # rating == 3 中性样本，直接剔除
                if bucket == 0:
                    tr_u.append(u)
                    tr_i.append(int(it))
                    tr_y.append(label)
                    if label == 1:
                        train_pos[u].append(int(it))
                elif bucket == 1:
                    if label == 1:
                        val_pos[u].append(int(it))
                else:
                    if label == 1:
                        test_pos[u].append(int(it))

    ds = Dataset(
        n_users=n_users,
        n_items=n_items,
        train_users=np.asarray(tr_u, dtype=np.int64),
        train_items=np.asarray(tr_i, dtype=np.int64),
        train_labels=np.asarray(tr_y, dtype=np.float32),
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
    )
    return ds


def build_negative_candidates(ds: Dataset) -> List[np.ndarray]:
    """为每个用户预计算可用于随机负采样的物品池。

    池 = 全部物品 - 该用户**训练集**正样本。验证 / 测试集的正样本可以落在池中：
    若把它们也从池中剔除，等于用未来的标签信息参与训练构造，属于数据泄露。
    """
    all_items = np.arange(ds.n_items, dtype=np.int64)
    candidates: List[np.ndarray] = []
    for u in range(ds.n_users):
        banned = np.asarray(sorted(set(ds.train_pos[u])), dtype=np.int64)
        mask = np.ones(ds.n_items, dtype=bool)
        if banned.size:
            mask[banned] = False
        candidates.append(all_items[mask])
    return candidates


def sample_1_1_pairs(
    ds: Dataset,
    candidates: List[np.ndarray],
    rng: np.random.Generator,
    users: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1:1 随机负采样：每个正样本配一个随机负样本。

    返回 (pos_users, pos_items, neg_users, neg_items)，四个等长数组。
    """
    pu, pi, nu, ni = [], [], [], []
    user_ids = range(ds.n_users) if users is None else users
    for u in user_ids:
        pos = np.asarray(ds.train_pos[u], dtype=np.int64)
        if pos.size == 0:
            continue
        cand = candidates[u]
        replace = cand.size < pos.size
        neg = rng.choice(cand, size=pos.size, replace=replace)
        pu.append(np.full(pos.size, u, dtype=np.int64))
        pi.append(pos)
        nu.append(np.full(pos.size, u, dtype=np.int64))
        ni.append(neg.astype(np.int64))
    return (
        np.concatenate(pu),
        np.concatenate(pi),
        np.concatenate(nu),
        np.concatenate(ni),
    )
