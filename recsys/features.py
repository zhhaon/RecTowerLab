"""用户侧 / 电影侧特征构建（来自 ml-100k/u.user、u.item、u.genre）。

用户特征（23 维）：年龄(z-score) + 性别 + 职业 one-hot(21)
电影特征（21 维）：19 个类型 one-hot + 上映年份(z-score) + 年份缺失标记

zip code 属于高基数标识、与偏好关联弱，未使用；title / IMDb URL 为标识类字段，未使用。
标准化只用特征表自身统计量，不涉及任何评分或数据划分信息。
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0:
        sd = 1.0
    return (x - mu) / sd


def load_genre_names(path) -> List[str]:
    """读取 u.genre（genre|id），按 id 排序返回 19 个类型名。"""
    rows = []
    with open(str(path), "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, idx = line.split("|")[:2]
            rows.append((int(idx), name))
    return [name for _, name in sorted(rows)]


def build_user_features(path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(str(path), sep="|", header=None,
                     names=["uid", "age", "gender", "occupation", "zip"],
                     encoding="latin-1")
    df = df.sort_values("uid").reset_index(drop=True)
    age = _zscore(df["age"].to_numpy())[:, None]
    gender = (df["gender"].str.upper() == "M").to_numpy(dtype=np.float64)[:, None]
    occ = df["occupation"].astype(str)
    categories = sorted(occ.unique())
    occ_onehot = np.zeros((len(df), len(categories)), dtype=np.float64)
    occ_onehot[np.arange(len(df)), [categories.index(v) for v in occ]] = 1.0
    feats = np.concatenate([age, gender, occ_onehot], axis=1).astype(np.float32)
    names = ["age_z", "gender_male"] + ["occ_{}".format(c) for c in categories]
    return feats, names


def build_item_features(path, genre_names: Sequence[str] = ()) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(str(path), sep="|", header=None, encoding="latin-1")
    df = df.sort_values(0).reset_index(drop=True)
    # 列 5..23 为 19 个类型标记（unknown, Action, ..., Western）
    genres = df.iloc[:, 5:24].to_numpy(dtype=np.float64)
    year = pd.to_datetime(df[2], format="%d-%b-%Y", errors="coerce").dt.year
    miss = year.isna().to_numpy(dtype=np.float64)[:, None]
    year_filled = year.fillna(year.mean()).to_numpy(dtype=np.float64)
    year_z = _zscore(year_filled)[:, None]
    feats = np.concatenate([genres, year_z, miss], axis=1).astype(np.float32)
    names = list(genre_names) if genre_names else []
    if len(names) != genres.shape[1]:
        names = ["genre_{}".format(i) for i in range(genres.shape[1])]
    else:
        names = ["genre_{}".format(g) for g in names]
    names += ["release_year_z", "release_year_missing"]
    return feats, names


def build_feature_matrices(data_dir) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """返回 (user_feat[n_users, D_u], item_feat[n_items, D_i], 用户特征名, 电影特征名)。

    行索引与 0 基的用户 / 物品 ID 对齐（u.user、u.item 的 id 均为 1..N 连续编号）。
    """
    data_dir = Path(data_dir)
    genre_names = load_genre_names(data_dir / "u.genre")
    user_feat, user_names = build_user_features(data_dir / "u.user")
    item_feat, item_names = build_item_features(data_dir / "u.item", genre_names)
    return user_feat, item_feat, user_names, item_names


def feature_spec(user_feat: np.ndarray, item_feat: np.ndarray,
                 user_names: Sequence[str], item_names: Sequence[str]) -> Dict:
    """侧特征的元信息，便于复现与排查。"""
    return {
        "user_feature_names": list(user_names),
        "item_feature_names": list(item_names),
        "user_feature_dim": int(np.asarray(user_feat).shape[1]),
        "item_feature_dim": int(np.asarray(item_feat).shape[1]),
        "notes": {
            "user": "u.user: age z-scored, gender male=1, occupation one-hot",
            "item": "u.item: 19 genres multi-hot, release year z-scored, missing-year flag",
            "unused": "zip code / title / IMDb url / video release date",
        },
    }
