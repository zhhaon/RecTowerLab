"""推荐模型：简易 MF 与仅 MLP 的 NCF（NeuMF 去掉 GMF 分支）。"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _init_embedding(emb: nn.Embedding, std: float = 0.01) -> None:
    nn.init.normal_(emb.weight, mean=0.0, std=std)


class MF(nn.Module):
    """矩阵分解：score(u, i) = <p_u, q_i>。"""

    def __init__(self, n_users: int, n_items: int, dim: int = 64, **kwargs):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        _init_embedding(self.user_emb)
        _init_embedding(self.item_emb)
        self.dim = dim

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users)
        i = self.item_emb(items)
        return (u * i).sum(dim=-1)


class NCF(nn.Module):
    """仅 MLP 的 NCF：concat(p_u, q_i) -> MLP -> 1。"""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        hidden: Sequence[int] = (64, 32, 16),
        **kwargs
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        _init_embedding(self.user_emb)
        _init_embedding(self.item_emb)

        layers = []
        in_dim = 2 * dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(users), self.item_emb(items)], dim=-1)
        return self.mlp(x).squeeze(-1)


def _feature_mlp(in_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    layers = []
    for h in hidden:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    last = nn.Linear(in_dim, 1)
    nn.init.normal_(last.weight, 0.0, 0.01)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class MFLatentFeat(nn.Module):
    """特征增强 MF：隐向量 = 自由隐向量 + 侧特征线性投影，score = <u_vec, i_vec>。

    u_vec = p_u + Wu·f_u,  i_vec = q_i + Wi·f_i
    特征通过 dot-product 参与交互，维度仍为 dim=64。
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 user_feat=None, item_feat=None, **kwargs):
        super().__init__()
        if user_feat is None or item_feat is None:
            raise ValueError("MFLatentFeat requires user_feat and item_feat")
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        _init_embedding(self.user_emb)
        _init_embedding(self.item_emb)
        self.register_buffer("user_feat", torch.as_tensor(user_feat, dtype=torch.float32))
        self.register_buffer("item_feat", torch.as_tensor(item_feat, dtype=torch.float32))
        self.user_proj = nn.Linear(self.user_feat.shape[1], dim)
        self.item_proj = nn.Linear(self.item_feat.shape[1], dim)
        for proj in (self.user_proj, self.item_proj):
            nn.init.normal_(proj.weight, 0.0, 0.01)
            nn.init.zeros_(proj.bias)
        self.dim = dim

    def latent(self, users: torch.Tensor, items: torch.Tensor):
        u = self.user_emb(users) + self.user_proj(self.user_feat[users])
        i = self.item_emb(items) + self.item_proj(self.item_feat[items])
        return u, i

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u, i = self.latent(users, items)
        return (u * i).sum(dim=-1)


class MFMLPFeat(nn.Module):
    """MF + 特征旁路：score = <p_u, q_i> + MLP([user_feat, item_feat])。"""

    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 hidden: Sequence[int] = (64, 32, 16),
                 user_feat=None, item_feat=None, **kwargs):
        super().__init__()
        if user_feat is None or item_feat is None:
            raise ValueError("MFMLPFeat requires user_feat and item_feat")
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        _init_embedding(self.user_emb)
        _init_embedding(self.item_emb)
        self.register_buffer("user_feat", torch.as_tensor(user_feat, dtype=torch.float32))
        self.register_buffer("item_feat", torch.as_tensor(item_feat, dtype=torch.float32))
        self.feat_mlp = _feature_mlp(self.user_feat.shape[1] + self.item_feat.shape[1],
                                     (64, 32))
        self.dim = dim

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        cf = (self.user_emb(users) * self.item_emb(items)).sum(dim=-1)
        f = torch.cat([self.user_feat[users], self.item_feat[items]], dim=-1)
        return cf + self.feat_mlp(f).squeeze(-1)


def build_model(name: str, n_users: int, n_items: int, dim: int = 64,
                hidden: Tuple[int, ...] = (64, 32, 16),
                user_feat=None, item_feat=None) -> nn.Module:
    name = name.upper()
    if name == "MF":
        return MF(n_users, n_items, dim=dim)
    if name == "NCF":
        return NCF(n_users, n_items, dim=dim, hidden=hidden)
    if name == "MF+FEAT":
        return MFLatentFeat(n_users, n_items, dim=dim, user_feat=user_feat, item_feat=item_feat)
    if name == "MF+FEATMLP":
        return MFMLPFeat(n_users, n_items, dim=dim, hidden=hidden,
                         user_feat=user_feat, item_feat=item_feat)
    raise ValueError("unknown model: {}".format(name))
