"""全局配置。

所有实验（MF / NCF）共享同一套超参数，保证对比公平：
    * 隐向量维度 dim = 64
    * 优化器 Adam，学习率 lr
    * batch size、最大 epoch 数、早停 patience 完全一致
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ml-100k"
DATA_PATH = DATA_DIR / "u.data"
RESULT_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULT_DIR / "raw"
FIGURE_DIR = RESULT_DIR / "figures"

# 数据划分
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

POS_THRESHOLD = 3  # rating > 3 -> 正样本
NEG_THRESHOLD = 3  # rating < 3 -> 负样本，rating == 3 视为中性并剔除


@dataclass
class TrainConfig:
    """单次训练的超参数（两组训练方式、两个模型都使用同一份配置）。"""

    dim: int = 64                                    # 隐向量维度
    ncf_hidden: Tuple[int, ...] = (64, 32, 16)       # 仅 MLP 的隐层结构
    lr: float = 1e-3
    batch_size: int = 1024
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 8                                # 验证集 NDCG@10 早停
    seed: int = 2024
    ks: Tuple[int, ...] = (5, 10, 20)
    eval_every: int = 1
    num_threads: int = 0                             # 0 表示使用 torch 默认


# 10 组基础实验：2 个模型 × （真实负样本 2 种损失 + 随机负采样 3 种损失）
MODELS = ("MF", "NCF", "MF+Feat", "MF+FeatMLP")

# 追加实验：只在当前最强设置（1:1 随机负采样 + BCE）下验证侧特征的增益
MODEL_ALLOWED = {
    "MF+Feat": {"setups": ("sampled",), "losses": ("BCE",)},
    "MF+FeatMLP": {"setups": ("sampled",), "losses": ("BCE",)},
}
FEATURE_MODELS = ("MF+Feat", "MF+FeatMLP")

SETUPS = {
    # 训练方式 1：使用全部真实正负样本（有多少用多少），逐点损失
    "real": {
        "label": "Real negatives (all pos & neg)",
        "losses": ("MSE", "BCE"),
        "sampling": "real",
    },
    # 训练方式 2：1:1 随机负采样
    "sampled": {
        "label": "1:1 random negative sampling",
        "losses": ("MSE", "BCE", "BPR"),
        "sampling": "uniform_1_1",
    },
}


def run_id(model: str, setup: str, loss: str) -> str:
    return "{}__{}__{}".format(model, setup, loss)
