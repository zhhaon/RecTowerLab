"""结果可视化：柱状图 / 折线图 / 热力图 / 训练曲线。"""
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregate import SETUP_LABEL, build_long_table, build_wide_table
from .config import run_id

METRICS = ("Recall", "HitRatio", "NDCG")
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
BAR_COLORS = {5: "#8ECAE6", 10: "#219EBC", 20: "#023047"}
MODEL_COLORS = {"MF": "#4C72B0", "NCF": "#DD8452",
                "MF+Feat": "#55A868", "MF+FeatMLP": "#8172B3"}
MODEL_STYLES = {"MF": "-", "NCF": "--", "MF+Feat": "-.", "MF+FeatMLP": ":"}


def _model_color(name):
    return MODEL_COLORS.get(name, "#7F7F7F")

_METRIC_TITLE = {
    "Recall": "Recall@K",
    "HitRatio": "HitRatio@K",
    "NDCG": "NDCG@K",
}


def _run_label(model: str, loss: str) -> str:
    return "{}-{}".format(model, loss)


def _grouped_bar(ax, runs, ks, df, metric):
    labels = [_run_label(r["model"], r["loss"]) for r in runs]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(ks))
    for j, k in enumerate(ks):
        vals = []
        for r in runs:
            sub = df[(df["run_id"] == run_id(r["model"], r["setup"], r["loss"]))
                     & (df["K"] == k) & (df["split"] == "test")]
            vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
        offset = -0.4 + width / 2 + j * width
        bars = ax.bar(x + offset, vals, width, label="K={}".format(k), color=BAR_COLORS.get(k, None))
        ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=1, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(_METRIC_TITLE[metric], fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    if metric in ("Recall", "HitRatio"):
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.25 + 1e-6))


def make_setup_figure(runs: List[Dict], df: pd.DataFrame, setup: str, out_path) -> None:
    sub_runs = [r for r in runs if r["setup"] == setup]
    sub_runs = sorted(sub_runs, key=lambda r: (r["model"], r["loss"]))
    if not sub_runs:
        return
    ks = sorted({int(k) for k in df["K"].unique()})
    fig, axes = plt.subplots(1, 3, figsize=(5.4 * 3, 4.6))
    for ax, metric in zip(axes, METRICS):
        _grouped_bar(ax, sub_runs, ks, df, metric)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title="top-K", fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Setup: {} (dim=64, test set)".format(SETUP_LABEL.get(setup, setup)),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all_runs_figure(runs: List[Dict], df: pd.DataFrame, metric: str, k: int, out_path) -> None:
    rows = []
    for r in runs:
        rid = run_id(r["model"], r["setup"], r["loss"])
        sub = df[(df["run_id"] == rid) & (df["K"] == k) & (df["split"] == "test")]
        if len(sub):
            rows.append((rid, float(sub[metric].iloc[0]), r["setup"], r["model"], r["loss"]))
    rows.sort(key=lambda t: t[1], reverse=True)
    labels = [t[0].replace("__", " | ") for t in rows]
    vals = [t[1] for t in rows]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = [_model_color(t[3]) for t in rows]
    bars = ax.barh(range(len(vals)), vals, color=colors)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("{}@{}".format(metric, k))
    ax.set_title("All runs on test set: {}@{}".format(metric, k))
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    models = []
    for t in rows:
        if t[3] not in models:
            models.append(t[3])
    handles = [plt.Rectangle((0, 0), 1, 1, color=_model_color(m)) for m in models]
    ax.legend(handles, models, title="model", loc="lower right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_metric_vs_k_figure(runs: List[Dict], df: pd.DataFrame, out_path) -> None:
    ks = sorted({int(k) for k in df["K"].unique()})
    pos = np.arange(len(ks))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    styles = MODEL_STYLES
    markers = {"MSE": "o", "BCE": "s", "BPR": "^"}
    for r in sorted(runs, key=lambda r: (r["setup"], r["model"], r["loss"])):
        rid = run_id(r["model"], r["setup"], r["loss"])
        label = "{} {}".format(rid.replace("__", " "), "(sampled)" if r["setup"] == "sampled" else "(real)")
        for ax, metric in zip(axes, METRICS):
            vals = []
            for k in ks:
                sub = df[(df["run_id"] == rid) & (df["K"] == k) & (df["split"] == "test")]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            ax.plot(pos, vals, styles.get(r["model"], "-."), marker=markers[r["loss"]],
                    label=label, linewidth=1.3, markersize=5)
            ax.set_title(_METRIC_TITLE[metric])
            ax.set_xticks(pos)
            ax.set_xticklabels(["K={}".format(k) for k in ks])
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.suptitle("Metrics vs K (test set)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_heatmap_figure(runs: List[Dict], df: pd.DataFrame, metric: str, out_path) -> None:
    ks = sorted({int(k) for k in df["K"].unique()})
    labels, mat = [], []
    for r in sorted(runs, key=lambda r: (r["setup"], r["model"], r["loss"])):
        rid = run_id(r["model"], r["setup"], r["loss"])
        row = []
        for k in ks:
            sub = df[(df["run_id"] == rid) & (df["K"] == k) & (df["split"] == "test")]
            row.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
        labels.append(rid.replace("__", " | "))
        mat.append(row)
    mat = np.asarray(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(["K={}".format(k) for k in ks])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "{:.4f}".format(mat[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > np.nanmean(mat) else "black")
    ax.set_title("{}@K on test set ({} runs)".format(metric, len(labels)))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_training_curves(runs: List[Dict], out_path) -> None:
    setups = sorted({r["setup"] for r in runs})
    fig, axes = plt.subplots(1, len(setups), figsize=(7 * len(setups), 4.4), squeeze=False)
    for ax, setup in zip(axes[0], setups):
        for r in sorted([x for x in runs if x["setup"] == setup],
                        key=lambda r: (r["model"], r["loss"])):
            epochs, vals, losses = [], [], []
            for rec in r["history"]:
                losses.append(rec["train_loss"])
                if "val" in rec:
                    epochs.append(rec["epoch"])
                    vals.append(rec["val"].get("NDCG@10", np.nan))
            ax.plot(epochs, vals, marker="o", markersize=3, linewidth=1.3,
                    label="{} | {}".format(r["model"], r["loss"]))
        ax.set_xlabel("epoch")
        ax.set_ylabel("val NDCG@10")
        ax.set_title(SETUP_LABEL.get(setup, setup))
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    fig.suptitle("Validation NDCG@10 vs epoch", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_loss_curves(runs: List[Dict], out_path) -> None:
    setups = sorted({r["setup"] for r in runs})
    fig, axes = plt.subplots(1, len(setups), figsize=(7 * len(setups), 4.4), squeeze=False)
    for ax, setup in zip(axes[0], setups):
        for r in sorted([x for x in runs if x["setup"] == setup],
                        key=lambda r: (r["model"], r["loss"])):
            epochs = [rec["epoch"] for rec in r["history"]]
            losses = [rec["train_loss"] for rec in r["history"]]
            ax.plot(epochs, losses, linewidth=1.3, label="{} | {}".format(r["model"], r["loss"]))
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.set_title(SETUP_LABEL.get(setup, setup))
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    fig.suptitle("Training loss vs epoch", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_feature_compare_figure(runs: List[Dict], df: pd.DataFrame, out_path) -> None:
    """最强设置（1:1 采样 + BCE）下：无特征 MF vs 两个特征增强变体。"""
    order = {"MF": 0, "MF+Feat": 1, "MF+FeatMLP": 2}
    sel = [r for r in runs
           if r["setup"] == "sampled" and r["loss"] == "BCE" and r["model"] in order]
    sel.sort(key=lambda r: order[r["model"]])
    if len(sel) < 2:
        return
    ks = sorted({int(k) for k in df["K"].unique()})
    fig, axes = plt.subplots(1, 3, figsize=(5.6 * 3, 4.8))
    for ax, metric in zip(axes, METRICS):
        _grouped_bar(ax, sel, ks, df, metric)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title="top-K", fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Side features in the strongest setting "
                 "(1:1 sampling + BCE, dim=64, test set)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(runs: List[Dict], out_dir) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_long_table(runs)
    made = []

    for setup in sorted({r["setup"] for r in runs}):
        p = out_dir / "metrics_by_setup_{}.png".format(setup)
        make_setup_figure(runs, df, setup, p)
        if p.exists():
            made.append(str(p))

    for metric, k in (("Recall", 10), ("HitRatio", 10), ("NDCG", 10)):
        p = out_dir / "all_runs_{}_at{}.png".format(metric.lower(), k)
        make_all_runs_figure(runs, df, metric, k, p)
        made.append(str(p))

    p = out_dir / "metrics_vs_k.png"
    make_metric_vs_k_figure(runs, df, p)
    made.append(str(p))

    p = out_dir / "feature_ablation_best_setting.png"
    make_feature_compare_figure(runs, df, p)
    if p.exists():
        made.append(str(p))

    for metric in ("Recall", "NDCG"):
        p = out_dir / "heatmap_{}.png".format(metric.lower())
        make_heatmap_figure(runs, df, metric, p)
        made.append(str(p))

    p = out_dir / "val_ndcg_curves.png"
    make_training_curves(runs, p)
    made.append(str(p))

    p = out_dir / "train_loss_curves.png"
    make_loss_curves(runs, p)
    made.append(str(p))

    return made
