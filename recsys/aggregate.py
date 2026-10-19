"""把实验的原始 json 结果汇总为表格。"""
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import run_id

SETUP_LABEL = {"real": "real-neg", "sampled": "sampled-1:1"}
SPLITS = ("val", "test")


def collect_raw(raw_dir, verbose: bool = True) -> List[Dict]:
    """读取 raw 目录下的结果，并校验文件名 / run_id / 模型字段是否自洽。

    不自洽的文件（例如文件名写 MF+Feat 但 model 字段是 MF）会被跳过并告警，
    避免同一 run 在汇总表中出现重复行、或图表取到错误数值。
    """
    raw_dir = Path(raw_dir)
    runs: List[Dict] = []
    seen = {}
    for path in sorted(raw_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            run = json.load(f)
        try:
            derived = run_id(run["model"], run["setup"], run["loss"])
        except KeyError as exc:
            if verbose:
                print("[warn] skip {}: missing field {}".format(path.name, exc), flush=True)
            continue
        stored = run.get("run_id", derived)
        if stored != derived or path.stem != derived:
            if verbose:
                print("[warn] skip {}: run_id/model mismatch (file stem '{}', stored run_id '{}', "
                      "derived from fields '{}')".format(path.name, path.stem, stored, derived),
                      flush=True)
            continue
        if derived in seen:
            if verbose:
                print("[warn] skip {}: duplicate of {}".format(path.name, seen[derived]), flush=True)
            continue
        seen[derived] = path.name
        runs.append(run)
    return runs


def _split_keys(run: Dict) -> List[int]:
    keys = [k for k in run["test_metrics"] if k.startswith("NDCG@")]
    return sorted(int(k.split("@")[1]) for k in keys)


def build_long_table(runs: List[Dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rid = run_id(run["model"], run["setup"], run["loss"])
        for split in SPLITS:
            metrics = run["{}_metrics".format(split)]
            for k in _split_keys(run):
                rows.append({
                    "run_id": rid,
                    "model": run["model"],
                    "setup": run["setup"],
                    "setup_label": SETUP_LABEL.get(run["setup"], run["setup"]),
                    "loss": run["loss"],
                    "split": split,
                    "K": k,
                    "Recall": metrics["Recall@{}".format(k)],
                    "HitRatio": metrics["HitRatio@{}".format(k)],
                    "NDCG": metrics["NDCG@{}".format(k)],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["run_id", "split", "K"])
        df = df.sort_values(["setup", "model", "loss", "split", "K"]).reset_index(drop=True)
    return df


def build_wide_table(runs: List[Dict], split: str = "test") -> pd.DataFrame:
    rows = []
    for run in runs:
        rid = run_id(run["model"], run["setup"], run["loss"])
        row = {
            "run_id": rid,
            "model": run["model"],
            "setup": run["setup"],
            "setup_label": SETUP_LABEL.get(run["setup"], run["setup"]),
            "loss": run["loss"],
            "best_epoch": run["best_epoch"],
            "epochs_run": run["epochs_run"],
            "n_train_samples_per_epoch": run["n_train_samples_per_epoch"],
            "elapsed_sec": round(run["elapsed_sec"], 2),
        }
        metrics = run["{}_metrics".format(split)]
        for k in _split_keys(run):
            row["Recall@{}".format(k)] = metrics["Recall@{}".format(k)]
            row["HitRatio@{}".format(k)] = metrics["HitRatio@{}".format(k)]
            row["NDCG@{}".format(k)] = metrics["NDCG@{}".format(k)]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["run_id"])
        df = df.sort_values(["setup", "model", "loss"]).reset_index(drop=True)
    return df


def save_tables(runs: List[Dict], out_dir) -> Dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = build_long_table(runs)
    test_wide = build_wide_table(runs, "test")
    val_wide = build_wide_table(runs, "val")
    long_df.to_csv(out_dir / "results_long.csv", index=False, encoding="utf-8-sig")
    test_wide.to_csv(out_dir / "results_test.csv", index=False, encoding="utf-8-sig")
    val_wide.to_csv(out_dir / "results_val.csv", index=False, encoding="utf-8-sig")
    return {"long": long_df, "test": test_wide, "val": val_wide}
