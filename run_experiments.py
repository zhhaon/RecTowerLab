"""MF / NCF 共 10 组对比实验的一键运行脚本。

训练方式 1（setups.real）   ：使用全部真实正负样本（有多少用多少），MSE / BCE
训练方式 2（setups.sampled）：1:1 随机负采样，MSE / BCE / BPR
模型：MF（简易矩阵分解）、NCF（仅 MLP 分支），维度均为 64，其余超参数完全一致。

用法示例：
    python run_experiments.py                       # 运行全部 10 组
    python run_experiments.py --models MF           # 只跑 MF 的 5 组
    python run_experiments.py --runs MF__real__MSE  # 只跑指定的若干组
    python run_experiments.py --aggregate-only      # 只汇总结果与画图
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from recsys import config as C
from recsys.aggregate import build_long_table, build_wide_table, collect_raw, save_tables
from recsys.data import build_dataset, build_negative_candidates, load_raw
from recsys.features import build_feature_matrices, feature_spec
from recsys.plots import make_all_figures
from recsys.train import train_run
from recsys.utils import configure_torch, get_device, save_json, set_seed

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def parse_args():
    p = argparse.ArgumentParser(description="MF vs NCF on MovieLens-100K")
    p.add_argument("--models", nargs="*", default=None, choices=list(C.MODELS))
    p.add_argument("--setups", nargs="*", default=None, choices=list(C.SETUPS.keys()))
    p.add_argument("--losses", nargs="*", default=None)
    p.add_argument("--runs", nargs="*", default=None, help="直接指定 run_id，如 MF__real__MSE")
    p.add_argument("--dim", type=int, default=C.TrainConfig.dim)
    p.add_argument("--lr", type=float, default=C.TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=C.TrainConfig.weight_decay)
    p.add_argument("--batch-size", type=int, default=C.TrainConfig.batch_size)
    p.add_argument("--max-epochs", type=int, default=C.TrainConfig.max_epochs)
    p.add_argument("--patience", type=int, default=C.TrainConfig.patience)
    p.add_argument("--eval-every", type=int, default=C.TrainConfig.eval_every)
    p.add_argument("--seed", type=int, default=C.TrainConfig.seed)
    p.add_argument("--out-dir", type=str, default=str(C.RESULT_DIR))
    p.add_argument("--overwrite", action="store_true", help="重跑已存在的实验")
    p.add_argument("--aggregate-only", action="store_true", help="不训练，只汇总与画图")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def build_run_list(args):
    runs = []
    for setup, spec in C.SETUPS.items():
        if args.setups and setup not in args.setups:
            continue
        for model in C.MODELS:
            if args.models and model not in args.models:
                continue
            allowed = C.MODEL_ALLOWED.get(model)
            if allowed and setup not in allowed["setups"]:
                continue
            for loss in spec["losses"]:
                if allowed and loss not in allowed["losses"]:
                    continue
                if args.losses and loss not in args.losses:
                    continue
                rid = C.run_id(model, setup, loss)
                if args.runs and rid not in args.runs:
                    continue
                runs.append({"run_id": rid, "model": model, "setup": setup,
                             "loss": loss, "sampling": spec["sampling"]})
    return runs


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    fig_dir = out_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)

    configure_torch(C.TrainConfig.num_threads)
    device = get_device()
    cfg = C.TrainConfig(
        dim=args.dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        seed=args.seed,
    )

    if not args.aggregate_only:
        print("device: {} | dim={} lr={} batch={} max_epochs={} patience={}".format(
            device, cfg.dim, cfg.lr, cfg.batch_size, cfg.max_epochs, cfg.patience), flush=True)

        raw_df = load_raw(C.DATA_PATH)
        ds = build_dataset(raw_df, C.TRAIN_RATIO, C.VAL_RATIO,
                           C.POS_THRESHOLD, C.NEG_THRESHOLD)
        candidates = build_negative_candidates(ds)
        run_list = build_run_list(args)

        stats = ds.stats()
        stats["split"] = "8:1:1 by time per user"
        stats["pos_rule"] = "rating > 3"
        stats["neg_rule"] = "rating < 3 (rating == 3 dropped)"

        user_feat = item_feat = None
        if any(r["model"] in C.FEATURE_MODELS for r in run_list):
            user_feat, item_feat, u_names, i_names = build_feature_matrices(C.DATA_DIR)
            stats["side_features"] = {
                "user_dim": int(user_feat.shape[1]),
                "item_dim": int(item_feat.shape[1]),
                "user_names": u_names,
                "item_names": i_names,
            }
            save_json(feature_spec(user_feat, item_feat, u_names, i_names),
                      out_dir / "feature_spec.json")
            print("side features: user {} dims, item {} dims".format(
                user_feat.shape[1], item_feat.shape[1]), flush=True)
        save_json(stats, out_dir / "dataset_stats.json")
        print("dataset stats: {}".format({k: v for k, v in stats.items()
                                          if k != "side_features"}), flush=True)

        for item in run_list:
            rid = item["run_id"]
            path = raw_dir / "{}.json".format(rid)
            if path.exists() and not args.overwrite:
                print("[skip] {} already exists".format(rid), flush=True)
                continue
            print("\n===== running {} =====".format(rid), flush=True)
            set_seed(cfg.seed)
            result = train_run(
                model_name=item["model"],
                setup=item["setup"],
                loss_name=item["loss"],
                ds=ds,
                cfg=cfg,
                sampling=item["sampling"],
                candidates=candidates,
                device=device,
                ks=cfg.ks,
                user_feat=user_feat if item["model"] in C.FEATURE_MODELS else None,
                item_feat=item_feat if item["model"] in C.FEATURE_MODELS else None,
            )
            result["run_id"] = rid
            save_json(result, path)
            print("[saved] {} -> {} ({:.1f}s)".format(rid, path, result["elapsed_sec"]), flush=True)

    runs = collect_raw(raw_dir)
    if not runs:
        print("no raw results found in {}".format(raw_dir))
        return

    tables = save_tables(runs, out_dir)
    summary = tables["test"]
    print("\n===== test set summary =====")
    print(summary.to_string(index=False))
    print("\n===== val set summary =====")
    print(tables["val"].to_string(index=False))

    if not args.no_plots:
        made = make_all_figures(runs, fig_dir)
        print("\nfigures:")
        for m in made:
            print("  -", m)

    save_json({"test": summary.to_dict(orient="records"),
               "val": tables["val"].to_dict(orient="records"),
               "long": tables["long"].to_dict(orient="records")},
              out_dir / "results_summary.json")


if __name__ == "__main__":
    main()
