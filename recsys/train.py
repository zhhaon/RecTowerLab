"""两套训练管线：
  1) real    —— 使用全部真实正负样本，逐点损失 MSE / BCE
  2) sampled —— 1:1 随机负采样，逐点损失 MSE / BCE 或 pairwise 损失 BPR
"""
import copy
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import TrainConfig
from .data import Dataset, sample_1_1_pairs
from .evaluate import evaluate
from .losses import bpr_loss, pointwise_loss
from .models import build_model
from .utils import set_seed


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, device=device)


def _train_epoch_pointwise_real(model, ds: Dataset, loss_name: str, cfg: TrainConfig,
                                optimizer, device, rng) -> float:
    n = len(ds.train_users)
    perm = rng.permutation(n)
    total, n_batch = 0.0, 0
    for start in range(0, n, cfg.batch_size):
        idx = perm[start:start + cfg.batch_size]
        users = _to_tensor(ds.train_users[idx], device)
        items = _to_tensor(ds.train_items[idx], device)
        labels = _to_tensor(ds.train_labels[idx], device)
        optimizer.zero_grad()
        scores = model(users, items)
        loss = pointwise_loss(loss_name, scores, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_batch += 1
    return total / max(1, n_batch)


def _train_epoch_sampled(model, ds: Dataset, loss_name: str, cfg: TrainConfig,
                         optimizer, device, rng, candidates) -> float:
    pu, pi, nu, ni = sample_1_1_pairs(ds, candidates, rng)
    perm = rng.permutation(len(pu))
    total, n_batch = 0.0, 0
    is_bpr = loss_name.upper() == "BPR"
    for start in range(0, len(perm), cfg.batch_size):
        idx = perm[start:start + cfg.batch_size]
        optimizer.zero_grad()
        if is_bpr:
            users_p = _to_tensor(pu[idx], device)
            items_p = _to_tensor(pi[idx], device)
            users_n = _to_tensor(nu[idx], device)
            items_n = _to_tensor(ni[idx], device)
            pos_scores = model(users_p, items_p)
            neg_scores = model(users_n, items_n)
            loss = bpr_loss(pos_scores, neg_scores)
        else:
            users = np.concatenate([pu[idx], nu[idx]])
            items = np.concatenate([pi[idx], ni[idx]])
            labels = np.concatenate([
                np.ones(len(idx), dtype=np.float32),
                np.zeros(len(idx), dtype=np.float32),
            ])
            scores = model(_to_tensor(users, device), _to_tensor(items, device))
            loss = pointwise_loss(loss_name, scores, _to_tensor(labels, device))
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_batch += 1
    return total / max(1, n_batch)


def train_run(
    model_name: str,
    setup: str,
    loss_name: str,
    ds: Dataset,
    cfg: TrainConfig,
    sampling: str,
    candidates,
    device: torch.device,
    ks: Sequence[int] = (5, 10, 20),
    verbose: bool = True,
    user_feat=None,
    item_feat=None,
) -> Dict:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = build_model(model_name, ds.n_users, ds.n_items, dim=cfg.dim,
                        hidden=cfg.ncf_hidden,
                        user_feat=user_feat, item_feat=item_feat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[Dict] = []
    best_score, best_epoch, best_state = -np.inf, -1, None
    wait = 0
    start_time = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        if sampling == "real":
            train_loss = _train_epoch_pointwise_real(model, ds, loss_name, cfg, optimizer, device, rng)
        else:
            train_loss = _train_epoch_sampled(model, ds, loss_name, cfg, optimizer, device, rng, candidates)

        record = {"epoch": epoch, "train_loss": train_loss}
        if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            val_metrics = evaluate(model, ds, split="val", ks=ks, device=device)
            record["val"] = {k: v for k, v in val_metrics.items() if k != "n_users_eval"}
            key = "NDCG@10" if 10 in ks else "NDCG@{}".format(max(ks))
            if record["val"].get(key, -np.inf) > best_score:
                best_score = float(record["val"][key])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
        history.append(record)

        if verbose:
            msg = "[{}] epoch {:3d} train_loss {:.5f}".format(
                "{}-{}-{}".format(model_name, setup, loss_name), epoch, train_loss)
            if "val" in record:
                msg += " | val NDCG@10 {:.4f}".format(record["val"].get("NDCG@10", float("nan")))
            msg += " | best epoch {}".format(best_epoch)
            print(msg, flush=True)

        if wait >= cfg.patience:
            if verbose:
                print("  early stop at epoch {} (patience {})".format(epoch, cfg.patience), flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, ds, split="val", ks=ks, device=device)
    test_metrics = evaluate(model, ds, split="test", ks=ks, device=device)
    elapsed = time.time() - start_time

    return {
        "model": model_name,
        "setup": setup,
        "loss": loss_name,
        "sampling": sampling,
        "dim": cfg.dim,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "seed": cfg.seed,
        "ncf_hidden": list(cfg.ncf_hidden),
        "best_epoch": best_epoch,
        "best_val_ndcg10": best_score,
        "epochs_run": len(history),
        "n_train_samples_per_epoch": (
            ds.n_interactions if sampling == "real" else 2 * int((ds.train_labels == 1).sum())
        ),
        "neg_pool_rule": (
            "all real pos/neg in train split"
            if sampling == "real"
            else "all items minus the user's train positives (val/test positives allowed)"
        ),
        "side_features": (
            None if user_feat is None else
            "u.user(age,gender,occupation) + u.item(19 genres,release year) "
            "[u_dim={}, i_dim={}]".format(user_feat.shape[1], item_feat.shape[1])
        ),
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "n_users_eval"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "n_users_eval"},
        "history": history,
        "elapsed_sec": elapsed,
    }
