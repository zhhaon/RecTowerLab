"""通用工具函数。"""
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch(num_threads: int = 0) -> None:
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)
    else:
        torch.set_num_threads(max(1, (os.cpu_count() or 4)))


@contextmanager
def timer(name: str = ""):
    start = time.time()
    box = {}
    yield box
    box["elapsed"] = time.time() - start
    print("[time] {}: {:.2f}s".format(name, box["elapsed"]), flush=True)


class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=JsonEncoder)
