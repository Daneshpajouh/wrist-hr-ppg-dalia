"""Shared helpers: data loading and the evaluation metric."""
from pathlib import Path

import numpy as np

SUBJECTS = [f"S{k}" for k in range(1, 16)]
FS = 64


def load(windows_dir, sid):
    d = np.load(Path(windows_dir) / f"{sid}.npz")
    return d["X"], d["y"], d["act"]


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))
