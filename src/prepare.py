"""Turn the PPG-DaLiA subject pickles into small, windowed arrays.

The dataset ships as Python pickles. Loading a pickle can run arbitrary code, so
this script uses a restricted unpickler that only rebuilds plain NumPy arrays,
dicts, lists and strings, and refuses anything else.

For each subject we keep only what the task needs: wrist BVP (64 Hz), wrist
3-axis accelerometer (32 Hz, upsampled to 64 Hz), the ECG-derived heart rate
label for each 8 s window (2 s shift) and the activity code.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

FS = 64
WIN = 8 * FS
SHIFT = 2 * FS

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def load_subject(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f, encoding="latin1").load()


def windows(subject_dir, sid):
    d = load_subject(subject_dir / f"{sid}.pkl")
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float64).ravel()
    acc32 = np.asarray(d["signal"]["wrist"]["ACC"], dtype=np.float64)
    acc = resample_poly(acc32, 2, 1, axis=0)  # 32 Hz to 64 Hz
    act4 = np.asarray(d["activity"]).ravel()  # 4 Hz activity codes
    hr = np.asarray(d["label"], dtype=np.float64).ravel()
    n = min(len(hr), (min(len(bvp), len(acc)) - WIN) // SHIFT + 1)
    X = np.empty((n, 4, WIN), dtype=np.float32)
    act = np.empty(n, dtype=np.int16)
    for i in range(n):
        s = i * SHIFT
        X[i, 0] = bvp[s:s + WIN]
        X[i, 1:] = acc[s:s + WIN].T
        a = act4[(s // 16):(s + WIN) // 16]  # 64 Hz index to 4 Hz index
        act[i] = np.bincount(a.astype(int)).argmax() if len(a) else 0
    return X, hr[:n].astype(np.float32), act, len(hr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path, help="folder PPG_FieldStudy")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(1, 16):
        sid = f"S{k}"
        X, y, act, nlab = windows(args.raw / sid, sid)
        np.savez(args.out / f"{sid}.npz", X=X, y=y, act=act)
        print(f"{sid}: {len(y)} windows (labels in file: {nlab}), HR {y.mean():.1f} +/- {y.std():.1f} bpm", flush=True)


if __name__ == "__main__":
    main()
