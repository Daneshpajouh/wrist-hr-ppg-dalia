"""Classical spectral baselines with leave-one-subject-out evaluation.

A) PPG peak: band-pass the PPG window (0.5 to 4 Hz), take the Hann-windowed,
   zero-padded power spectrum and return the strongest peak between 40 and
   200 bpm. No parameters are fitted.

B) PPG peak with motion suppression: the normalised accelerometer spectrum
   (sum over the three axes) is subtracted from the normalised PPG spectrum
   with a weight alpha, so peaks that also appear in the motion signal are
   pushed down, in the spirit of the spectral subtraction step used by
   SpaMa (Salehizadeh et al., Sensors 2016). alpha is chosen on the 14
   training subjects of each fold.

C) Method B plus simple tracking: the search is limited to plus or minus
   delta bpm around the previous estimate. alpha and delta are both chosen on
   the training subjects of each fold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from common import FS, SUBJECTS, load, mae

NFFT = 8192
FREQ = np.fft.rfftfreq(NFFT, 1 / FS) * 60.0  # bpm
BAND = (FREQ >= 40) & (FREQ <= 200)
F_BAND = FREQ[BAND]
SOS = butter(4, [0.5, 4.0], btype="band", fs=FS, output="sos")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
DELTAS = [2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]


def spectra(X):
    """Normalised power spectra in the 40 to 200 bpm band: (PPG, summed ACC)."""
    x = sosfiltfilt(SOS, X.astype(np.float64), axis=-1)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x * np.hanning(X.shape[-1])
    p = np.abs(np.fft.rfft(x, n=NFFT, axis=-1)) ** 2
    p = p[..., BAND]
    p = p / (p.max(axis=-1, keepdims=True) + 1e-12)
    return p[:, 0].astype(np.float32), p[:, 1:].sum(axis=1).astype(np.float32) / 3.0


def peak(S):
    return F_BAND[np.argmax(S, axis=-1)]


def suppressed(P, A, alpha):
    return np.clip(P - alpha * A, 0, None)


def track(S, delta):
    est = np.empty(len(S))
    prev = F_BAND[np.argmax(S[0])]
    for i, s in enumerate(S):
        m = np.abs(F_BAND - prev) <= delta
        if not np.any(s[m] > 0):
            m = np.ones_like(m)
        j = np.flatnonzero(m)[np.argmax(s[m])]
        prev = F_BAND[j]
        est[i] = prev
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = {}
    for sid in SUBJECTS:
        X, y, _ = load(args.windows, sid)
        P, A = spectra(X)
        data[sid] = (P, A, y)
        print(f"spectra {sid}", flush=True)

    # Per-subject error for every parameter value, computed once; each fold
    # then picks the value with the lowest mean error over its training subjects.
    err_b = {s: {a: mae(peak(suppressed(P, A, a)), y) for a in ALPHAS} for s, (P, A, y) in data.items()}
    err_c = {}
    for s, (P, A, y) in data.items():
        err_c[s] = {}
        for a in ALPHAS:
            S = suppressed(P, A, a)
            for d in DELTAS:
                err_c[s][(a, d)] = mae(track(S, d), y)
        print(f"grid {s}", flush=True)

    rows, preds, chosen = [], {}, {}
    for test in SUBJECTS:
        train = [s for s in SUBJECTS if s != test]
        a_b = min(ALPHAS, key=lambda a: np.mean([err_b[s][a] for s in train]))
        k_c = min(err_c[test], key=lambda k: np.mean([err_c[s][k] for s in train]))
        P, A, y = data[test]
        pa = peak(P)
        pb = peak(suppressed(P, A, a_b))
        pc = track(suppressed(P, A, k_c[0]), k_c[1])
        preds[test] = dict(y=y, ppg_peak=pa, ppg_acc_suppress=pb, ppg_acc_track=pc)
        chosen[test] = dict(alpha_b=a_b, alpha_c=k_c[0], delta_c=k_c[1])
        rows.append((test, mae(pa, y), mae(pb, y), mae(pc, y)))
        print(f"{test}: A {rows[-1][1]:.2f}  B {rows[-1][2]:.2f} (alpha {a_b})  "
              f"C {rows[-1][3]:.2f} (alpha {k_c[0]}, delta {k_c[1]})", flush=True)

    np.savez(args.out / "baseline_preds.npz",
             **{f"{s}_{k}": v for s, d in preds.items() for k, v in d.items()})
    (args.out / "baseline_params.json").write_text(json.dumps(chosen, indent=1))
    m = np.array([r[1:] for r in rows])
    print("mean A %.2f  B %.2f  C %.2f" % tuple(m.mean(axis=0)))


if __name__ == "__main__":
    main()
