"""Collect per-subject results into CSV files and draw the figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SUBJECTS, load, mae

ACTIVITIES = {0: "transition", 1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling",
              5: "driving", 6: "lunch", 7: "walking", 8: "working"}
METHODS = [
    ("ppg_peak", "A. PPG spectral peak"),
    ("ppg_acc_suppress", "B. PPG peak, ACC suppression"),
    ("ppg_acc_track", "C. B plus tracking"),
    ("cnn_only", "D. CNN, single window"),
    ("cnn_gru", "E. CNN + GRU, 8-window context"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("figures", type=Path)
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    base = np.load(args.results / "baseline_preds.npz")

    preds, truth, acts = {}, {}, {}
    for s in SUBJECTS:
        _, y, act = load(args.windows, s)
        truth[s], acts[s] = y, act
        for key, _ in METHODS:
            if key.startswith("cnn"):
                f = args.results / key / f"{s}_pred.npy"
                if f.exists():
                    preds[(key, s)] = np.load(f)
            else:
                preds[(key, s)] = base[f"{s}_{key}"]
                assert np.allclose(base[f"{s}_y"], y)

    methods = [(k, n) for k, n in METHODS if all((k, s) in preds for s in SUBJECTS)]
    rows = []
    for s in SUBJECTS:
        r = {"subject": s, "windows": len(truth[s])}
        for k, _ in methods:
            r[k] = round(mae(preds[(k, s)], truth[s]), 2)
        rows.append(r)
    df = pd.DataFrame(rows)
    mean = {"subject": "mean", "windows": int(df["windows"].sum())}
    sd = {"subject": "sd", "windows": ""}
    for k, _ in methods:
        mean[k] = round(df[k].mean(), 2)
        sd[k] = round(df[k].std(ddof=0), 2)
    df = pd.concat([df, pd.DataFrame([mean, sd])], ignore_index=True)
    df.to_csv(args.results / "results.csv", index=False)
    print(df.to_string(index=False))

    # Error by activity, pooled over all windows of all held-out subjects.
    arows = []
    for a, name in ACTIVITIES.items():
        r = {"activity": name, "windows": int(sum((acts[s] == a).sum() for s in SUBJECTS))}
        for k, _ in methods:
            err = np.concatenate([np.abs(preds[(k, s)] - truth[s])[acts[s] == a] for s in SUBJECTS])
            r[k] = round(float(err.mean()), 2)
        arows.append(r)
    adf = pd.DataFrame(arows)
    adf.to_csv(args.results / "results_by_activity.csv", index=False)
    print(adf.to_string(index=False))

    # Figure 1: per-subject MAE for every method.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(methods)
    x = np.arange(len(SUBJECTS))
    colours = ["#b9b9b9", "#8c8c8c", "#5a5a5a", "#6fa8dc", "#1f5fa8"]
    for i, (k, n) in enumerate(methods):
        v = df[k].iloc[:len(SUBJECTS)].astype(float).values
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, v, w, label=f"{n} (mean {v.mean():.2f})",
               color=colours[i % len(colours)])
    ax.set_xticks(x, SUBJECTS)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("PPG-DaLiA, leave-one-subject-out: heart rate error per held-out subject")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.figures / "mae_per_subject.png", dpi=150)

    # Figure 2: estimated against reference heart rate over a whole session.
    s = "S7"
    t = np.arange(len(truth[s])) * 2 / 60
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, truth[s], color="black", lw=1.2, label="ECG reference")
    ax.plot(t, preds[("ppg_acc_track", s)], color="#999999", lw=0.8,
            label=f"C. classical ({mae(preds[('ppg_acc_track', s)], truth[s]):.2f} bpm MAE)")
    if ("cnn_gru", s) in preds:
        ax.plot(t, preds[("cnn_gru", s)], color="#1f5fa8", lw=0.9,
                label=f"E. CNN + GRU ({mae(preds[('cnn_gru', s)], truth[s]):.2f} bpm MAE)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("heart rate (bpm)")
    ax.set_title(f"Held-out subject {s}: whole session (about 2.5 h of daily activities)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.figures / f"trace_{s}.png", dpi=150)

    info = {}
    for k, _ in methods:
        if k.startswith("cnn"):
            info[k] = {s: json.loads((args.results / k / f"{s}_info.json").read_text()) for s in SUBJECTS}
    (args.results / "cnn_fold_info.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
