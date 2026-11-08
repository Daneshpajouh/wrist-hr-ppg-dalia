"""Compact deep model for heart rate from wrist PPG and accelerometer.

Each 8 s window (PPG plus 3 accelerometer axes, 64 Hz, 4 x 512 samples) goes
through a small 1-D CNN that gives one 128-value embedding. A one-layer GRU
then reads the embeddings of the current window and the previous ones (up to
CONTEXT windows, so at most 8 + 2 * (CONTEXT - 1) seconds of past signal) and
predicts the heart rate of the current window. The model is causal: it never
looks at future windows, so it could run online on a watch.

Evaluation is leave-one-subject-out: for test subject k, one other subject is
held out for early stopping and the remaining 13 are used for training.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import SUBJECTS, load, mae

torch.set_num_threads(4)
HR_MEAN, HR_STD = 90.0, 20.0


def prep(X):
    """PPG: z-score per window. ACC: raw units are 1/64 g, so divide by 64."""
    X = X.copy()
    p = X[:, 0]
    X[:, 0] = (p - p.mean(axis=1, keepdims=True)) / (p.std(axis=1, keepdims=True) + 1e-6)
    X[:, 1:] /= 64.0
    return X


class Encoder(nn.Module):
    def __init__(self, ch=(32, 32, 64, 64, 128)):
        super().__init__()
        layers, c_in = [], 4
        for c in ch:
            layers += [nn.Conv1d(c_in, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU(),
                       nn.Conv1d(c, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU(),
                       nn.MaxPool1d(2)]
            c_in = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (N, 4, 512) -> (N, 128)
        return self.net(x).mean(dim=-1)


class HRNet(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.enc = Encoder()
        self.context = context
        self.gru = nn.GRU(128, 64, batch_first=True) if context > 1 else None
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64 if context > 1 else 128, 1))

    def forward(self, x):  # (B, L, 4, 512) -> (B,)
        B, L = x.shape[:2]
        z = self.enc(x.reshape(B * L, *x.shape[2:])).reshape(B, L, -1)
        if self.gru is not None:
            z, _ = self.gru(z)
        return self.head(z[:, -1]).squeeze(-1)


def seq_index(n, L):
    """Index matrix (n, L): row i holds windows i-L+1..i, clamped at the start."""
    idx = np.arange(n)[:, None] - np.arange(L - 1, -1, -1)[None, :]
    return np.clip(idx, 0, None)


class Pool:
    """All windows of a group of subjects, with sequence indices that never cross subjects."""

    def __init__(self, data, sids, L):
        Xs, ys, idx, off = [], [], [], 0
        for s in sids:
            X, y, _ = data[s]
            Xs.append(X)
            ys.append(y)
            idx.append(seq_index(len(y), L) + off)
            off += len(y)
        self.X = torch.from_numpy(np.concatenate(Xs))
        self.y = torch.from_numpy(np.concatenate(ys))
        self.idx = torch.from_numpy(np.concatenate(idx))

    def batch(self, rows, device, augment=False):
        ii = self.idx[rows]
        x = self.X[ii.reshape(-1)].reshape(*ii.shape, *self.X.shape[1:])
        if augment:  # random gain on each channel group, same for the whole sequence
            g = torch.empty(x.shape[0], 1, 4, 1).uniform_(0.8, 1.2)
            x = x * g
        return x.to(device), self.y[rows].to(device)


@torch.no_grad()
def predict(model, pool, device, bs=512):
    model.eval()
    out = []
    for s in range(0, len(pool.y), bs):
        x, _ = pool.batch(torch.arange(s, min(s + bs, len(pool.y))), device)
        out.append(model(x).cpu())
    return torch.cat(out).numpy() * HR_STD + HR_MEAN


def run_fold(data, test, args, device, log):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    k = SUBJECTS.index(test)
    val = SUBJECTS[(k + 1) % len(SUBJECTS)]
    train = [s for s in SUBJECTS if s not in (test, val)]
    tr, va, te = (Pool(data, g, args.context) for g in (train, [val], [test]))
    model = HRNet(args.context).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    steps = args.epochs * (len(tr.y) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    best, best_state, best_ep = 1e9, None, -1
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr.y))
        t0, tot = time.time(), 0.0
        for b in range(len(tr.y) // args.bs):
            x, y = tr.batch(perm[b * args.bs:(b + 1) * args.bs], device, augment=True)
            loss = nn.functional.smooth_l1_loss(model(x), (y - HR_MEAN) / HR_STD, beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        v = mae(predict(model, va, device), va.y.numpy())
        log(f"{test} ep {ep + 1:2d} loss {tot / (b + 1):.4f} val({val}) MAE {v:.2f} [{time.time() - t0:.0f} s]")
        if v < best:
            best, best_ep = v, ep + 1
            best_state = {n: t.detach().clone() for n, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    p = predict(model, te, device)
    return p, te.y.numpy(), dict(val_subject=val, best_epoch=best_ep, val_mae=best,
                                 params=sum(t.numel() for t in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--name", default="cnn_gru")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"device {device}, args {vars(args)}")
    data = {}
    for s in SUBJECTS:
        X, y, act = load(args.windows, s)
        data[s] = (prep(X), y, act)
    for test in args.subjects:
        t0 = time.time()
        p, y, info = run_fold(data, test, args, device, log)
        info.update(test_mae=mae(p, y), seconds=round(time.time() - t0))
        np.save(out / f"{test}_pred.npy", p)
        (out / f"{test}_info.json").write_text(json.dumps(info, indent=1))
        log(f"{test} TEST MAE {info['test_mae']:.2f} bpm (best epoch {info['best_epoch']}, {info['seconds']} s)")


if __name__ == "__main__":
    main()
