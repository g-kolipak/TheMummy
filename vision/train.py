"""Fine-tune a small pretrained CNN on the leakage-filtered single-sign crops.

    python train.py --epochs 12 --arch resnet18

Writes models/sign_classifier.pt and results/train_summary.json. A 10% random
split of the (already test-disjoint) training crops is used for validation.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps

from classifier import IMG, MODELS, build_model, device, square_pad, to_tensor
from hbcommon import DATA, RESULTS, read_jsonl


class Crops(torch.utils.data.Dataset):
    def __init__(self, rows, classes, train: bool):
        self.rows = rows
        self.cidx = {c: i for i, c in enumerate(classes)}
        self.train = train
        self.rng = np.random.default_rng(0)
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def load(self, i):
        if i not in self.cache:
            r = self.rows[i]
            self.cache[i] = Image.open(DATA / "crops" / r["code"] / r["file"]).convert("L")
        return self.cache[i]

    def __getitem__(self, i):
        im = self.load(i)
        r = self.rows[i]
        if self.train:
            rng = self.rng
            # geometric jitter: slight rotation, random pad/scale, random crop of edges
            if rng.random() < 0.7:
                im = im.rotate(rng.uniform(-7, 7), resample=Image.BILINEAR, fillcolor=int(np.median(np.asarray(im))))
            w, h = im.size
            cx1 = int(rng.uniform(0, 0.08) * w); cy1 = int(rng.uniform(0, 0.08) * h)
            cx2 = w - int(rng.uniform(0, 0.08) * w); cy2 = h - int(rng.uniform(0, 0.08) * h)
            if cx2 - cx1 > 4 and cy2 - cy1 > 4:
                im = im.crop((cx1, cy1, cx2, cy2))
            im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.4))
            im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.3))
            if rng.random() < 0.3:
                im = im.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(rng.uniform(0, 1.0)))
            arr = square_pad(im, IMG, jitter=0.35, rng=rng)
            if rng.random() < 0.5:
                arr = arr + rng.normal(0, 0.03, arr.shape).astype(np.float32)
            arr = np.clip(arr, 0, 1)
        else:
            arr = square_pad(im, IMG)
        return arr, self.cidx[r["code"]]


def collate(b):
    xs, ys = zip(*b)
    return to_tensor(list(xs)), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--all", action="store_true", help="train on all crops (no val split)")
    args = ap.parse_args()

    rows = read_jsonl(DATA / "signs_train.jsonl")
    classes = json.loads((DATA / "classes.json").read_text())
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(rows))
    n_val = 0 if args.all else len(rows) // 10
    val_rows = [rows[i] for i in idx[:n_val]]
    tr_rows = [rows[i] for i in idx[n_val:]]
    print(f"train={len(tr_rows)} val={len(val_rows)} classes={len(classes)}")

    # mild class-balancing via sampling weights (sqrt inverse frequency)
    cnt = Counter(r["code"] for r in tr_rows)
    w = torch.tensor([1.0 / np.sqrt(cnt[r["code"]]) for r in tr_rows], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(tr_rows), replacement=True)
    dl = torch.utils.data.DataLoader(Crops(tr_rows, classes, True), batch_size=args.bs,
                                     sampler=sampler, collate_fn=collate, num_workers=0)
    vdl = torch.utils.data.DataLoader(Crops(val_rows, classes, False), batch_size=128,
                                      collate_fn=collate, num_workers=0) if val_rows else None

    dev = device()
    model = build_model(len(classes), args.arch, pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = lossf(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        rec = {"epoch": ep + 1, "train_loss": tl / n, "sec": round(time.time() - t0)}
        if vdl:
            model.eval(); c = t = 0; c3 = 0
            with torch.no_grad():
                for x, y in vdl:
                    p = model(x.to(dev)).cpu()
                    c += (p.argmax(1) == y).sum().item(); t += len(y)
                    c3 += (p.topk(3, dim=1).indices == y[:, None]).any(1).sum().item()
            rec["val_top1"] = c / t; rec["val_top3"] = c3 / t
        hist.append(rec); print(rec, flush=True)

    MODELS.mkdir(exist_ok=True)
    torch.save({"classes": classes, "arch": args.arch, "state_dict": model.state_dict()},
               MODELS / "sign_classifier.pt")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "train_summary.json").write_text(json.dumps({
        "arch": args.arch, "epochs": args.epochs, "img": IMG, "n_train": len(tr_rows),
        "n_val": len(val_rows), "n_classes": len(classes),
        "note": "val is a random 10% of the leakage-filtered training crops (same plates), "
                "NOT the benchmark; it measures crop-level classification only",
        "history": hist}, indent=2))
    print("saved", MODELS / "sign_classifier.pt")


if __name__ == "__main__":
    main()
