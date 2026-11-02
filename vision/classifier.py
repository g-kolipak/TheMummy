"""Single-sign classifier: timm backbone fine-tuned on GlyphDataset crops."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from hbcommon import VISION

MODELS = VISION / "models"
IMG = 96


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def square_pad(im: Image.Image, size: int = IMG, jitter: float = 0.0, rng=None) -> np.ndarray:
    """Grayscale, pad to square with the image's border colour, resize to `size`.
    Returns float32 array in [0,1] of shape (size,size)."""
    g = im.convert("L")
    w, h = g.size
    arr = np.asarray(g)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    fill = int(np.median(border))
    s = int(max(w, h) * (1.0 + (rng.uniform(0, jitter) if (rng is not None and jitter) else 0.1)))
    canvas = Image.new("L", (s, s), fill)
    if rng is not None and jitter:
        ox = rng.integers(0, s - w + 1)
        oy = rng.integers(0, s - h + 1)
    else:
        ox, oy = (s - w) // 2, (s - h) // 2
    canvas.paste(g, (int(ox), int(oy)))
    canvas = canvas.resize((size, size), Image.BILINEAR)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def to_tensor(batch: list[np.ndarray]) -> torch.Tensor:
    x = torch.from_numpy(np.stack(batch))[:, None]  # (B,1,H,W)
    x = (x - 0.5) / 0.25
    return x.expand(-1, 3, -1, -1).contiguous()


def build_model(n_classes: int, name: str = "resnet18", pretrained: bool = True):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes)


class SignClassifier:
    def __init__(self, path: Path | None = None):
        path = path or (MODELS / "sign_classifier.pt")
        ck = torch.load(path, map_location="cpu")
        self.classes: list[str] = ck["classes"]
        self.model = build_model(len(self.classes), ck.get("arch", "resnet18"), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.dev = device()
        self.model.to(self.dev).eval()

    @torch.no_grad()
    def predict(self, crops: list[Image.Image], topk: int = 3) -> list[list[tuple[str, float]]]:
        if not crops:
            return []
        out = []
        for i in range(0, len(crops), 128):
            x = to_tensor([square_pad(c) for c in crops[i:i + 128]]).to(self.dev)
            p = torch.softmax(self.model(x), dim=1).cpu()
            v, ix = p.topk(min(topk, p.shape[1]), dim=1)
            for vs, ixs in zip(v, ix):
                out.append([(self.classes[j], float(s)) for s, j in zip(vs, ixs)])
        return out
