"""Shared helpers for the vision pipeline.

The scoring functions and Gardiner canonicalisation are ported verbatim (in
behaviour) from JakeBoggs/HieroglyphBench (src/hieroglyphbench/scoring.py and
mdc.py) so our numbers are directly comparable with the published leaderboard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

VISION = Path(__file__).resolve().parent
RAW = VISION / "raw"
DATA = VISION / "data"
RESULTS = VISION / "results"
BENCH_DIR = RAW / "bench" / "test"
GLYPH_DS = RAW / "glyphds" / "Dataset"

_GARDINER = re.compile(r"^([A-Za-z]{1,2})0*([0-9]{1,3})([a-zA-Z]?)$")


def canonical_gardiner(code: str) -> str | None:
    """'aa01' -> 'Aa1', 'G017' -> 'G17', 'a14A' -> 'A14a' (HieroglyphBench mdc.py)."""
    m = _GARDINER.match(code.strip())
    if not m:
        return None
    letters, num, suffix = m.groups()
    lu = letters.upper()
    if lu in ("NL", "NU"):
        letters = lu
    elif lu == "AA":
        letters = "Aa"
    elif len(lu) == 2:
        return None
    else:
        letters = lu
    return f"{letters}{int(num)}{suffix.lower()}"


def levenshtein(a, b) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def sign_accuracy(pred: list[str], gt: list[str]) -> dict | None:
    """HieroglyphBench metric: Levenshtein over whole codes / len(gt); acc floored at 0."""
    if not gt:
        return None
    pred = pred or []
    edits = levenshtein(pred, gt)
    error_rate = edits / len(gt)
    return {
        "edits": edits,
        "pred_len": len(pred),
        "gt_len": len(gt),
        "sign_error_rate": error_rate,
        "accuracy": max(0.0, 1.0 - error_rate),
    }


def aggregate(items: list[dict]) -> dict:
    accs = [i["scores"]["accuracy"] for i in items if i.get("scores")]
    sers = [i["scores"]["sign_error_rate"] for i in items if i.get("scores")]
    return {
        "items_scored": len(accs),
        "sign_accuracy_mean": sum(accs) / len(accs) if accs else None,
        "sign_error_rate_mean": sum(sers) / len(sers) if sers else None,
        "errors": sum(1 for i in items if i.get("error")),
    }


def load_bench() -> list[dict]:
    items = []
    with open(BENCH_DIR / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
