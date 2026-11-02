"""Build the training/eval manifests and handle train/test leakage.

HieroglyphBench's 30 test columns are rectangular crops of the very same
Piankoff plates (Franken GlyphDataset) that the single-sign training crops come
from. We re-run the bench's own column construction (ported from
HieroglyphBench/unas.py) on the plate annotations to recover the exact crop
rectangle of every test item, verify it against the published PNG size, and
then EXCLUDE from training every annotated sign whose box intersects any test
rectangle. No training crop is ever visible in a test image.

Outputs (vision/data/):
  bench_items.jsonl  - id, plate, crop rect, GT codes, GT boxes (crop coords)
  signs_train.jsonl  - plate, box, code for every allowed training sign
  signs_excluded.jsonl - the signs removed because they lie in a test crop
  crops/<code>/*.png - training crops extracted from the plates
  classes.json       - class list
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from hbcommon import DATA, GLYPH_DS, canonical_gardiner, load_bench, write_jsonl

PAD = 10          # bench crop padding
CHUNK = 20        # bench chunk size
COL_SPACING = 70  # bench column spacing
MIN_SIGNS = 8


def parse_plate(loc_path: Path) -> list[dict]:
    signs = []
    for line in loc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p for p in line.strip().split(",") if p != ""]
        if len(parts) < 5:
            continue
        name, coords = parts[0], parts[1:5]
        m = re.match(r"\d+_(.+)\.png$", name)
        if not m:
            continue
        raw = m.group(1)
        code = canonical_gardiner(raw)
        try:
            x1, y1, x2, y2 = map(int, coords)
        except ValueError:
            continue
        signs.append({"file": name, "raw": raw, "code": code, "x1": x1, "y1": y1,
                      "x2": x2, "y2": y2, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2})
    return signs


def cluster_columns(signs, min_spacing):
    """Ported from HieroglyphBench unas.py (columns right-to-left, signs by cy)."""
    if not signs:
        return []
    xs = [s["cx"] for s in signs]
    lo, hi = min(xs), max(xs)
    bin_w = 10
    nbins = int((hi - lo) // bin_w) + 1
    hist = [0] * nbins
    for x in xs:
        hist[int((x - lo) // bin_w)] += 1
    win = max(1, (min_spacing // 2) // bin_w)
    sm = [sum(hist[max(0, i - win):i + win + 1]) for i in range(nbins)]
    span = max(1, min_spacing // bin_w)
    peaks = []
    for i in range(nbins):
        a, b = max(0, i - span), min(nbins, i + span + 1)
        if sm[i] == max(sm[a:b]) and sm[i] > 0:
            pos = lo + i * bin_w + bin_w / 2
            if not peaks or pos - peaks[-1] >= min_spacing:
                peaks.append(pos)
    grouped = {p: [] for p in peaks}
    for s in signs:
        grouped[min(peaks, key=lambda p: abs(p - s["cx"]))].append(s)
    cols = [c for c in grouped.values() if c]

    def width(c):
        return max(s["x2"] for s in c) - min(s["x1"] for s in c)
    widths = sorted(width(c) for c in cols)
    med = widths[len(widths) // 2] if widths else 0
    final = []
    for c in cols:
        k = max(1, round(width(c) / med)) if med else 1
        if k <= 1:
            final.append(c)
            continue
        lo_x = min(s["cx"] for s in c)
        bw = (max(s["cx"] for s in c) - lo_x) / k or 1
        subs = {}
        for s in c:
            subs.setdefault(min(k - 1, int((s["cx"] - lo_x) // bw)), []).append(s)
        final.extend(subs[i] for i in sorted(subs) if subs[i])
    for c in final:
        c.sort(key=lambda s: s["cy"])
    final.sort(key=lambda c: -sum(s["cx"] for s in c) / len(c))
    return final


def chunk(col, size):
    chunks = [col[i:i + size] for i in range(0, len(col), size)]
    if len(chunks) >= 2 and len(chunks[-1]) < size // 2:
        chunks[-2].extend(chunks.pop())
    return chunks


def intersects(s, r):
    return not (s["x2"] <= r[0] or s["x1"] >= r[2] or s["y2"] <= r[1] or s["y1"] >= r[3])


def main():
    bench = {b["id"]: b for b in load_bench()}
    locs = GLYPH_DS / "Manual" / "Locations"
    pics = GLYPH_DS / "Pictures"
    all_signs = {}        # plate -> signs
    rects = {}            # bench id -> (plate, rect)
    bench_rows = []
    for loc in sorted(locs.glob("*.txt"), key=lambda p: int(p.stem)):
        plate = loc.stem
        signs = parse_plate(loc)
        all_signs[plate] = signs
        good = [s for s in signs if s["code"]]
        with Image.open(pics / f"egyptianTexts{plate}.jpg") as im:
            W, H = im.size
        for ci, col in enumerate(cluster_columns(good, COL_SPACING), 1):
            for chi, ch in enumerate(chunk(col, CHUNK), 1):
                if len(ch) < MIN_SIGNS:
                    continue
                eid = f"unas-p{plate}-c{ci}-{chi}"
                if eid not in bench:
                    continue
                x1 = max(0, min(s["x1"] for s in ch) - PAD)
                y1 = max(0, min(s["y1"] for s in ch) - PAD)
                x2 = min(W, max(s["x2"] for s in ch) + PAD)
                y2 = min(H, max(s["y2"] for s in ch) + PAD)
                rect = (x1, y1, x2, y2)
                with Image.open(DATA.parent / "raw" / "bench" / "test" / bench[eid]["file_name"]) as bim:
                    bsize = bim.size
                codes = [s["code"] for s in ch]
                ok_size = (x2 - x1, y2 - y1) == bsize
                ok_codes = codes == bench[eid]["gardiner"]
                rects[eid] = (plate, rect)
                bench_rows.append({
                    "id": eid, "plate": plate, "rect": list(rect),
                    "size_match": ok_size, "codes_match": ok_codes,
                    "gardiner": bench[eid]["gardiner"],
                    "gt_boxes": [[s["x1"] - x1, s["y1"] - y1, s["x2"] - x1, s["y2"] - y1, s["code"]]
                                 for s in ch],
                })
    missing = sorted(set(bench) - set(rects))
    print(f"matched {len(rects)}/{len(bench)} bench items; missing: {missing}")
    print("size mismatches:", [r["id"] for r in bench_rows if not r["size_match"]])
    print("code mismatches:", [r["id"] for r in bench_rows if not r["codes_match"]])
    write_jsonl(DATA / "bench_items.jsonl", sorted(bench_rows, key=lambda r: r["id"]))

    # leakage exclusion: any sign intersecting any test rect (with extra margin)
    MARGIN = 4
    train_rows, excl_rows = [], []
    for plate, signs in all_signs.items():
        test_rects = [(x1 - MARGIN, y1 - MARGIN, x2 + MARGIN, y2 + MARGIN)
                      for p, (x1, y1, x2, y2) in rects.values() if p == plate]
        for s in signs:
            hit = [eid for eid, (p, r) in rects.items()
                   if p == plate and intersects(s, (r[0] - MARGIN, r[1] - MARGIN, r[2] + MARGIN, r[3] + MARGIN))]
            row = {"plate": plate, "file": s["file"], "code": s["code"] or "UNKNOWN",
                   "box": [s["x1"], s["y1"], s["x2"], s["y2"]]}
            if hit:
                row["in_test"] = hit
                excl_rows.append(row)
            else:
                train_rows.append(row)
    write_jsonl(DATA / "signs_train.jsonl", train_rows)
    write_jsonl(DATA / "signs_excluded.jsonl", excl_rows)
    print(f"signs total={len(train_rows) + len(excl_rows)} train={len(train_rows)} "
          f"excluded(in test crops)={len(excl_rows)}")

    # extract training crops
    crops_dir = DATA / "crops"
    counts = Counter()
    for plate in all_signs:
        with Image.open(pics / f"egyptianTexts{plate}.jpg") as im:
            im = im.convert("RGB")
            for r in train_rows:
                if r["plate"] != plate:
                    continue
                d = crops_dir / r["code"]
                d.mkdir(parents=True, exist_ok=True)
                im.crop(tuple(r["box"])).save(d / r["file"])
                counts[r["code"]] += 1
    classes = sorted(counts)
    (DATA / "classes.json").write_text(json.dumps(classes))
    print(f"{len(classes)} classes, {sum(counts.values())} crops written to {crops_dir}")
    test_codes = Counter(c for r in bench_rows for c in r["gardiner"])
    unseen = {c: n for c, n in test_codes.items() if c not in counts}
    print(f"test sign tokens={sum(test_codes.values())}; tokens whose class has NO training "
          f"crop after exclusion: {sum(unseen.values())} ({unseen})")


if __name__ == "__main__":
    main()
