"""Build image -> Gardiner-codes (+ boxes) chat JSONL for VLM fine-tuning on Fireworks.

Training items are column crops cut from the GlyphDataset plates exactly the way
HieroglyphBench cuts its test columns (same column clustering, padding and reading
order), but only from regions that do NOT overlap any HieroglyphBench test crop.
Test items are the 30 HieroglyphBench images themselves.

Target format, one sign per line, top to bottom (boxes normalised to 0-1000 of the crop):
    O4 20 2 698 64
    M17 521 69 773 192

    python3 make_vlm_data.py      # writes data/vlm_train.jsonl, data/vlm_bench.jsonl
"""
from __future__ import annotations

import base64
import io
import json
import random

from PIL import Image

from hbcommon import BENCH_DIR, DATA, GLYPH_DS, load_bench, read_jsonl, write_jsonl
from prep_data import COL_SPACING, PAD, cluster_columns, intersects, parse_plate

SYSTEM = (
    "You read ancient Egyptian hieroglyphs from photographs. The image shows one column of "
    "signs. List every sign from top to bottom, one per line, as its Gardiner code followed "
    "by its bounding box x1 y1 x2 y2, with coordinates scaled to 0-1000 of the image width "
    "and height. Reply with only these lines."
)
PROMPT = "Read the signs in this column."
WINDOWS = (8, 12, 16, 20)   # signs per training crop
STRIDE = 4
MARGIN = 4                  # extra exclusion margin around test crops (px)
SEED = 0


def png_data_uri(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def target(boxes: list[tuple[int, int, int, int, str]], w: int, h: int) -> str:
    def nx(v):
        return max(0, min(1000, round(v * 1000 / w)))

    def ny(v):
        return max(0, min(1000, round(v * 1000 / h)))
    return "\n".join(f"{c} {nx(x1)} {ny(y1)} {nx(x2)} {ny(y2)}" for x1, y1, x2, y2, c in boxes)


def row(im: Image.Image, boxes, extra: dict) -> dict:
    w, h = im.size
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_data_uri(im)}},
                {"type": "text", "text": PROMPT},
            ]},
            {"role": "assistant", "content": target(boxes, w, h)},
        ],
        **extra,
    }


def main() -> None:
    bench_items = read_jsonl(DATA / "bench_items.jsonl")      # from prep_data.py
    test_rects: dict[str, list[tuple]] = {}
    for b in bench_items:
        x1, y1, x2, y2 = b["rect"]
        test_rects.setdefault(b["plate"], []).append((x1 - MARGIN, y1 - MARGIN, x2 + MARGIN, y2 + MARGIN))

    train, seen = [], set()
    per_plate = {}
    for loc in sorted((GLYPH_DS / "Manual" / "Locations").glob("*.txt"), key=lambda p: int(p.stem)):
        plate = loc.stem
        good = [s for s in parse_plate(loc) if s["code"]]
        with Image.open(GLYPH_DS / "Pictures" / f"egyptianTexts{plate}.jpg") as src:
            src = src.convert("RGB")
            W, H = src.size
            n_plate = 0
            for col in cluster_columns(good, COL_SPACING):
                for size in WINDOWS:
                    if len(col) < size:
                        continue
                    for start in range(0, len(col) - size + 1, STRIDE):
                        win = col[start:start + size]
                        x1 = max(0, min(s["x1"] for s in win) - PAD)
                        y1 = max(0, min(s["y1"] for s in win) - PAD)
                        x2 = min(W, max(s["x2"] for s in win) + PAD)
                        y2 = min(H, max(s["y2"] for s in win) + PAD)
                        rect = (x1, y1, x2, y2)
                        if rect in seen:
                            continue
                        crop_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        if any(intersects(crop_box, r) for r in test_rects.get(plate, [])):
                            continue
                        seen.add(rect)
                        boxes = [(s["x1"] - x1, s["y1"] - y1, s["x2"] - x1, s["y2"] - y1, s["code"]) for s in win]
                        train.append(row(src.crop(rect), boxes,
                                         {"id": f"p{plate}-{x1}-{y1}-{x2}-{y2}", "gardiner": [b[4] for b in boxes]}))
                        n_plate += 1
            per_plate[plate] = n_plate

    random.Random(SEED).shuffle(train)
    write_jsonl(DATA / "vlm_train.jsonl", train)

    bench_meta = {b["id"]: b for b in load_bench()}
    test = []
    for b in sorted(bench_items, key=lambda r: r["id"]):
        with Image.open(BENCH_DIR / bench_meta[b["id"]]["file_name"]) as im:
            im = im.convert("RGB")
            boxes = [tuple(gb[:4]) + (gb[4],) for gb in b["gt_boxes"]]
            test.append(row(im, boxes, {"id": b["id"], "gardiner": bench_meta[b["id"]]["gardiner"]}))
    write_jsonl(DATA / "vlm_bench.jsonl", test)

    n_signs = sum(len(r["gardiner"]) for r in train)
    print(f"train crops: {len(train)} ({n_signs} sign instances) per plate: {per_plate}")
    print(f"bench items: {len(test)}; codes match bench GT: "
          f"{sum(r['gardiner'] == [ln.split()[0] for ln in r['messages'][2]['content'].splitlines()] for r in test)}/{len(test)}")


if __name__ == "__main__":
    main()
