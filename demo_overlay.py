"""Draw the fine-tuned model's real predictions (codes + boxes + confidence) on HieroglyphBench photos.

Reads the saved eval answers in runs/<run-id>/predictions_tuned.jsonl, so every box, code and
confidence shown is exactly what the model produced on that photo during evaluation.

    python demo_overlay.py unas-p20-c1-1 unas-p7-c12-2     # -> docs/demo_<id>.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "vision"))
from hbcommon import sign_accuracy  # noqa: E402

RUN = "runs/hiero-onemodel-20261102/predictions_tuned.jsonl"
BENCH_META = "vision/raw/bench/test/metadata.jsonl"
SCALE = 3
BLUE, GREEN, RED, INK, MUTED, PAPER = (42, 120, 214), (0, 131, 0), (227, 73, 72), (11, 11, 11), (119, 117, 111), (252, 252, 251)


def font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def render(case_id: str) -> Path:
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(RUN, encoding="utf-8")}
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH_META, encoding="utf-8")}
    r, m = preds[case_id], meta[case_id]
    gold = m["gardiner"]
    signs = r.get("signs") or []
    pred_codes = [s["code"] for s in signs]
    acc = sign_accuracy(pred_codes, gold)["accuracy"]

    photo = Image.open(Path(BENCH_META).parent / m["file_name"]).convert("RGB")
    W, H = photo.size
    photo = photo.resize((W * SCALE, H * SCALE), Image.LANCZOS)
    draw = ImageDraw.Draw(photo)
    f_small = font(15)
    for i, s in enumerate(signs):
        x1, y1, x2, y2 = s["box"]
        bx = [x1 * W * SCALE / 1000, y1 * H * SCALE / 1000, x2 * W * SCALE / 1000, y2 * H * SCALE / 1000]
        draw.rectangle(bx, outline=BLUE, width=3)
        draw.text((bx[0] + 4, bx[1] + 2), str(i + 1), fill=BLUE, font=f_small)

    panel_w, row_h, top = 520, 34, 96
    rows = max(len(pred_codes), len(gold))
    canvas = Image.new("RGB", (photo.width + panel_w + 40, max(photo.height, top + rows * row_h + 40) + 20), PAPER)
    canvas.paste(photo, (20, 20))
    d = ImageDraw.Draw(canvas)
    x0 = photo.width + 50
    d.text((x0, 20), f"{case_id}", fill=INK, font=font(22))
    d.text((x0, 50), f"sign accuracy {acc:.0%}  ·  {len(pred_codes)} predicted / {len(gold)} expert signs",
           fill=MUTED, font=font(15))
    d.text((x0, top - 26), "#   model (confidence)            expert", fill=MUTED, font=font(15))
    f_row = font(19)
    for i in range(rows):
        y = top + i * row_h
        p = signs[i] if i < len(signs) else None
        g = gold[i] if i < len(gold) else "-"
        ok = p is not None and p["code"] == g
        d.text((x0, y), f"{i + 1:>2}", fill=MUTED, font=f_row)
        if p is not None:
            d.text((x0 + 44, y), p["code"], fill=INK, font=f_row)
            conf = p.get("code_confidence")
            if conf is not None:
                d.text((x0 + 120, y), f"{conf:.2f}", fill=MUTED, font=f_row)
                d.rectangle([x0 + 180, y + 7, x0 + 180 + int(90 * conf), y + 17], fill=BLUE)
        d.text((x0 + 300, y), g, fill=INK, font=f_row)
        cx, cy = x0 + 392, y + 12
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=GREEN if ok else RED)
        d.text((cx + 14, y), "match" if ok else "wrong", fill=GREEN if ok else RED, font=font(15))
    out = Path("docs") / f"demo_{case_id}.png"
    out.parent.mkdir(exist_ok=True)
    canvas.save(out)
    print(f"{out}: accuracy {acc:.0%}, {len(pred_codes)} predicted vs {len(gold)} expert signs")
    return out


if __name__ == "__main__":
    for cid in sys.argv[1:]:
        render(cid)
