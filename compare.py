"""Score every system's saved predictions on the same cases with the same functions.

  image cases: HieroglyphBench sign accuracy (Levenshtein over Gardiner codes / gold length,
               floored at 0, averaged over the 30 columns; ported from HieroglyphBench).
               Predicted codes = first token of each non-empty answer line.
  text cases : bench.py rules: clean_output -> NFC/whitespace normalize -> character edit
               distance; CER = total edits / total reference characters; exact match.

    python compare.py results/frontier_claude-opus-5-5_medium.jsonl \
        runs/hiero-onemodel-20261102/predictions_tuned.jsonl
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vision"))
from bench import clean_output, edit_distance, normalize  # noqa: E402
from hbcommon import sign_accuracy  # noqa: E402

IMAGE_CASES = "vision/data/vlm_bench.jsonl"
TEXT_CASES = "data/mix_text_dev.jsonl"

# Prices for cost per 1,000 cases (USD per 1M input / output tokens), from each provider's pricing
# page on 2026-11-02 (Fireworks: docs.fireworks.ai/serverless/pricing, standard tier).
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "kimi-k3": (3.00, 15.00),
    "qwen3p8-max": (2.00, 6.00),
    "deepseek-v4p1-flash": (0.30, 1.20),
    "glm-5p3-flash": (0.15, 0.50),
    # Ours: Fireworks Serverless Training in-session sampling for Muse Glimmer 30B (prefill / sample).
    "hiero-onemodel": (1.96, 4.88),
}


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def gold_cases():
    img = {r["id"]: r["gardiner"] for r in read_jsonl(IMAGE_CASES)}
    txt = {}
    for i, r in enumerate(read_jsonl(TEXT_CASES)):
        txt[r.get("id", f"textdev-{i:03d}")] = r["messages"][-1]["content"]
    return img, txt


def codes_of(answer: str) -> list[str]:
    return [ln.split()[0] for ln in answer.splitlines() if ln.strip()]


def score(path, img_gold, txt_gold):
    preds = {r["id"]: r for r in read_jsonl(path)}
    accs, edits, ref_chars, exact, n_txt, capped = [], 0, 0, 0, 0, []
    for cid, gold in img_gold.items():
        ans = (preds.get(cid) or {}).get("answer_text") or ""
        accs.append(sign_accuracy(codes_of(ans), gold)["accuracy"])
    for cid, gold in txt_gold.items():
        ans = (preds.get(cid) or {}).get("answer_text") or ""
        pred, ref = normalize(clean_output(ans)), normalize(gold)
        d = edit_distance(pred, ref)
        edits += d
        ref_chars += len(ref)
        capped.append(min(1.0, d / max(len(ref), 1)))  # one runaway line cannot exceed 100%
        exact += pred == ref
        n_txt += 1
    rows = list(preds.values())
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    out = {
        "system": Path(path).as_posix(),
        "image_cases": len(img_gold),
        "image_missing": sum(1 for c in img_gold if c not in preds),
        "sign_accuracy": sum(accs) / len(accs) if accs else None,
        "text_cases": n_txt,
        "text_missing": sum(1 for c in txt_gold if c not in preds),
        "text_cer": edits / ref_chars if ref_chars else None,
        "text_cer_line_capped_mean": sum(capped) / len(capped) if capped else None,
        "text_exact": exact / n_txt if n_txt else None,
        "errors": sum(1 for r in rows if r.get("error")),
        "median_latency_s": statistics.median(lat) if lat else None,
    }
    tin = sum(r.get("input_tokens") or 0 for r in rows)
    tout = sum(r.get("output_tokens") or 0 for r in rows)
    model = next((m for m in PRICES if m in Path(path).as_posix()), None)
    if model and rows:
        pi, po = PRICES[model]
        out["cost_per_1000_cases_usd"] = (tin * pi + tout * po) / 1e6 / len(rows) * 1000
    return out


def main():
    img_gold, txt_gold = gold_cases()
    results = [score(p, img_gold, txt_gold) for p in sys.argv[1:]]
    for r in results:
        print(json.dumps(r, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/compare.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
