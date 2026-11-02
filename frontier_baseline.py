"""Frontier-model baseline on the exact evaluation cases the fine-tuned model is scored on.

  image cases : vision/data/vlm_bench.jsonl   (30 HieroglyphBench column photos)
  text cases  : data/mix_text_dev.jsonl       (300 held-out codes -> transliteration lines)

Same system prompts as the fine-tuned model. Text cases also get the same 5 solved examples
(data/shots.jsonl) that bench.py gives every few-shot system. Every raw answer, token count and
latency is saved; scoring is done by compare.py with the same functions for every system.

    source .env   # ANTHROPIC_API_KEY
    python frontier_baseline.py --model claude-opus-5-5 --effort medium
"""
import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from bench import user_message

IMAGE_CASES = "vision/data/vlm_bench.jsonl"
TEXT_CASES = "data/mix_text_dev.jsonl"
SHOTS = "data/shots.jsonl"


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def image_request(row):
    system = row["messages"][0]["content"]
    parts = row["messages"][1]["content"]
    url = next(p["image_url"]["url"] for p in parts if p["type"] == "image_url")
    header, data = url.split(",", 1)
    media_type = header.split(":", 1)[1].split(";", 1)[0]
    text = next(p["text"] for p in parts if p["type"] == "text")
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
        {"type": "text", "text": text},
    ]
    return system, [{"role": "user", "content": content}]


def text_request(row, shots):
    system = row["messages"][0]["content"]
    msgs = []
    for s in shots:
        msgs.append({"role": "user", "content": user_message(s["hieroglyphs"])})
        msgs.append({"role": "assistant", "content": s["transcription"]})
    msgs.append({"role": "user", "content": row["messages"][1]["content"]})
    return system, msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="first N cases of each kind (smoke test)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    shots = read_jsonl(SHOTS)
    images = read_jsonl(IMAGE_CASES)
    texts = read_jsonl(TEXT_CASES)
    for i, r in enumerate(texts):
        r.setdefault("id", f"textdev-{i:03d}")
    if args.limit:
        images, texts = images[: args.limit], texts[: args.limit]
    cases = [("image", r) for r in images] + [("text", r) for r in texts]

    client = anthropic.Anthropic(max_retries=6)

    def one(case):
        kind, row = case
        system, messages = image_request(row) if kind == "image" else text_request(row, shots)
        rec = {"id": row["id"], "kind": kind, "gold": row["messages"][-1]["content"]}
        t0 = time.time()
        try:
            resp = client.messages.create(
                model=args.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_config={"effort": args.effort},
            )
            rec["answer_text"] = "".join(b.text for b in resp.content if b.type == "text")
            rec["stop_reason"] = resp.stop_reason
            rec["input_tokens"] = resp.usage.input_tokens
            rec["output_tokens"] = resp.usage.output_tokens
        except anthropic.BadRequestError as e:
            rec["error"] = f"bad_request: {e.message}"
        except anthropic.RateLimitError as e:
            rec["error"] = f"rate_limited: {e.message}"
        except anthropic.APIStatusError as e:
            rec["error"] = f"api_{e.status_code}: {e.message}"
        except anthropic.APIConnectionError as e:
            rec["error"] = f"connection: {e}"
        rec["latency_s"] = round(time.time() - t0, 3)
        return rec

    out = Path(args.out or f"results/frontier_{args.model}_{args.effort}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as pool, open(out, "w", encoding="utf-8") as f:
        for rec in pool.map(one, cases):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if done % 25 == 0 or done == len(cases):
                print(f"{done}/{len(cases)} done", flush=True)
    errs = sum(1 for l in open(out) if '"error"' in l)
    print(f"wrote {out} ({len(cases)} cases, {errs} errors)")


if __name__ == "__main__":
    main()
