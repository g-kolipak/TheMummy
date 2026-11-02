"""Large open-model baselines on Fireworks serverless, on the exact cases our model is scored on.

Same cases, prompts and 5 solved text examples as frontier_baseline.py; OpenAI-compatible
chat completions on https://api.fireworks.ai/inference/v1. Scored by compare.py.

    source .env   # FIREWORKS_API_KEY
    python fireworks_baseline.py --model accounts/fireworks/models/kimi-k3
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from bench import user_message

IMAGE_CASES = "vision/data/vlm_bench.jsonl"
TEXT_CASES = "data/mix_text_dev.jsonl"
SHOTS = "data/shots.jsonl"


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def messages_for(kind, row, shots):
    system = {"role": "system", "content": row["messages"][0]["content"]}
    if kind == "image":
        return [system, {"role": "user", "content": row["messages"][1]["content"]}]  # image_url + text parts
    msgs = [system]
    for s in shots:
        msgs.append({"role": "user", "content": user_message(s["hieroglyphs"])})
        msgs.append({"role": "assistant", "content": s["transcription"]})
    msgs.append({"role": "user", "content": row["messages"][1]["content"]})
    return msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--reasoning-effort", default=None, help="low|medium|high, passed through if set")
    p.add_argument("--provider", default="fireworks", choices=["fireworks", "openai"])
    args = p.parse_args()

    import os
    if args.provider == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=6, timeout=600)
    else:
        client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                        api_key=os.environ["FIREWORKS_API_KEY"], max_retries=6, timeout=600)
    shots = read_jsonl(SHOTS)
    images = read_jsonl(IMAGE_CASES)
    texts = read_jsonl(TEXT_CASES)
    for i, r in enumerate(texts):
        r.setdefault("id", f"textdev-{i:03d}")
    if args.limit:
        images, texts = images[: args.limit], texts[: args.limit]
    cases = [("image", r) for r in images] + [("text", r) for r in texts]

    def one(case):
        kind, row = case
        rec = {"id": row["id"], "kind": kind, "gold": row["messages"][-1]["content"]}
        t0 = time.time()
        try:
            if args.provider == "openai":  # reasoning models: max_completion_tokens, no temperature
                kw = {"max_completion_tokens": args.max_tokens}
                if args.reasoning_effort:
                    kw["reasoning_effort"] = args.reasoning_effort
            else:
                kw = {"max_tokens": args.max_tokens, "temperature": 0}
                if args.reasoning_effort:
                    kw["extra_body"] = {"reasoning_effort": args.reasoning_effort}
            resp = client.chat.completions.create(
                model=args.model, messages=messages_for(kind, row, shots), **kw)
            rec["answer_text"] = resp.choices[0].message.content or ""
            rec["finish_reason"] = resp.choices[0].finish_reason
            rec["input_tokens"] = resp.usage.prompt_tokens
            rec["output_tokens"] = resp.usage.completion_tokens
        except Exception as e:  # recorded per case; compare.py counts errors
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["latency_s"] = round(time.time() - t0, 3)
        return rec

    short = args.model.split("/")[-1]
    out = Path(f"results/{args.provider}_{short}{'_' + args.reasoning_effort if args.reasoning_effort else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as pool, open(out, "w", encoding="utf-8") as f:
        for rec in pool.map(one, cases):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0 or done == len(cases):
                print(f"{short}: {done}/{len(cases)}", flush=True)
    errs = sum(1 for l in open(out) if '"error"' in l)
    print(f"wrote {out} ({len(cases)} cases, {errs} errors)")


if __name__ == "__main__":
    main()
