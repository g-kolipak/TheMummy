"""
Text side of the project: builds the codes -> transliteration splits from bbaw_egyptian and
defines the scoring rules (clean_output, normalize, edit_distance) that compare.py uses for
every system. The `run` / `report` commands below are the original single-task benchmark
runner; the submitted comparison uses frontier_baseline.py / fireworks_baseline.py + compare.py.

Hieroglyph reading benchmark: Gardiner sign codes -> Egyptological transliteration.

Data: bbaw_egyptian (Thesaurus Linguae Aegyptiae excerpt, BBAW, CC BY-SA 4.0).

Steps:
  1) python bench.py prepare
       -> data/test.jsonl (300 held-out rows), data/shots.jsonl (5 prompt examples),
          data/train.jsonl, data/ft_train.jsonl (upload this to Fireworks to fine-tune)

  2) python bench.py run --name frontier --provider openai --model <frontier-model> --in-price X --out-price Y
     python bench.py run --name frontier --provider anthropic --model <frontier-model> --in-price X --out-price Y
     python bench.py run --name open_base --provider fireworks --model accounts/fireworks/models/<model> --in-price X --out-price Y
     python bench.py run --name open_finetuned --provider fireworks --model <your-finetuned-model-id> --zero-shot --in-price X --out-price Y
       (prices are USD per 1M input/output tokens, from each provider's pricing page)

  3) python bench.py report      -> prints table and writes results/results.md

API keys are read from environment variables:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, FIREWORKS_API_KEY
Never commit keys. Put them in a .env file (git-ignored) or export them in your shell.
"""

import argparse
import json
import os
import random
import statistics
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "data"
RESULTS_DIR = "results"
SEED = 42
N_TEST = 300
N_SHOTS = 5
MAX_TRAIN = 20000  # cap so the fine-tune finishes within the hackathon

SYSTEM_PROMPT = (
    "You are an expert Egyptologist. You will be given a line of Middle Egyptian "
    "hieroglyphs written as Gardiner sign codes (Manuel de Codage style). Convert it "
    "into its standard Egyptological transliteration, following the conventions shown "
    "in the examples. Reply with only the transliteration on a single line, nothing else."
)


# ---------------------------------------------------------------- data

def user_message(hieroglyphs):
    return f"Hieroglyphs: {hieroglyphs}\nTransliteration:"


def load_raw_rows():
    """Try Hugging Face first, then the dataset's raw JSON source."""
    try:
        from datasets import load_dataset
        for name in ("bbaw_egyptian", "phiwi/bbaw_egyptian"):
            try:
                ds = load_dataset(name, split="train")
                print(f"Loaded {len(ds)} rows from Hugging Face ({name})")
                return [dict(r) for r in ds]
            except Exception as e:
                print(f"  could not load {name}: {e}")
    except ImportError:
        print("  `datasets` not installed, using raw JSON")
    url = "https://phiwi.github.io/all.json"
    print(f"Downloading raw JSON from {url}")
    with urllib.request.urlopen(url) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} rows from raw JSON")
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(_args):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = load_raw_rows()

    clean, seen = [], set()
    for r in rows:
        h = (r.get("hieroglyphs") or "").strip()
        t = (r.get("transcription") or "").strip()
        if not h or not t:
            continue  # only about a third of rows have hieroglyph encodings
        if (h, t) in seen:
            continue
        seen.add((h, t))
        clean.append({
            "hieroglyphs": h,
            "transcription": t,
            "translation": (r.get("translation") or "").strip(),
        })

    random.Random(SEED).shuffle(clean)
    test = clean[:N_TEST]

    # Leakage guard: drop any training row whose hieroglyph line also appears in test.
    test_lines = {r["hieroglyphs"] for r in test}
    pool = [r for r in clean[N_TEST:] if r["hieroglyphs"] not in test_lines]
    shots = pool[:N_SHOTS]
    train = pool[N_SHOTS:N_SHOTS + MAX_TRAIN]

    write_jsonl(f"{DATA_DIR}/test.jsonl", test)
    write_jsonl(f"{DATA_DIR}/shots.jsonl", shots)
    write_jsonl(f"{DATA_DIR}/train.jsonl", train)

    # Chat-format training file for supervised fine-tuning (same prompt as evaluation).
    ft_rows = [{
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(r["hieroglyphs"])},
            {"role": "assistant", "content": r["transcription"]},
        ]
    } for r in train]
    write_jsonl(f"{DATA_DIR}/ft_train.jsonl", ft_rows)

    print(f"\nRows with hieroglyphs (deduplicated): {len(clean)}")
    print(f"Test: {len(test)} | Prompt examples: {len(shots)} | Train: {len(train)}")
    print(f"Fine-tuning file: {DATA_DIR}/ft_train.jsonl")
    print("\nSample test row:")
    print(json.dumps(test[0], ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- model calls

def build_messages(hieroglyphs, shots):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": user_message(s["hieroglyphs"])})
        msgs.append({"role": "assistant", "content": s["transcription"]})
    msgs.append({"role": "user", "content": user_message(hieroglyphs)})
    return msgs


def make_caller(args):
    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        def call(messages):
            system = messages[0]["content"]
            resp = client.messages.create(
                model=args.model,
                system=system,
                messages=messages[1:],
                max_tokens=args.max_tokens,
                temperature=0,
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text, resp.usage.input_tokens, resp.usage.output_tokens
        return call

    from openai import OpenAI
    if args.provider == "fireworks":
        client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                        api_key=os.environ["FIREWORKS_API_KEY"])
    elif args.provider == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:  # any other OpenAI-compatible endpoint
        client = OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env])

    def call(messages):
        kwargs = dict(model=args.model, messages=messages, temperature=0,
                      max_tokens=args.max_tokens)
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            # Some newer models reject max_tokens or temperature; retry without them.
            if "max_tokens" in msg:
                kwargs.pop("max_tokens")
                kwargs["max_completion_tokens"] = args.max_tokens
            if "temperature" in msg:
                kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)
    return call


def clean_output(text):
    for line in text.strip().splitlines():
        line = line.strip().strip("`").strip('"').strip()
        if line.lower().startswith("transliteration:"):
            line = line.split(":", 1)[1].strip()
        if line:
            return line
    return ""


# ---------------------------------------------------------------- scoring

def normalize(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------- run + report

def run(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test = read_jsonl(f"{DATA_DIR}/test.jsonl")
    if args.limit:
        test = test[:args.limit]
    shots = [] if args.zero_shot else read_jsonl(f"{DATA_DIR}/shots.jsonl")
    call = make_caller(args)

    def one(row):
        messages = build_messages(row["hieroglyphs"], shots)
        for attempt in range(4):
            try:
                t0 = time.time()
                text, tin, tout = call(messages)
                latency = time.time() - t0
                break
            except Exception as e:
                if attempt == 3:
                    text, tin, tout, latency = f"ERROR: {e}", 0, 0, 0.0
                else:
                    time.sleep(2 ** attempt)
        pred = normalize(clean_output(text)) if not text.startswith("ERROR") else ""
        ref = normalize(row["transcription"])
        edits = edit_distance(pred, ref)
        return {**row, "prediction": pred, "raw_output": text, "edits": edits,
                "ref_chars": len(ref), "exact": pred == ref, "latency_s": latency,
                "input_tokens": tin, "output_tokens": tout}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, test))

    write_jsonl(f"{RESULTS_DIR}/{args.name}.jsonl", results)

    n = len(results)
    errors = sum(r["raw_output"].startswith("ERROR") for r in results)
    cer = sum(r["edits"] for r in results) / max(1, sum(r["ref_chars"] for r in results))
    exact = sum(r["exact"] for r in results) / n
    lats = [r["latency_s"] for r in results if r["latency_s"] > 0]
    tin = sum(r["input_tokens"] for r in results)
    tout = sum(r["output_tokens"] for r in results)
    cost = (tin * args.in_price + tout * args.out_price) / 1e6
    summary = {
        "name": args.name, "provider": args.provider, "model": args.model,
        "prompting": "zero-shot" if args.zero_shot else f"{N_SHOTS}-shot",
        "n": n, "api_errors": errors,
        "char_error_rate": round(cer, 4),
        "char_accuracy": round(1 - cer, 4),
        "exact_match": round(exact, 4),
        "median_latency_s": round(statistics.median(lats), 3) if lats else None,
        "cost_usd_total": round(cost, 4),
        "cost_usd_per_1000": round(cost / n * 1000, 4) if n else None,
        "prices_per_1M_tokens": {"input": args.in_price, "output": args.out_price},
    }
    with open(f"{RESULTS_DIR}/{args.name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def report(_args):
    summaries = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith("_summary.json"):
            with open(f"{RESULTS_DIR}/{fn}") as f:
                summaries.append(json.load(f))
    if not summaries:
        print("No results yet. Run `python bench.py run ...` first.")
        return
    lines = [
        "| System | Model | Prompting | Char error rate ↓ | Exact match ↑ | Median latency (s) ↓ | Cost per 1,000 lines (USD) ↓ | API errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(summaries, key=lambda s: s["char_error_rate"]):
        lines.append(
            f"| {s['name']} | {s['model']} | {s['prompting']} | {s['char_error_rate']:.1%} | "
            f"{s['exact_match']:.1%} | {s['median_latency_s']} | {s['cost_usd_per_1000']} | {s['api_errors']} |"
        )
    table = "\n".join(lines)
    note = (f"\n\nEvaluated on {summaries[0]['n']} held-out lines from bbaw_egyptian "
            "(seed 42). Char error rate = character edit distance / reference length, "
            "summed over all lines.\n")
    with open(f"{RESULTS_DIR}/results.md", "w", encoding="utf-8") as f:
        f.write(table + note)
    print(table + note)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--provider", required=True,
                   choices=["openai", "anthropic", "fireworks", "openai-compatible"])
    r.add_argument("--model", required=True)
    r.add_argument("--zero-shot", action="store_true",
                   help="no prompt examples (use for the fine-tuned model)")
    r.add_argument("--in-price", type=float, default=0.0, help="USD per 1M input tokens")
    r.add_argument("--out-price", type=float, default=0.0, help="USD per 1M output tokens")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=0, help="run on first N test rows only")
    r.add_argument("--base-url")
    r.add_argument("--api-key-env")
    sub.add_parser("report")
    args = p.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
