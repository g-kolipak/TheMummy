"""Copy a training run's evidence into runs/<run-id>/ in the shape compare.py reads.

The training script records, per eval row, the sampled tokens but not the prompt length or
a per-row latency (rows are sampled concurrently in-session). We add:
  output_tokens = number of sampled tokens for the row (exact)
  input_tokens  = the run's total eval prompt tokens / number of eval rows (exact total,
                  evenly allocated; only used for the cost-per-1,000-cases figure)

    python collect_run.py <cookbook>/training/hiero_vision_logs/hiero-onemodel-20261102 \
        <cookbook>/training/hiero-onemodel-20261102.log
"""
import json
import re
import shutil
import sys
from pathlib import Path


def main(run_dir: str, log_path: str) -> None:
    src = Path(run_dir)
    dst = Path("runs") / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.jsonl", "summary.json", "plan.json", "eval_ids.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    log = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'"eval_prompt_tokens_expanded":\s*(\d+)', log)
    rows = [json.loads(l) for l in open(src / "predictions_tuned.jsonl", encoding="utf-8") if l.strip()]
    per_row_in = int(m.group(1)) / len(rows) if m else None
    with open(dst / "predictions_tuned.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            r["output_tokens"] = len(r.get("tokens") or [])
            if per_row_in is not None:
                r["input_tokens"] = per_row_in
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy2(log_path, dst / "train.log")
    print(f"copied {len(rows)} predictions + metrics to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
