#!/usr/bin/env python3
"""Muse Glimmer mixed SFT: hieroglyph photos -> sign lists, sign codes -> transliteration.

Serverless Training API variant of ``muse_glimmer_starter.py`` that trains ONE
LoRA on a mixed chat JSONL. Rows are either

* image rows: system prompt, a user turn with one ``image_url`` data URL plus a
  short instruction, and an assistant reply with one sign per line
  (``<Gardiner code> x1 y1 x2 y2``, coordinates scaled to 0-1000); or
* text rows: system prompt, a user turn of Gardiner codes, and an assistant
  reply with the transliteration.

Both kinds render through the same renderer and share batches; image rows keep
their ``ImageChunk`` in the Datum, text rows have none.

What differs from the text starter:

* the renderer is built with an image processor, and Datums come from
  ``render_messages_to_datums`` so image chunks survive (``to_ints()`` would
  turn every image position into a sentinel integer);
* the training file is indexed by byte offset and rendered lazily per batch in
  a small prefetch pool, so a 400 MB file is never held in memory;
* training steps are pipelined: up to ``--max-inflight`` (forward_backward,
  optim_step) pairs stay queued on the trainer, which applies them in
  submission order (request ids are reserved at call time and sent as
  ``seq_id``);
* several ``--eval-file``s are sampled greedily with raw per-token logprobs
  (and optional top-k) through the deployment sampler; one
  ``predictions_<evalname>_<label>.jsonl`` per eval set;
* the base-model eval is skipped by default (the base reasoning model burns
  the whole token budget on every row).

Commands (run from ``training/``)::

    python -m examples.sft.hiero_vision_starter dryrun --file A.jsonl --file B.jsonl --eval-file E1.jsonl --eval-file E2.jsonl
    python -m examples.sft.hiero_vision_starter plan   --file mix.jsonl --eval-file E1.jsonl,E2.jsonl
    python -m examples.sft.hiero_vision_starter launch --file mix.jsonl --eval-file E1.jsonl,E2.jsonl --run-id <id>
    python -m examples.sft.hiero_vision_starter cleanup --run-id <id>

``dryrun`` never contacts the API; ``plan`` only resolves the account.
``launch`` creates billable-surface resources and promotes the tuned checkpoint
to ``accounts/<account>/models/muse-starter-sl-<run-id>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import unicodedata
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
RENDERER_NAME = "muse_glimmer"
# End of Nov 30 Pacific; DST ends Nov 1, so Dec 1 00:00 is PST (UTC-8).
PROMOTION_DEADLINE = datetime(2026, 12, 1, tzinfo=timezone(timedelta(hours=-8)))
PROMOTION_SOURCE = "https://docs.fireworks.ai/fine-tuning/models"
LORA_ALPHA = 32
LORA_TARGETS = {"train_mlp": True, "train_attn": True, "train_unembed": True}
LOG_ROOT = "./hiero_vision_logs"

# "O4 168 3 697 64" -> code + four ints. Gardiner codes: letter(s), digits, optional suffix.
_SIGN_LINE_RE = re.compile(r"^\s*([A-Z][a-z]?\d+[A-Za-z]?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


# ---------------------------------------------------------------------------
# Small helpers shared with the text starter
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    try:
        import dotenv
    except ImportError:
        path = Path.cwd() / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True), override=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def promotion_active(now: datetime | None = None) -> bool:
    return (now or _now()) < PROMOTION_DEADLINE


def _connect():
    """Return a client bound to the key's account, and that account ID."""
    from fireworks import Fireworks

    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("FIREWORKS_API_KEY is not set. Export it or add it to a gitignored .env. Never paste it into chat.")
    account = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not account:
        first = next(iter(Fireworks().accounts.list(page_size=1)), None)
        if first is None:
            sys.exit("Could not resolve the account for this API key; set FIREWORKS_ACCOUNT_ID.")
        account = first.name.split("/")[-1]
    return Fireworks(account_id=account, timeout=120), account


def _names(account: str, run_id: str) -> dict:
    serverless_id = f"muse-starter-sl-{run_id}"[:63].rstrip("-")
    return {
        "serverless_model_id": serverless_id,
        "serverless_model": f"accounts/{account}/models/{serverless_id}",
    }


def _split_paths(values: list[str] | None) -> list[Path]:
    """``--file a.jsonl --file b.jsonl`` and ``--file a.jsonl,b.jsonl`` both work."""
    paths: list[Path] = []
    for value in values or []:
        for part in value.split(","):
            if part.strip():
                paths.append(Path(part.strip()))
    for path in paths:
        if not path.is_file():
            sys.exit(f"{path} does not exist.")
    return paths


# ---------------------------------------------------------------------------
# Data: streamed, indexed by byte offset, validated, never rewritten
# ---------------------------------------------------------------------------


def _iter_rows(path: Path, limit: int | None = None) -> Iterator[tuple[int, dict]]:
    """Yield (1-based line number, row) without reading the whole file into memory."""
    with path.open("r", encoding="utf-8") as f:
        n_yielded = 0
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            yield lineno, json.loads(line)
            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                return


def _row_kind(row: dict) -> str:
    """'image' when the last user turn carries an image part, else 'text'."""
    for m in reversed(row.get("messages", [])):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in {"image_url", "image"} for p in content
            ):
                return "image"
            return "text"
    return "text"


def _row_problem(row: Any) -> str | None:
    msgs = row.get("messages") if isinstance(row, dict) else None
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
        return "messages is not a non-empty list of objects"
    if msgs[-1].get("role") != "assistant" or not str(msgs[-1].get("content") or "").strip():
        return "last message is not a non-empty assistant reply"
    if not any(m.get("role") == "user" for m in msgs):
        return "no user message"
    return None


def _index_file(path: Path) -> dict:
    """One streaming pass: byte offset of every row, validation, and kind counts."""
    offsets: list[int] = []
    kinds = {"image": 0, "text": 0}
    problems: list[str] = []
    offset = 0
    with path.open("rb") as f:
        for lineno, raw in enumerate(f, 1):
            start, offset = offset, offset + len(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                problems.append(f"line {lineno}: not JSON")
                continue
            problem = _row_problem(row)
            if problem:
                problems.append(f"line {lineno}: {problem}")
                continue
            kinds[_row_kind(row)] += 1
            offsets.append(start)
    return {"path": path, "offsets": offsets, "kinds": kinds, "problems": problems}


def _read_row(fh, offset: int) -> dict:
    fh.seek(offset)
    return json.loads(fh.readline())


def _context(ex: dict) -> list[dict]:
    return ex["messages"][:-1]


def _gold(ex: dict) -> str:
    return str(ex["messages"][-1].get("content") or "")


# ---------------------------------------------------------------------------
# Rendering: renderer with image processor, Datums that keep image chunks
# ---------------------------------------------------------------------------


def _build_renderer():
    from training.renderer.tokenizer import get_tokenizer
    from training.utils.supervised import build_renderer

    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    renderer = build_renderer(tokenizer, TOKENIZER_MODEL, RENDERER_NAME, load_image_processor=True)
    if getattr(renderer, "image_processor", None) is None:
        sys.exit("The renderer has no image processor; image rows cannot be rendered.")
    return tokenizer, renderer


def _render_train(ex: dict, renderer):
    """One RenderedSupervisedDatum per row; loss only on the last assistant turn."""
    from training.renderer import TrainOnWhat
    from training.utils.supervised import render_messages_to_datums

    rendered = render_messages_to_datums(
        ex["messages"], renderer=renderer, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    if len(rendered) != 1:
        raise ValueError(f"expected one rendered example per row, got {len(rendered)}")
    return rendered[0]


def _image_stats(datum) -> tuple[int, int]:
    from fireworks.training import sdk

    chunks = [
        c for c in datum.model_input.chunks if isinstance(c, (sdk.types.ImageChunk, sdk.types.ImageAssetPointerChunk))
    ]
    return len(chunks), sum(int(c.length) for c in chunks)


def _inspect_rendered(rendered, tokenizer, max_seq_len: int) -> dict:
    """Token counts, image counts, and the loss-weighted span for one rendered row."""
    from fireworks.training import sdk

    token_ids = list(rendered.token_ids)
    weights = list(rendered.token_weights)
    datum = rendered.datum
    n_images, image_tokens = _image_stats(datum)
    weighted = [i for i, w in enumerate(weights) if w > 0]
    # Image positions are negative sentinels in token_ids; they must never carry loss.
    weighted_on_image = [i for i in weighted if token_ids[i] < 0]
    cursor, last_image_end = 0, 0
    for chunk in datum.model_input.chunks:
        length = len(chunk.tokens) if isinstance(chunk, sdk.types.EncodedTextChunk) else int(chunk.length)
        cursor += length
        if not isinstance(chunk, sdk.types.EncodedTextChunk):
            last_image_end = cursor
    return {
        "total_tokens": len(token_ids),
        "text_tokens": sum(1 for t in token_ids if t >= 0),
        "image_chunks": n_images,
        "image_tokens": image_tokens,
        "weighted_tokens": len(weighted),
        "weighted_span": [weighted[0], weighted[-1] + 1] if weighted else None,
        "weighted_after_last_image": bool(weighted) and weighted[0] >= last_image_end,
        "weighted_on_image_positions": len(weighted_on_image),
        "weighted_contiguous": bool(weighted) and weighted[-1] - weighted[0] + 1 == len(weighted),
        "datum_model_input_length": int(datum.model_input.length),
        "datum_target_len": len(datum.loss_fn_inputs["target_tokens"].data),
        "datum_weights_len": len(datum.loss_fn_inputs["weights"].data),
        "fits_max_seq_len": len(token_ids) <= max_seq_len,
        "decoded_weighted": tokenizer.decode([token_ids[i] for i in weighted]) if weighted else "",
    }


class BatchRenderer:
    """Render batches of rows by byte offset in a small thread pool, ahead of use."""

    def __init__(self, files: list[dict], renderer, max_seq_len: int, workers: int):
        self.files = files
        self.renderer = renderer
        self.max_seq_len = max_seq_len
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self.lock = threading.Lock()
        self.dropped = {"oversized": 0, "no_loss": 0}
        self.local = threading.local()

    def _handle(self, file_idx: int):
        handles = getattr(self.local, "handles", None)
        if handles is None:
            handles = self.local.handles = {}
        if file_idx not in handles:
            handles[file_idx] = self.files[file_idx]["path"].open("rb")
        return handles[file_idx]

    def _render(self, keys: list[tuple[int, int]]) -> tuple[list, dict]:
        datums, images, tokens, loss_tokens = [], 0, 0, 0
        for file_idx, offset in keys:
            ex = _read_row(self._handle(file_idx), offset)
            rendered = _render_train(ex, self.renderer)
            n = len(rendered.token_ids)
            weighted = sum(1 for w in rendered.token_weights if w > 0)
            if n > self.max_seq_len:
                with self.lock:
                    self.dropped["oversized"] += 1
                continue
            if n < 2 or weighted == 0:
                with self.lock:
                    self.dropped["no_loss"] += 1
                continue
            n_img, _ = _image_stats(rendered.datum)
            images += n_img > 0
            tokens += n
            loss_tokens += weighted
            datums.append(rendered.datum)
        return datums, {"images": images, "tokens": tokens, "loss_tokens": loss_tokens}

    def submit(self, keys: list[tuple[int, int]]):
        return self.pool.submit(self._render, keys)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Eval prompts: image chunks projected to placeholders + base64 payloads
# ---------------------------------------------------------------------------


def _eval_prompt(ex: dict, renderer, tokenizer) -> dict:
    """Generation prompt for one eval row, projected for the completions sampler."""
    from fireworks.training import sdk
    from training.utils.supervised import build_tool_prefixed_messages

    messages = build_tool_prefixed_messages(_context(ex), renderer=renderer, tools=ex.get("tools"))
    prompt = renderer.build_generation_prompt(messages)
    if all(isinstance(c, sdk.types.EncodedTextChunk) for c in prompt.chunks):
        token_ids, images = [int(t) for t in prompt.to_ints()], []
    else:
        from training.utils.rl.rollout.renderer import build_multimodal_completions_prompt_token_ids

        token_ids, images = build_multimodal_completions_prompt_token_ids(
            messages, prompt, tokenizer, renderer=renderer
        )
    return {
        "kind": "image" if images else "text",
        "prompt_length": int(prompt.length),  # expanded: text + visual tokens
        "token_ids": token_ids,  # unexpanded: one <|patch|> per image
        "images": images,
        "n_images": len(images),
    }


def _load_eval_sets(paths: list[Path], renderer, tokenizer) -> list[dict]:
    sets = []
    for path in paths:
        rows = []
        for lineno, row in _iter_rows(path):
            problem = _row_problem(row)
            if problem:
                sys.exit(f"{path} line {lineno}: {problem}")
            rows.append(row)
        if not rows:
            sys.exit(f"{path} has no rows.")
        prompts = [_eval_prompt(ex, renderer, tokenizer) for ex in rows]
        sets.append({"name": path.stem, "path": str(path), "rows": rows, "prompts": prompts})
    names = [s["name"] for s in sets]
    if len(set(names)) != len(names):
        sys.exit(f"eval files need distinct basenames: {names}")
    return sets


# ---------------------------------------------------------------------------
# Scoring and per-sign confidence
# ---------------------------------------------------------------------------


def _parse_signs(text: str) -> list[dict]:
    signs = []
    for line in text.splitlines():
        m = _SIGN_LINE_RE.match(line)
        if m:
            signs.append({"code": m.group(1), "box": [int(m.group(i)) for i in range(2, 6)]})
    return signs


def _iou(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    return "\n".join(" ".join(line.split()) for line in s.strip().splitlines())


# Text-row scoring copied from themummy/bench.py so numbers are comparable.
def _clean_output(text: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip().strip("`").strip('"').strip()
        if line.lower().startswith("transliteration:"):
            line = line.split(":", 1)[1].strip()
        if line:
            return line
    return ""


def _normalize(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def _score(pred_text: str, gold_text: str, kind: str) -> dict:
    """Image rows: multi-line exact match plus sign metrics. Text rows: bench.py exact + edits."""
    if kind == "text":
        pred, ref = _normalize(_clean_output(pred_text)), _normalize(gold_text)
        edits = _edit_distance(pred, ref)
        return {
            "exact": pred == ref,
            "edits": edits,
            "ref_chars": len(ref),
            "cer": edits / max(1, len(ref)),
            "codes_exact": False,
            "n_gold": 0,
            "n_pred": 0,
            "pos_match": 0,
            "mean_iou": 0.0,
        }
    gold_n, pred_n = _norm(gold_text), _norm(pred_text)
    gold, pred = _parse_signs(gold_text), _parse_signs(pred_text)
    pairs = list(zip(gold, pred))
    pos_match = sum(1 for g, p in pairs if g["code"] == p["code"])
    ious = [_iou(g["box"], p["box"]) for g, p in pairs if g["code"] == p["code"]]
    edits = _edit_distance(pred_n, gold_n)
    return {
        "exact": pred_n == gold_n,
        "edits": edits,
        "ref_chars": len(gold_n),
        "cer": edits / max(1, len(gold_n)),
        "codes_exact": [g["code"] for g in gold] == [p["code"] for p in pred],
        "n_gold": len(gold),
        "n_pred": len(pred),
        "pos_match": pos_match,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
    }


def _token_char_spans(tokens: list[int], tokenizer) -> tuple[str, list[tuple[int, int]]]:
    """Decoded text plus the [start, end) character span each token covers."""
    full = tokenizer.decode(tokens)
    pieces = [tokenizer.decode([t]) for t in tokens]
    if "".join(pieces) != full:  # multi-byte merges: fall back to prefix decoding
        pieces, prev = [], ""
        for i in range(len(tokens)):
            cur = tokenizer.decode(tokens[: i + 1])
            pieces.append(cur[len(prev) :])
            prev = cur
        full = prev
    spans, pos = [], 0
    for piece in pieces:
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    return full, spans


def _sign_confidences(tokens: list[int], logprobs: list[float] | None, visible: str, tokenizer) -> list[dict]:
    """Attach token spans and log-prob sums to every predicted sign line.

    Tokens are mapped to characters of the full decoded completion; the visible
    answer is located inside it (after the last ``<|message|>``), and each sign
    line's characters are intersected with the token spans. ``code_logprob`` sums
    the logprobs of tokens touching the Gardiner code, ``line_logprob`` those of
    the whole line including its newline. Confidence is ``exp`` of either sum.
    """
    if not tokens or not visible:
        return []
    full, spans = _token_char_spans(tokens, tokenizer)
    start = full.rfind(visible)
    if start < 0:
        return []
    out, line_start = [], 0
    for line in visible.split("\n"):
        line_end = line_start + len(line) + 1  # include the newline
        m = _SIGN_LINE_RE.match(line)
        if m:
            abs_line = (start + line_start, start + min(line_end, len(visible)))
            abs_code = (start + line_start + m.start(1), start + line_start + m.end(1))
            line_tok = [i for i, (a, b) in enumerate(spans) if a < abs_line[1] and b > abs_line[0]]
            code_tok = [i for i, (a, b) in enumerate(spans) if a < abs_code[1] and b > abs_code[0]]
            entry = {
                "code": m.group(1),
                "box": [int(m.group(i)) for i in range(2, 6)],
                "token_span": [line_tok[0], line_tok[-1] + 1] if line_tok else None,
                "code_tokens": code_tok,
            }
            if logprobs is not None:
                code_lp = sum(logprobs[i] for i in code_tok)
                line_lp = sum(logprobs[i] for i in line_tok)
                entry.update(
                    code_logprob=code_lp,
                    line_logprob=line_lp,
                    code_confidence=math.exp(code_lp),
                    line_confidence=math.exp(line_lp),
                    min_token_prob=math.exp(min(logprobs[i] for i in line_tok)) if line_tok else None,
                )
            out.append(entry)
        line_start = line_end
    return out


# ---------------------------------------------------------------------------
# Sampling with logprobs
# ---------------------------------------------------------------------------


def _sample_with_logprobs(sampler, prompt_ids: list[int], images: list[str], *, max_tokens: int, stop, top_logprobs: int):
    """Submit one greedy completion and return a future of a SampledCompletion list.

    ``FiretitanSamplingClient.sample()`` wraps exactly this call but keeps only
    ``sampling_logprob`` per token. The ``SampledCompletion`` returned by the
    deployment sampler also carries ``inference_logprobs`` (raw model logprob of
    the chosen token) and, when the server honours ``top_logprobs``,
    ``inference_topk_token_ids`` / ``inference_topk_logprobs``.
    """
    kwargs: dict[str, Any] = {"logprobs": True}
    if images:
        kwargs["images"] = list(images)
    if top_logprobs > 0:
        kwargs["top_logprobs"] = int(top_logprobs)
    coro = sampler.deployment_sampler.sample_with_prompt_tokens(
        list(prompt_ids), n=1, max_tokens=max_tokens, temperature=0.0, stop=stop, **kwargs
    )
    submit = getattr(sampler, "_submit", None)
    if submit is not None:  # same loop/thread sample() uses
        return submit(coro)
    import asyncio

    return ThreadPoolExecutor(max_workers=1).submit(asyncio.run, coro)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_dryrun(args: argparse.Namespace) -> int:
    """Render a few rows from every --file and --eval-file locally. No API calls."""
    train_paths = _split_paths(args.file)
    if not train_paths:
        sys.exit("Pass --file <train.jsonl> (repeatable).")
    tokenizer, renderer = _build_renderer()
    report: dict = {
        "tokenizer": TOKENIZER_MODEL,
        "renderer": type(renderer).__name__,
        "image_processor": type(renderer.image_processor).__name__,
        "max_seq_len": args.max_seq_len,
        "train_rows": [],
        "eval_rows": [],
    }
    ok = True
    for path in train_paths:
        for lineno, ex in _iter_rows(path, limit=args.dryrun_rows_per_file):
            problem = _row_problem(ex)
            if problem:
                report["train_rows"].append({"file": str(path), "line": lineno, "error": problem})
                ok = False
                continue
            kind = _row_kind(ex)
            info = {"file": path.name, "line": lineno, "id": ex.get("id"), "kind": kind}
            info.update(_inspect_rendered(_render_train(ex, renderer), tokenizer, args.max_seq_len))
            gold = _gold(ex)
            # LAST_ASSISTANT_TURN covers the ATEM header the model must emit after
            # "<|start|>assistant" (" to=user<|message|>"), the reply, and "<|eot|>".
            decoded = info["decoded_weighted"]
            at = decoded.find(gold)
            info["decoded_weighted_contains_gold"] = at >= 0
            info["weighted_prefix"] = decoded[:at] if at >= 0 else None
            info["weighted_suffix"] = decoded[at + len(gold) :] if at >= 0 else None
            info["image_chunks_as_expected"] = (info["image_chunks"] > 0) == (kind == "image")
            info["ok"] = (
                info["weighted_tokens"] > 0
                and info["weighted_after_last_image"]
                and info["weighted_on_image_positions"] == 0
                and info["weighted_contiguous"]
                and info["image_chunks_as_expected"]
                and info["datum_target_len"] == info["datum_weights_len"] == info["datum_model_input_length"]
                and info["decoded_weighted_contains_gold"]
                and info["fits_max_seq_len"]
            )
            ok = ok and info["ok"]
            report["train_rows"].append(info)
    for path in _split_paths(args.eval_file):
        for lineno, ex in _iter_rows(path, limit=args.dryrun_eval_rows):
            problem = _row_problem(ex)
            if problem:
                report["eval_rows"].append({"file": str(path), "line": lineno, "error": problem})
                ok = False
                continue
            prompt = _eval_prompt(ex, renderer, tokenizer)
            n_placeholders = sum(1 for t in prompt["token_ids"] if t == renderer.image_placeholder_token_id)
            info = {
                "file": path.name,
                "eval_name": path.stem,
                "line": lineno,
                "id": ex.get("id"),
                "kind": prompt["kind"],
                "prompt_length_expanded": prompt["prompt_length"],
                "prompt_tokens_unexpanded": len(prompt["token_ids"]),
                "images": prompt["n_images"],
                "image_placeholders_in_prompt": n_placeholders,
                "image_payload_chars": sum(len(i) for i in prompt["images"]),
                "max_tokens": min(args.max_eval_tokens, args.max_seq_len - prompt["prompt_length"]),
                "prompt_tail": tokenizer.decode(prompt["token_ids"][-12:]),
                "gold_signs": len(_parse_signs(_gold(ex))),
                "gold_head": _gold(ex)[:80],
            }
            expected_images = 1 if _row_kind(ex) == "image" else 0
            info["ok"] = n_placeholders == prompt["n_images"] == expected_images and info["max_tokens"] > 0
            ok = ok and info["ok"]
            report["eval_rows"].append(info)
    report["ok"] = ok
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 1


def _plan_dict(args: argparse.Namespace, account: str | None) -> dict:
    return {
        "account": account,
        "utc_now": _now().isoformat(timespec="seconds"),
        "promotion": {
            "active": promotion_active(),
            "deadline": PROMOTION_DEADLINE.isoformat(),
            "source": PROMOTION_SOURCE,
        },
        "base_model": BASE_MODEL,
        "method": "LoRA SFT (mixed image-to-text and text-to-text)",
        "run_id": args.run_id,
        "dataset": {
            "train_files": args.file,
            "eval_files": args.eval_file or f"random holdout of {args.n_eval} rows",
            "train_rows": args.n_train or "all",
        },
        "evaluation": {
            "base": "skipped" if args.skip_base_eval else "sampled",
            "tuned": "greedy, per-token raw logprobs"
            + (f" + top-{args.top_logprobs} alternatives" if args.top_logprobs else ""),
            "metrics": "text rows: bench.py exact match + CER; image rows: exact, code-sequence match, "
            "positional sign accuracy, mean IoU",
            "initial_concurrency": args.eval_concurrency,
        },
        "parameters": {
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_inflight_steps": args.max_inflight,
            "render_workers": args.render_workers,
            "learning_rate": args.learning_rate,
            "max_seq_len": args.max_seq_len,
            "max_eval_tokens": args.max_eval_tokens,
            "seed": args.seed,
            "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS,
            "train_on": "last_assistant_turn",
            "loss": "cross_entropy",
            "optimizer": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 1.0},
        },
        "resources": {
            "promoted_model": None if args.no_promote or not account else _names(account, args.run_id)["serverless_model"],
        },
        "cost": {
            "training": "$0 (promotional)" if promotion_active() else "paid: resolve the live rate",
            "evaluation_sampling": "billed separately per token",
        },
    }


def cmd_plan(args: argparse.Namespace) -> int:
    _, account = _connect()
    print(json.dumps(_plan_dict(args, account), indent=2))
    return 0


def _prepare(args: argparse.Namespace):
    """Index and validate the train files, load eval sets, build the renderer."""
    train_paths = _split_paths(args.file)
    if not train_paths:
        sys.exit("Pass --file <train.jsonl> (repeatable).")
    files = [_index_file(p) for p in train_paths]
    problems = [f"{f['path']}: {p}" for f in files for p in f["problems"]]
    if problems:
        print(json.dumps({"problems": problems[:50]}, indent=2))
        sys.exit("Fix or remove these rows in a copy of the file, then rerun.")
    keys = [(i, off) for i, f in enumerate(files) for off in f["offsets"]]
    run_dir = Path(args.log_path or f"{LOG_ROOT}/{args.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, renderer = _build_renderer()
    eval_paths = _split_paths(args.eval_file)
    if eval_paths:
        eval_sets = _load_eval_sets(eval_paths, renderer, tokenizer)
        if args.n_train:
            keys = keys[: args.n_train]
    else:
        if len(keys) <= args.n_eval:
            sys.exit(f"{len(keys)} rows is too few to hold out {args.n_eval}; pass --eval-file or lower --n-eval.")
        random.Random(args.seed).shuffle(keys)
        held = keys[: args.n_eval]
        keys = keys[args.n_eval :][: args.n_train] if args.n_train else keys[args.n_eval :]
        rows = []
        for file_idx, offset in held:
            with files[file_idx]["path"].open("rb") as fh:
                rows.append(_read_row(fh, offset))
        eval_sets = [
            {
                "name": "holdout",
                "path": "holdout",
                "rows": rows,
                "prompts": [_eval_prompt(ex, renderer, tokenizer) for ex in rows],
            }
        ]
    (run_dir / "eval_sets.json").write_text(
        json.dumps(
            [{"name": s["name"], "path": s["path"], "ids": [ex.get("id") for ex in s["rows"]]} for s in eval_sets],
            indent=2,
        )
    )
    stats = {
        "files": [{"path": str(f["path"]), "rows": len(f["offsets"]), **f["kinds"]} for f in files],
        "train_rows": len(keys),
        "image_rows": sum(f["kinds"]["image"] for f in files),
        "text_rows": sum(f["kinds"]["text"] for f in files),
    }
    return run_dir, files, keys, eval_sets, tokenizer, renderer, stats


def cmd_launch(args: argparse.Namespace) -> int:
    if not promotion_active() and not args.allow_paid:
        sys.exit("The Muse Glimmer promotion has ended. Re-run with --allow-paid after approving the live rate.")
    _, account = _connect()
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from fireworks.training.sdk.concurrency import AdaptiveConcurrencyController
    from training.renderer import get_text_content
    from training.utils.serverless import enable_serverless_supervised_409_retry, serverless_base_url

    run_dir, files, keys, eval_sets, tokenizer, renderer, stats = _prepare(args)
    steps_per_epoch = math.ceil(len(keys) / args.batch_size)
    print(
        json.dumps(
            {
                "train": stats,
                "eval_sets": {s["name"]: len(s["rows"]) for s in eval_sets},
                "eval_prompt_tokens_expanded": sum(p["prompt_length"] for s in eval_sets for p in s["prompts"]),
                "steps_per_epoch": steps_per_epoch,
                "total_steps": steps_per_epoch * args.epochs,
                "max_inflight": args.max_inflight,
                "run_dir": str(run_dir),
            }
        ),
        flush=True,
    )
    (run_dir / "plan.json").write_text(json.dumps(_plan_dict(args, account), indent=2))

    service = FiretitanServiceClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=serverless_base_url(os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")),
    )
    enable_serverless_supervised_409_retry(service)
    log = (run_dir / "metrics.jsonl").open("a")
    stop = renderer.get_stop_sequences()
    results: dict[str, dict] = {}
    batch_renderer = BatchRenderer(files, renderer, args.max_seq_len, args.render_workers)
    try:
        training_client = service.create_lora_training_client(
            base_model=BASE_MODEL, rank=args.lora_rank, alpha=LORA_ALPHA, **LORA_TARGETS
        )
        print(
            f"session={getattr(service, 'training_session_id', None)} run={getattr(training_client, 'run_id', None)}",
            flush=True,
        )

        def evaluate(label: str) -> dict:
            snapshot = training_client.save_weights_for_sampler(f"starter-{label}").result().path
            sampler = service.create_sampling_client(
                model_path=snapshot,
                tokenizer=tokenizer,
                concurrency_controller=AdaptiveConcurrencyController(initial_window=args.eval_concurrency),
            )
            out: dict[str, dict] = {}
            try:
                for eval_set in eval_sets:
                    out[eval_set["name"]] = _evaluate_set(
                        eval_set, label, sampler, run_dir, args, stop, renderer, tokenizer, get_text_content
                    )
            finally:
                close = getattr(sampler, "close", None)
                if close:
                    close()
            log.write(json.dumps({"eval": {"label": label, "sets": out}}) + "\n")
            log.flush()
            results[label] = out
            return out

        if not args.skip_base_eval:
            evaluate("base")

        adam = sdk.AdamParams(
            learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0
        )
        rng = random.Random(args.seed)
        step = 0
        in_flight: deque = deque()
        started = time.monotonic()

        def collect() -> None:
            s, epoch, info, t_submit, fb_fut, opt_fut = in_flight.popleft()
            fb = fb_fut.result(timeout=args.step_timeout)
            opt = opt_fut.result(timeout=args.step_timeout)
            metrics = dict(getattr(fb, "metrics", None) or {})
            tokens = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or info["loss_tokens"] or 1.0
            loss = metrics.get("loss:sum", 0.0) / max(float(tokens), 1.0)
            grad = {k: v for k, v in (getattr(opt, "metrics", None) or {}).items() if "grad" in k}
            row = {
                "step": s,
                "epoch": epoch,
                "loss": round(loss, 4),
                "batch": info["batch"],
                "image_rows": info["images"],
                "tokens": info["tokens"],
                "loss_tokens": info["loss_tokens"],
                "latency_s": round(time.monotonic() - t_submit, 1),
                "elapsed_s": round(time.monotonic() - started, 1),
                **grad,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)

        try:
            for epoch in range(args.epochs):
                order = list(keys)
                rng.shuffle(order)
                batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
                prefetch: deque = deque()
                next_batch = 0
                depth = args.max_inflight + 2
                while next_batch < len(batches) and len(prefetch) < depth:
                    prefetch.append(batch_renderer.submit(batches[next_batch]))
                    next_batch += 1
                while prefetch:
                    datums, info = prefetch.popleft().result()
                    if next_batch < len(batches):
                        prefetch.append(batch_renderer.submit(batches[next_batch]))
                        next_batch += 1
                    if not datums:
                        continue
                    info["batch"] = len(datums)
                    step += 1
                    # Request ids are reserved here, in call order, and sent as seq_id;
                    # the trainer applies them in that order, so queue without waiting.
                    fb_fut = training_client.forward_backward(datums, "cross_entropy")
                    opt_fut = training_client.optim_step(adam, emit_grad_norm_metrics=True)
                    in_flight.append((step, epoch + 1, info, time.monotonic(), fb_fut, opt_fut))
                    while len(in_flight) >= args.max_inflight:
                        collect()
                while in_flight:  # drain before the next epoch / eval
                    collect()
        finally:
            while in_flight:
                try:
                    collect()
                except Exception as exc:  # noqa: BLE001 - drain on the way out
                    print(f"drain: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr, flush=True)
        print(json.dumps({"dropped_rows": batch_renderer.dropped, "steps": step}), flush=True)

        tuned = evaluate("tuned")
        promoted = None if args.no_promote else _promote_tuned(service, account, args.run_id)
        summary = {
            "steps": step,
            "train": stats,
            "dropped_rows": batch_renderer.dropped,
            "base": results.get("base"),
            "tuned": tuned,
            "promoted_model": promoted,
            "session": getattr(service, "training_session_id", None),
            "log": str(run_dir),
            "train_seconds": round(time.monotonic() - started, 1),
            "cost": {
                "training": "$0 (promotional)" if promotion_active() else "paid at the live rate",
                "evaluation_sampling": {
                    "prompt_tokens": sum(r["prompt_tokens"] for sets in results.values() for r in sets.values()),
                    "sampled_tokens": sum(r["sampled_tokens"] for sets in results.values() for r in sets.values()),
                    "note": "billed separately from training at published per-token rates",
                },
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        batch_renderer.close()
        log.close()
        service.close()
        print(
            "Serverless session closed; no deployment was created. Its training slot frees when the "
            "session expires, so a new run started right away may wait or get a 429.",
            flush=True,
        )
    return 0


def _evaluate_set(eval_set: dict, label: str, sampler, run_dir: Path, args, stop, renderer, tokenizer, get_text_content):
    """Sample every row of one eval set at once and write predictions_<name>_<label>.jsonl."""
    rows, prompts, name = eval_set["rows"], eval_set["prompts"], eval_set["name"]
    preds = (run_dir / f"predictions_{name}_{label}.jsonl").open("w")
    totals = {
        "exact": 0,
        "edits": 0,
        "ref_chars": 0,
        "codes_exact": 0,
        "pos_match": 0,
        "n_gold": 0,
        "n_pred": 0,
        "iou": 0.0,
    }
    truncated = failed = 0
    prompt_tokens = sampled_tokens = 0
    saw_raw_logprobs = saw_topk = False
    t0 = time.monotonic()
    try:
        futures = [
            _sample_with_logprobs(
                sampler,
                p["token_ids"],
                p["images"],
                max_tokens=max(1, min(args.max_eval_tokens, args.max_seq_len - p["prompt_length"])),
                stop=stop,
                top_logprobs=args.top_logprobs,
            )
            for p in prompts
        ]
        for i, (ex, p, fut) in enumerate(zip(rows, prompts, futures)):
            gold = _gold(ex)
            record: dict[str, Any] = {"index": i, "id": ex.get("id"), "kind": p["kind"], "gold": gold}
            prompt_tokens += p["prompt_length"]
            try:
                completions = fut.result(timeout=args.sample_timeout)
            except Exception as exc:  # noqa: BLE001 - keep scoring the other rows
                failed += 1
                record.update(error=f"{type(exc).__name__}: {str(exc)[:300]}", answer_text="")
                preds.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            if not completions:
                failed += 1
                record.update(error="empty completion list", answer_text="")
                preds.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            comp = completions[0]
            tokens = [int(t) for t in comp.full_tokens[comp.prompt_len :]]
            sampled_tokens += len(tokens)
            msg, termination = renderer.parse_response(tokens)
            text = get_text_content(msg)
            reasoning = "".join(
                str(part.get("thinking", ""))
                for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                if isinstance(part, dict) and part.get("type") == "thinking"
            )
            is_truncated = not termination.is_clean or comp.finish_reason == "length"
            truncated += is_truncated
            raw_lp = list(comp.inference_logprobs) if comp.inference_logprobs is not None else None
            sampling_lp = list(comp.sampling_logprobs) if comp.sampling_logprobs is not None else None
            saw_raw_logprobs = saw_raw_logprobs or raw_lp is not None
            topk = None
            if comp.inference_topk_token_ids is not None and comp.inference_topk_logprobs is not None:
                topk = [
                    [[int(t), float(lp)] for t, lp in zip(ids, lps)]
                    for ids, lps in zip(comp.inference_topk_token_ids, comp.inference_topk_logprobs)
                ]
                saw_topk = True
            score = _score(text, gold, p["kind"])
            for key in ("exact", "edits", "ref_chars", "codes_exact", "pos_match", "n_gold", "n_pred"):
                totals[key] += int(score[key])
            totals["iou"] += score["mean_iou"] * score["pos_match"]
            record.update(
                answer_text=text,
                reasoning_chars=len(reasoning),
                truncated=is_truncated,
                stop_reason=comp.finish_reason,
                **score,
                tokens=tokens,
                logprobs=raw_lp,
                sampling_logprobs=sampling_lp,
                topk=topk,
                signs=_sign_confidences(tokens, raw_lp, text, tokenizer) if p["kind"] == "image" else [],
            )
            preds.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"{label}/{name}: scored {i + 1}/{len(rows)}", flush=True)
    finally:
        preds.close()
    n = len(rows)
    result = {
        "label": label,
        "eval_set": name,
        "rows": n,
        "exact_match": totals["exact"] / n,
        "char_error_rate": totals["edits"] / max(1, totals["ref_chars"]),  # bench.py: total edits / total ref chars
        "char_accuracy": 1 - totals["edits"] / max(1, totals["ref_chars"]),
        "code_sequence_match": totals["codes_exact"] / n,
        "sign_accuracy": totals["pos_match"] / totals["n_gold"] if totals["n_gold"] else None,
        "mean_iou_on_matched": totals["iou"] / totals["pos_match"] if totals["pos_match"] else None,
        "pred_signs_per_gold_sign": totals["n_pred"] / totals["n_gold"] if totals["n_gold"] else None,
        "unterminated": truncated,
        "failed_rows": failed,
        "prompt_tokens": prompt_tokens,
        "sampled_tokens": sampled_tokens,
        "seconds": round(time.monotonic() - t0, 1),
        "raw_logprobs_present": saw_raw_logprobs,
        "topk_present": saw_topk,
    }
    print(json.dumps(result), flush=True)
    return result


def _promote_tuned(service, account: str, run_id: str) -> str | None:
    """Promote the tuned sampler checkpoint to a LoRA model before the session (and its checkpoints) go away."""
    from fireworks.training.sdk import FireworksClient

    fw = FireworksClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"),
    )
    rows = [
        r
        for r in fw.list_training_session_checkpoints(service.training_session_name)
        if r.get("promotable") and "starter-tuned" in (r.get("checkpointName") or r.get("name") or "")
    ]
    if not rows:
        print("no promotable tuned checkpoint found; the adapter stays session-only", flush=True)
        return None
    target = max(rows, key=lambda r: r.get("createTime", ""))
    model_id = _names(account, run_id)["serverless_model_id"]
    fw.promote_session_checkpoint(name=target["name"], output_model_id=model_id, base_model=BASE_MODEL)
    name = f"accounts/{account}/models/{model_id}"
    print(json.dumps({"promoted_model": name, "from_checkpoint": target["name"]}), flush=True)
    return name


def cmd_cleanup(args: argparse.Namespace) -> int:
    if not args.run_id:
        sys.exit("Pass the --run-id printed by launch.")
    import fireworks

    client, account = _connect()
    names = _names(account, args.run_id)
    try:
        client.models.delete(names["serverless_model_id"])
        print(f"deleted {names['serverless_model']}")
    except fireworks.NotFoundError:
        print(f"skip: {names['serverless_model']} does not exist")
    except fireworks.APIStatusError as exc:
        print(f"FAILED to delete {names['serverless_model']}: HTTP {exc.status_code} {exc.message}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("plan", "dryrun", "launch", "cleanup"):
        s = sub.add_parser(name)
        s.add_argument("--run-id", default=None, help="Names the promoted model and log dir.")
        s.add_argument(
            "--file",
            action="append",
            default=None,
            help="Train chat JSONL (image and/or text rows). Repeatable or comma-separated.",
        )
        s.add_argument(
            "--eval-file",
            action="append",
            default=None,
            help="Eval chat JSONL; repeatable or comma-separated. Replaces the holdout.",
        )
        s.add_argument("--lora-rank", type=int, default=16)
        s.add_argument("--epochs", type=int, default=2)
        s.add_argument("--learning-rate", type=float, default=2e-4)
        s.add_argument("--batch-size", type=int, default=64)
        s.add_argument("--max-inflight", type=int, default=4, help="Pipelined (forward_backward, optim_step) pairs.")
        s.add_argument("--render-workers", type=int, default=2, help="Threads rendering upcoming batches.")
        s.add_argument("--max-seq-len", type=int, default=4096)
        s.add_argument("--max-eval-tokens", type=int, default=1024)
        s.add_argument("--eval-concurrency", type=int, default=32, help="Initial adaptive sampling window.")
        s.add_argument(
            "--skip-base-eval",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Skip sampling the base model (it reasons until max tokens on every row). Default: skip.",
        )
        s.add_argument(
            "--top-logprobs",
            type=int,
            default=0,
            help="Ask the sampler for this many alternatives per generated token (0 = off). "
            "Saved as `topk` when the server returns them.",
        )
        s.add_argument("--sample-timeout", type=int, default=900, help="Seconds to wait for one eval completion.")
        s.add_argument("--step-timeout", type=int, default=1800, help="Seconds to wait for one training step.")
        s.add_argument("--n-train", type=int, default=None, help="Cap on train rows (default: all).")
        s.add_argument("--n-eval", type=int, default=30, help="Holdout size when --eval-file is not given.")
        s.add_argument("--dryrun-rows-per-file", type=int, default=2)
        s.add_argument("--dryrun-eval-rows", type=int, default=1)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--log-path", default=None)
        s.add_argument("--allow-paid", action="store_true", help="Launch after the promotion ends.")
        s.add_argument("--no-promote", action="store_true", help="Keep the tuned adapter session-only.")
    args = p.parse_args(argv)
    if args.run_id is None and args.command in ("plan", "launch"):
        args.run_id = f"r{_now():%Y%m%d-%H%M%S}"
        if args.command == "launch":
            print(f"run id: {args.run_id} (pass --run-id {args.run_id} to cleanup)", file=sys.stderr)
    return args


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    # Match run_serverless.sh: image processors behind remote code load without a prompt.
    os.environ.setdefault("HF_TRUST_REMOTE_CODE", "1")
    args = parse_args(argv)
    if args.command == "dryrun":
        return cmd_dryrun(args)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "cleanup":
        return cmd_cleanup(args)
    return cmd_launch(args)


if __name__ == "__main__":
    sys.exit(main())
