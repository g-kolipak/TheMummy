"""Stricter train/test dedupe for fine-tuning files.

Drops a training row if EITHER
  - its Gardiner code sequence (layout marks like : * - [ ] ? & stripped) equals a test line's, or
  - its transliteration is identical to a test line's transliteration.

    python dedupe_ft.py data/ft_train.jsonl data/ft_train_20k_dedup.jsonl
"""
import json
import re
import sys
import unicodedata

CODE = re.compile(r"\b(?:[A-Z]|Aa|Ff|NL|NU)\d{1,3}[A-Za-z]?\b")


def codes(hiero: str) -> tuple:
    return tuple(CODE.findall(hiero))


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def hiero_of(row: dict) -> str:
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    return user.split("Hieroglyphs:", 1)[1].split("\nTransliteration:")[0]


def main(src: str, dst: str, test_path: str = "data/test.jsonl") -> None:
    test = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    test_codes = {codes(r["hieroglyphs"]) for r in test}
    test_answers = {norm(r["transcription"]) for r in test}
    kept = dropped_codes = dropped_answer = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            row = json.loads(line)
            if codes(hiero_of(row)) in test_codes:
                dropped_codes += 1
                continue
            if norm(row["messages"][-1]["content"]) in test_answers:
                dropped_answer += 1
                continue
            out.write(line)
            kept += 1
    print(f"{src}: kept {kept}, dropped {dropped_codes} (same signs ignoring layout) "
          f"+ {dropped_answer} (identical transliteration) -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
