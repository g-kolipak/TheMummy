"""Build one mixed training file for a single model that does both jobs.

  image rows : vision/data/vlm_train.jsonl  (photo -> Gardiner codes + boxes), repeated IMG_REPEAT times
  text rows  : data/ft_train_20k_dedup.jsonl (codes -> transliteration), N_TEXT rows sampled;
               half keep the MdC layout marks, half have them stripped to a plain code list,
               which is what the image task outputs, so the two steps chain.
  text dev   : N_DEV other rows (same 50/50 split), never trained on.

    python3 make_mix.py   # -> data/mix_train.jsonl, data/mix_text_dev.jsonl
"""
import json
import random
import re

IMG_FILE = "vision/data/vlm_train.jsonl"
TEXT_FILE = "data/ft_train_20k_dedup.jsonl"
N_TEXT, N_DEV, IMG_REPEAT, SEED = 8000, 300, 2, 7
CODE = re.compile(r"\b(?:[A-Z]|Aa|Ff|NL|NU)\d{1,3}[A-Za-z]?\b")


def strip_layout(row: dict) -> dict:
    row = json.loads(json.dumps(row))
    user = next(m for m in row["messages"] if m["role"] == "user")
    head, rest = user["content"].split("Hieroglyphs:", 1)
    hiero, tail = rest.split("\nTransliteration:", 1)
    user["content"] = f"{head}Hieroglyphs: {' '.join(CODE.findall(hiero))}\nTransliteration:{tail}"
    return row


def main() -> None:
    rng = random.Random(SEED)
    text = [json.loads(l) for l in open(TEXT_FILE, encoding="utf-8")]
    rng.shuffle(text)
    train_text, dev_text = text[:N_TEXT], text[N_TEXT:N_TEXT + N_DEV]
    train_text = [r if i % 2 == 0 else strip_layout(r) for i, r in enumerate(train_text)]
    dev_text = [r if i % 2 == 0 else strip_layout(r) for i, r in enumerate(dev_text)]

    lines = [l for l in open(IMG_FILE, encoding="utf-8") if l.strip()] * IMG_REPEAT
    lines += [json.dumps(r, ensure_ascii=False) + "\n" for r in train_text]
    rng.shuffle(lines)
    with open("data/mix_train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(lines)
    with open("data/mix_text_dev.jsonl", "w", encoding="utf-8") as f:
        for r in dev_text:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_img = len(lines) - len(train_text)
    print(f"mix_train: {len(lines)} rows ({n_img} image = {n_img // IMG_REPEAT} x{IMG_REPEAT}, "
          f"{len(train_text)} text: {N_TEXT // 2} with layout marks, {N_TEXT - N_TEXT // 2} plain codes); "
          f"mix_text_dev: {len(dev_text)} rows")
    print("plain-codes example:", strip_layout(text[1])["messages"][1]["content"][:160])


if __name__ == "__main__":
    main()
