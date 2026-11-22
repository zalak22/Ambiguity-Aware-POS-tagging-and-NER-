"""Phase 3 - Gujarati text normalisation + NER data preprocessing (Naamapadam, Gujarati).

Run from project root:   python src/preprocessing.py

The SAME normalize_token() / tokenize() functions are imported later by training and by
predict.py, so preprocessing is identical at training and prediction time.

Inputs  (read-only):  datasets/raw/naamapadam/gu_extracted/gu_{train,val,test}.json
Outputs:
  datasets/processed/ner/{train,train_150k,val,test,test_dedup}.jsonl
  models/vocabulary/ner_word_vocab.json, ner_char_vocab.json
  models/label_mappings/ner_labels.json
  results/reports/ner_preprocessing_report.json
"""
import json, random, re, unicodedata, collections, sys
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets/raw/naamapadam/gu_extracted"
OUT = ROOT / "datasets/processed/ner"
VOCAB = ROOT / "models/vocabulary"
LABELS = ROOT / "models/label_mappings"
REPORT = ROOT / "results/reports"

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
PAD, UNK = "<PAD>", "<UNK>"
WORD_MIN_FREQ = 2          # words seen once in train -> <UNK> (teaches the model to handle unknowns)
SUBSET_SIZE = 150_000      # fixed, seeded subset for fast ablation runs

# ---------------------------------------------------------------- normalisation
# Invisible characters that create different spellings of the same word.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
# Gujarati digits -> ASCII (the NER data mostly uses ASCII digits).
_DIGITS = {0x0AE6 + i: str(i) for i in range(10)}
# Devanagari danda sometimes appears in Gujarati text; Naamapadam uses '.'
_PUNCT = {0x0964: ".", 0x0965: "."}
_TRANS = {**_INVISIBLE, **_DIGITS, **_PUNCT}

def normalize_token(tok: str) -> str:
    """Unicode NFC + remove invisible chars + unify digits/danda + lowercase Latin."""
    tok = unicodedata.normalize("NFC", tok).translate(_TRANS).strip()
    return tok.lower() if tok.isascii() else tok

# Tokeniser for raw input sentences (prediction time). Matches the dataset convention:
# words split on whitespace, punctuation as separate tokens.
_PUNCT_RE = re.compile(r"([.,!?;:\"'()\[\]{}\-–—/|।॥“”‘’…])")

def tokenize(sentence: str) -> list[str]:
    sentence = unicodedata.normalize("NFC", sentence)
    toks = _PUNCT_RE.sub(r" \1 ", sentence).split()
    return [t for t in (normalize_token(x) for x in toks) if t]

def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences on . ? ! । (keeps the delimiter)."""
    parts = re.split(r"(?<=[.?!।])\s+", text.strip())
    return [p for p in parts if p.strip()]

# ---------------------------------------------------------------- data handling
def load_json_records(path: Path):
    txt = path.read_text(encoding="utf-8")
    if txt.lstrip().startswith("["):
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]

def clean_record(words, tags):
    toks, labs = [], []
    for w, t in zip(words, tags):
        w = normalize_token(w)
        if w:                       # drop tokens that become empty after normalisation
            toks.append(w); labs.append(t)
    return toks, labs

def repair_bio(tags):
    """I-X not preceded by B-X/I-X becomes B-X. Returns (fixed_tags, n_fixed)."""
    out, fixed, prev = [], 0, "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]; fixed += 1
        out.append(t); prev = t
    return out, fixed

def key(toks): return " ".join(toks)

def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    for d in (OUT, VOCAB, LABELS, REPORT):
        d.mkdir(parents=True, exist_ok=True)
    rep = {"seed": SEED, "word_min_freq": WORD_MIN_FREQ}
    data = {}
    for split in ("train", "val", "test"):
        p = RAW / f"gu_{split}.json"
        if not p.exists():
            sys.exit(f"Missing {p} - run inspect_naamapadam.py first.")
        recs = load_json_records(p)
        rows, bad_labels, empty, mismatch, bio_fixed, dropped_tokens = [], 0, 0, 0, 0, 0
        for i, r in enumerate(recs):
            words, tags = r["words"], r["ner"]
            if len(words) != len(tags): mismatch += 1; continue
            if any(t not in NER_LABELS for t in tags): bad_labels += 1; continue
            toks, labs = clean_record(words, tags)
            dropped_tokens += len(words) - len(toks)
            if not toks: empty += 1; continue
            if split != "test":                       # gold test is never altered
                labs, n = repair_bio(labs); bio_fixed += n
            rows.append({"id": f"{split}-{i}", "tokens": toks, "ner": labs})
        data[split] = rows
        rep[f"{split}_raw"] = len(recs)
        rep[f"{split}_dropped"] = {"length_mismatch": mismatch, "unknown_label": bad_labels,
                                   "empty_after_cleaning": empty, "empty_tokens_removed": dropped_tokens,
                                   "bio_repaired": bio_fixed}

    # ---- leakage + duplicates: train must not contain any val/test sentence
    held_out = {key(r["tokens"]) for r in data["val"] + data["test"]}
    seen, train, dup, leak = set(), [], 0, 0
    for r in data["train"]:
        k = key(r["tokens"])
        if k in held_out: leak += 1; continue
        if k in seen: dup += 1; continue
        seen.add(k); train.append(r)
    data["train"] = train
    rep["train_removed"] = {"overlap_with_val_or_test": leak, "duplicates": dup}
    # val: remove sentences that are also in test
    test_keys = {key(r["tokens"]) for r in data["test"]}
    before = len(data["val"])
    data["val"] = [r for r in data["val"] if key(r["tokens"]) not in test_keys]
    rep["val_removed_overlap_with_test"] = before - len(data["val"])
    # test_dedup: gold test with internal duplicates removed (report both)
    s, test_dedup = set(), []
    for r in data["test"]:
        k = key(r["tokens"])
        if k not in s: s.add(k); test_dedup.append(r)
    rep["test_internal_duplicates"] = len(data["test"]) - len(test_dedup)
    _, rep["test_invalid_bio_kept_as_is"] = repair_bio([t for r in data["test"] for t in r["ner"] + ["O"]])

    # ---- fixed, seeded subset for ablations
    rng = random.Random(SEED)
    subset = rng.sample(data["train"], min(SUBSET_SIZE, len(data["train"])))

    # ---- vocabularies (built from TRAIN ONLY -> no test leakage)
    wc = collections.Counter(t for r in data["train"] for t in r["tokens"])
    cc = collections.Counter(c for r in data["train"] for t in r["tokens"] for c in t)
    words = [PAD, UNK] + sorted(w for w, n in wc.items() if n >= WORD_MIN_FREQ)
    chars = [PAD, UNK] + sorted(cc)
    word2id = {w: i for i, w in enumerate(words)}
    char2id = {c: i for i, c in enumerate(chars)}
    label2id = {l: i for i, l in enumerate(NER_LABELS)}

    def oov_rate(rows):
        toks = [t for r in rows for t in r["tokens"]]
        return round(sum(t not in word2id for t in toks) / max(len(toks), 1), 4)

    # ---- save
    write_jsonl(data["train"], OUT / "train.jsonl")
    write_jsonl(subset, OUT / "train_150k.jsonl")
    write_jsonl(data["val"], OUT / "val.jsonl")
    write_jsonl(data["test"], OUT / "test.jsonl")
    write_jsonl(test_dedup, OUT / "test_dedup.jsonl")
    (VOCAB / "ner_word_vocab.json").write_text(json.dumps(word2id, ensure_ascii=False), encoding="utf-8")
    (VOCAB / "ner_char_vocab.json").write_text(json.dumps(char2id, ensure_ascii=False), encoding="utf-8")
    (LABELS / "ner_labels.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")

    def ents(rows):
        return dict(collections.Counter(t[2:] for r in rows for t in r["ner"] if t.startswith("B-")))
    rep["final"] = {name: {"sentences": len(rows), "tokens": sum(len(r["tokens"]) for r in rows),
                           "entities": ents(rows), "oov_rate": oov_rate(rows)}
                    for name, rows in (("train", data["train"]), ("train_150k", subset), ("val", data["val"]),
                                       ("test", data["test"]), ("test_dedup", test_dedup))}
    rep["vocab"] = {"words": len(word2id), "chars": len(char2id),
                    "train_word_types_total": len(wc), "labels": label2id}
    rep["tokenize_example"] = tokenize("રાહુલ અમદાવાદમાં રહે છે.")
    (REPORT / "ner_preprocessing_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("\nSaved processed data, vocabularies, label map and report.")

if __name__ == "__main__":
    main()
