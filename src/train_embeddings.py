"""FastText word embeddings for Gujarati.

Loads the pre-trained FastText model (models/embeddings/fasttext_gu_100d.model)
and exports its word vectors as a Gensim KeyedVectors file
(models/embeddings/fasttext_gu_100d.kv) so that train_ner.py and train_pos.py
can consume them without re-training.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from gensim.models import FastText, KeyedVectors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import normalize_token  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "datasets/processed/embeddings_corpus.txt"
OUT = ROOT / "models/embeddings"
REPORT = ROOT / "results/reports/embeddings_fasttext_report.json"
SOURCES = {"naamapadam_train": ROOT / "datasets/processed/ner/train.jsonl",
           "hi2gu_train": ROOT / "datasets/processed/pos_hi2gu/train.jsonl"}
GUJTB = ROOT / "datasets/processed/pos_gujtb/test.jsonl"
PROBES = []
SEED = 42
DIM = 100        # embedding size (same as before, so model.py needs no change)
FASTTEXT_MODEL = OUT / "fasttext_gu_100d.model"


def tokens(path):
    """Reads a JSONL dataset, extracts tokens from each sentence and normalizes them."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            toks = [normalize_token(t) for t in json.loads(line)["tokens"]]
            toks = [t for t in toks if t]
            if toks:
                yield toks


def main():
    """Load the pre-trained FastText model and export its KeyedVectors."""
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"seed": SEED, "method": "FastText (pre-trained gensim model)", "sources": {}}

    # ---- collect corpus statistics (sentences / tokens) for the report
    docs = []
    with open(CORPUS, "w", encoding="utf-8") as fh:
        for name, path in SOURCES.items():
            n_s = n_t = 0
            for toks in tokens(path):
                line = " ".join(toks)
                fh.write(line + "\n")
                docs.append(line)
                n_s += 1; n_t += len(toks)
            rep["sources"][name] = {"sentences": n_s, "tokens": n_t}
    print("corpus:", rep["sources"])

    rep["params"] = dict(dim=DIM, model_file=str(FASTTEXT_MODEL))
    t0 = time.time()

    # ---- load the pre-trained FastText model and copy its word vectors
    print(f"Loading FastText model from {FASTTEXT_MODEL} ...")
    ft = FastText.load(str(FASTTEXT_MODEL))
    wv = ft.wv          # FastText KeyedVectors (includes subword/OOV vectors)

    words = list(wv.key_to_index.keys())
    # Retrieve the in-vocabulary vectors as a float32 matrix
    vecs = np.array([wv[w] for w in words], dtype=np.float32)
    # Unit-normalise so cosine similarities stay in [−1, 1]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms + 1e-8

    rep["train_seconds"] = round(time.time() - t0, 1)
    rep["vocab_size"] = len(words)
    print(f"FastText model: {len(words):,} words, vector size {wv.vector_size}")

    # ---- store as KeyedVectors so train_ner.py / train_pos.py stay unchanged
    # Gensim's KeyedVectors() to store each word with its 100-dimensional embedding
    # and save the model as fasttext_gu_100d.kv
    kv = KeyedVectors(vector_size=wv.vector_size)
    kv.add_vectors(words, vecs)
    kv.save(str(OUT / "fasttext_gu_100d.kv"))

    # ---- checks: coverage of real-Gujarati test words + nearest neighbours
    gu = [t for toks in tokens(GUJTB) for t in toks]
    rep["gujtb_token_coverage"] = round(sum(t in kv.key_to_index for t in gu) / len(gu), 4)
    rep["gujtb_token_coverage_note"] = "words without a FastText vector start from zero in train_pos.py"
    rep["nearest_neighbours"] = {w: [(n, round(float(s), 3)) for n, s in kv.most_similar(w, topn=6)]
                                 for w in PROBES if w in kv.key_to_index}
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "nearest_neighbours"}, indent=2, ensure_ascii=False))
    for w, nn in rep["nearest_neighbours"].items():
        print(f"  {w}: " + ", ".join(n for n, _ in nn))
    print(f"\nSaved -> {OUT / 'fasttext_gu_100d.kv'}\n         {REPORT}")


if __name__ == "__main__":
    main()