"""Phase 4 - train + evaluate the BiLSTM-CRF NER model on Naamapadam (Gujarati).
The BiLSTM understands the context of words from both directions, while CRF helps predict the correct sequence of entity labels 
such as PER, ORG and LOC. During training, we use the Adam optimiser, dropout, backpropagation and early stopping to improve model
 performance. Finally, we evaluate the model using precision, recall, F1-score and accuracy, and save the best model, training history 
   prediction results.

"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BiLSTMTagger  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets/processed/ner"
VOCAB = ROOT / "models/vocabulary"
LABELS = ROOT / "models/label_mappings/ner_labels.json"
PAD_ID, UNK_ID = 0, 1
MAX_WORD_LEN = 20          # characters kept per word for the char-CNN


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


class Batcher:
    """Turns rows {tokens, <field>} into padded tensors (field = "ner" or "pos"). Encoding is done per batch to keep
    RAM low (the full train set has 6M tokens)."""

    def __init__(self, word2id, char2id, label2id, device, field="ner"):
        self.w, self.c, self.l, self.device, self.field = word2id, char2id, label2id, device, field

    def tensors(self, rows, word_dropout=0.0, rng=None):
        T = max(len(r["tokens"]) for r in rows)
        L = min(MAX_WORD_LEN, max(len(t) for r in rows for t in r["tokens"]))
        words = torch.zeros(len(rows), T, dtype=torch.long)
        chars = torch.zeros(len(rows), T, L, dtype=torch.long)
        tags = torch.zeros(len(rows), T, dtype=torch.long)
        mask = torch.zeros(len(rows), T, dtype=torch.bool)
        for i, r in enumerate(rows):
            for j, tok in enumerate(r["tokens"]):
                wid = self.w.get(tok, UNK_ID)
                # word dropout: randomly hide known words so the model learns to use spelling
                if word_dropout and wid != UNK_ID and rng.random() < word_dropout:
                    wid = UNK_ID
                words[i, j] = wid
                for k, ch in enumerate(tok[:L]):
                    chars[i, j, k] = self.c.get(ch, UNK_ID)
                tags[i, j] = self.l[r[self.field][j]]
                mask[i, j] = True
        d = self.device
        return words.to(d), chars.to(d), mask.to(d), tags.to(d)


def batches(rows, size, shuffle, rng):
    """Length-bucketed batches: shuffle, sort within chunks of 50 batches, shuffle batches."""
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    chunk = size * 50
    out = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: len(rows[i]["tokens"]))
        out += [part[k:k + size] for k in range(0, len(part), size)]
    if shuffle:
        rng.shuffle(out)
    return out


def predict(model, rows, batcher, id2label, size=128):
    model.eval()
    preds = [None] * len(rows)
    for b in batches(rows, size, shuffle=False, rng=None):
        batch = [rows[i] for i in b]
        words, chars, mask, _ = batcher.tensors(batch)
        for i, p in zip(b, model.decode(words, chars, mask)["ner"]):
            preds[i] = [id2label[x] for x in p]
    return preds


def score(rows, preds, word2id=None):
    gold = [r["ner"] for r in rows]
    n_tok = sum(len(g) for g in gold)
    res = {"precision": round(precision_score(gold, preds), 4),
           "recall": round(recall_score(gold, preds), 4),
           "f1": round(f1_score(gold, preds), 4),
           "token_accuracy": round(sum(a == b for g, p in zip(gold, preds) for a, b in zip(g, p)) / n_tok, 4)}
    rep = classification_report(gold, preds, output_dict=True, zero_division=0)
    res["per_type"] = {k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score", "support")}
                       for k, v in rep.items() if k in ("PER", "ORG", "LOC")}
    if word2id is not None:
        # entity tokens whose word is out of vocabulary: tagging accuracy on them
        # (OOV names are a central difficulty for Gujarati NER)
        hits = tot = 0
        for r, p in zip(rows, preds):
            for tok, g, q in zip(r["tokens"], r["ner"], p):
                if g != "O" and tok not in word2id:
                    tot += 1; hits += g == q
        res["oov_entity_token_accuracy"] = round(hits / tot, 4) if tot else None
        res["oov_entity_tokens"] = tot
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_150k", choices=["train_150k", "train"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=3, help="stop after N epochs without val F1 gain")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--word-dropout", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only first N train rows (smoke test)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--embeddings", default=None,
                    help="FastText word vectors saved as gensim KeyedVectors (models/embeddings/fasttext_gu_100d.kv); "
                         "initialises the word embedding layer from them")
    ap.add_argument("--tune-emb", action="store_true",
                    help="fine-tune the FastText word vectors during training (default: frozen)")
    args = ap.parse_args()

    suffix = ("_fasttext" + ("tuned" if args.tune_emb else "frozen")) if args.embeddings else ""
    run = args.run_name or f"ner_bilstm_crf_{args.train}" + suffix + (f"_limit{args.limit}" if args.limit else "")
    ckpt_dir = ROOT / "models/checkpoints" / run
    hist_path = ROOT / "results/training_history" / f"{run}.json"
    metrics_path = ROOT / "results/metrics" / f"{run}.json"
    for p in (ckpt_dir, hist_path.parent, metrics_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word2id = json.loads((VOCAB / "ner_word_vocab.json").read_text(encoding="utf-8"))
    char2id = json.loads((VOCAB / "ner_char_vocab.json").read_text(encoding="utf-8"))
    label2id = json.loads(LABELS.read_text(encoding="utf-8"))
    id2label = {i: l for l, i in label2id.items()}

    train = load_jsonl(DATA / f"{args.train}.jsonl", args.limit)
    val = load_jsonl(DATA / "val.jsonl")
    test = load_jsonl(DATA / "test.jsonl")
    test_dedup = load_jsonl(DATA / "test_dedup.jsonl")
    print(f"run={run} device={device} train={len(train)} val={len(val)} test={len(test)}")

    kv = None
    if args.embeddings:
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(str(ROOT / args.embeddings))      # FastText word vectors

    model = BiLSTMTagger(len(word2id), len(char2id), {"ner": len(label2id)},
                         word_dim=kv.vector_size if kv is not None else 100,
                         hidden=args.hidden, dropout=args.dropout).to(device)
    if kv is not None:
        mat = torch.zeros(len(word2id), kv.vector_size)
        missing = 0
        for w, i in word2id.items():
            if i > 1:                                   # PAD and UNK stay zero
                if w in kv.key_to_index:
                    mat[i] = torch.from_numpy(kv[w].copy())
                else:
                    missing += 1                        # no FastText vector -> stays zero
        model.word_emb.weight.data.copy_(mat)
        model.word_emb.weight.requires_grad = args.tune_emb
        print(f"embeddings: {args.embeddings} dim={kv.vector_size} frozen={not args.tune_emb} "
              f"words without FastText vector: {missing:,} of {len(word2id) - 2:,}")
        del kv
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    batcher = Batcher(word2id, char2id, label2id, device)

    history, best_f1, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        bar = tqdm(batches(train, args.batch_size, True, rng), desc=f"epoch {epoch}", leave=False)
        for b in bar:
            words, chars, mask, tags = batcher.tensors([train[i] for i in b], args.word_dropout, rng)
            loss = model.loss(words, chars, mask, {"ner": tags})
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item(); n += 1
            if n % 50 == 0:
                bar.set_postfix(loss=f"{total / n:.3f}")
        val_scores = score(val, predict(model, val, batcher, id2label))
        sched.step(val_scores["f1"])
        rec = {"epoch": epoch, "train_loss": round(total / max(n, 1), 4), "val": val_scores,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        hist_path.write_text(json.dumps({"args": vars(args), "history": history}, indent=2), encoding="utf-8")
        improved = val_scores["f1"] > best_f1
        print(f"epoch {epoch}: loss {rec['train_loss']:.4f}  val P/R/F1 {val_scores['precision']:.4f}/"
              f"{val_scores['recall']:.4f}/{val_scores['f1']:.4f}  {rec['seconds']}s" + ("  *best*" if improved else ""))
        if improved:
            best_f1, bad = val_scores["f1"], 0
            torch.save({"state_dict": model.state_dict(), "config": model.config, "args": vars(args),
                        "epoch": epoch, "val": val_scores, "word2id": word2id, "char2id": char2id,
                        "label2id": label2id}, ckpt_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val F1 gain for {args.patience} epochs")
                break

    # ---- final evaluation with the best checkpoint
    ck = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    test_preds = predict(model, test, batcher, id2label)
    final = {"run": run, "best_epoch": ck["epoch"], "args": vars(args),
             "val": score(val, predict(model, val, batcher, id2label), word2id),
             "test": score(test, test_preds, word2id),
             "test_dedup": score(test_dedup, predict(model, test_dedup, batcher, id2label), word2id)}
    metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    with open(metrics_path.with_name(f"{run}_test_predictions.jsonl"), "w", encoding="utf-8") as fh:
        for r, p in zip(test, test_preds):
            fh.write(json.dumps({"id": r["id"], "tokens": r["tokens"], "gold": r["ner"], "pred": p},
                                ensure_ascii=False) + "\n")
    print(json.dumps({k: final[k] for k in ("best_epoch", "val", "test", "test_dedup")}, indent=2))
    print(f"\nSaved -> {ckpt_dir / 'best.pt'}\n         {metrics_path}\n         {hist_path}")


if __name__ == "__main__":
    main()