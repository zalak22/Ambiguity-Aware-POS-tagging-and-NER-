"""Phase 5 - POS tagger: BiLSTM-CRF trained on Hindi HDTB transliterated to Gujarati script,

The BiLSTM understands the surrounding context of each word and predicts grammatical categories such as

 NOUN, VERB, ADJ and PRON, while CRF helps generate a suitable sequence of tags. We also compare our model with a Most Frequent Tag (MFT)
   baseline to understand how effectively our model performs. Finally, we evaluate it using accuracy, macro F1-score, precision, 
   recall and OOV accuracy, and save the best model, evaluation metrics and Gujarati POS predictions.

"""
import argparse, collections, json, random, sys, time
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BiLSTMTagger  # noqa: E402
from preprocessing import normalize_token, PAD, UNK  # noqa: E402
from train_ner import Batcher, batches, load_jsonl, set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HI = ROOT / "datasets/processed/pos_hi2gu"
GU = ROOT / "datasets/processed/pos_gujtb/test.jsonl"
VOCAB = ROOT / "models/vocabulary"
LABELS = ROOT / "models/label_mappings"
UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
        "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


def prepare(rows):
    """Normalise tokens exactly like the NER data; drop tokens that become empty."""
    out, dropped, unknown_tag = [], 0, 0
    for r in rows:
        toks, tags = [], []
        for t, p in zip(r["tokens"], r["upos"]):
            t = normalize_token(t)
            if not t:
                dropped += 1; continue
            if p not in UPOS:
                unknown_tag += 1; p = "X"
            toks.append(t); tags.append(p)
        if toks:
            out.append({"id": r["id"], "tokens": toks, "pos": tags})
    return out, {"empty_tokens_dropped": dropped, "unknown_tag_to_X": unknown_tag}


def predict(model, rows, batcher, id2label):
    model.eval()
    preds = [None] * len(rows)
    for b in batches(rows, 128, shuffle=False, rng=None):
        words, chars, mask, _ = batcher.tensors([rows[i] for i in b])
        for i, p in zip(b, model.decode(words, chars, mask)["pos"]):
            preds[i] = [id2label[x] for x in p]
    return preds


def mft_predict(rows, word_tags, default):
    """Most-frequent-tag baseline: each known word gets its most common train tag."""
    return [[word_tags[t].most_common(1)[0][0] if t in word_tags else default for t in r["tokens"]]
            for r in rows]


def score(rows, preds, word_tags):
    gold = [g for r in rows for g in r["pos"]]
    pred = [p for ps in preds for p in ps]
    toks = [t for r in rows for t in r["tokens"]]

    def acc(sel):
        idx = [i for i in range(len(gold)) if sel(toks[i])]
        return {"accuracy": round(sum(gold[i] == pred[i] for i in idx) / len(idx), 4) if idx else None,
                "tokens": len(idx)}

    labels = [l for l in UPOS if l in set(gold)]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    confusions = collections.Counter((g, q) for g, q in zip(gold, pred) if g != q)
    return {
        "accuracy": round(sum(g == q for g, q in zip(gold, pred)) / len(gold), 4),
        "macro_f1": round(float(f.mean()), 4),
        "tokens": len(gold),
        "oov": acc(lambda t: t not in word_tags),
        "ambiguous": acc(lambda t: t in word_tags and len(word_tags[t]) > 1),
        "unambiguous": acc(lambda t: t in word_tags and len(word_tags[t]) == 1),
        "per_tag": {l: {"precision": round(float(a), 4), "recall": round(float(b), 4),
                        "f1": round(float(c), 4), "support": int(d)}
                    for l, a, b, c, d in zip(labels, p, r, f, s)},
        "top_confusions(gold->pred)": {f"{g}->{q}": n for (g, q), n in confusions.most_common(10)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--word-dropout", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--word-min-freq", type=int, default=2)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only first N train rows (smoke test)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--embeddings", default=None,
                    help="FastText word vectors saved as gensim KeyedVectors (models/embeddings/fasttext_gu_100d.kv). "
                         "Adds the Gujarati corpus vocabulary to the word vocab and initialises word vectors from it.")
    ap.add_argument("--tune-emb", action="store_true",
                    help="fine-tune pretrained word vectors (default: frozen, so Gujarati words that never "
                         "occur in the Hindi training data stay aligned with the ones that do)")
    args = ap.parse_args()

    suffix = ("_fasttext" + ("tuned" if args.tune_emb else "frozen")) if args.embeddings else ""
    run = args.run_name or "pos_bilstm_crf_hi2gu" + suffix + (f"_limit{args.limit}" if args.limit else "")
    ckpt_dir = ROOT / "models/checkpoints" / run
    hist_path = ROOT / "results/training_history" / f"{run}.json"
    metrics_path = ROOT / "results/metrics" / f"{run}.json"
    for p in (ckpt_dir, hist_path.parent, metrics_path.parent, VOCAB, LABELS):
        p.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prep = {}
    train, prep["train"] = prepare(load_jsonl(HI / "train.jsonl"))
    dev, prep["dev"] = prepare(load_jsonl(HI / "dev.jsonl"))
    test, prep["test"] = prepare(load_jsonl(HI / "test.jsonl"))
    gujtb, prep["gujtb"] = prepare(load_jsonl(GU))

    # leakage: no train sentence may also appear in any evaluation set
    key = lambda r: " ".join(r["tokens"])
    held_out = {key(r) for r in dev + test + gujtb}
    before = len(train)
    train = [r for r in train if key(r) not in held_out]
    prep["train_removed_overlap_with_eval"] = before - len(train)
    if args.limit:
        train = train[:args.limit]

    # vocabularies + tag lexicon from TRAIN only
    wc = collections.Counter(t for r in train for t in r["tokens"])
    cc = collections.Counter(c for r in train for t in r["tokens"] for c in t)
    vocab_words = {w for w, n in wc.items() if n >= args.word_min_freq}
    kv = None
    if args.embeddings:
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(str(ROOT / args.embeddings))      # FastText word vectors
        # Gujarati corpus vocabulary (Naamapadam TRAIN words, freq >= 2) - no test text involved
        ner_vocab = json.loads((VOCAB / "ner_word_vocab.json").read_text(encoding="utf-8"))
        vocab_words |= set(ner_vocab) - {PAD, UNK}
    word2id = {w: i for i, w in enumerate([PAD, UNK] + sorted(vocab_words))}
    char2id = {c: i for i, c in enumerate([PAD, UNK] + sorted(cc))}
    label2id = {l: i for i, l in enumerate(UPOS)}
    id2label = {i: l for l, i in label2id.items()}
    word_tags = collections.defaultdict(collections.Counter)
    for r in train:
        for t, p in zip(r["tokens"], r["pos"]):
            word_tags[t][p] += 1
    word_tags = dict(word_tags)
    default_tag = collections.Counter(p for r in train for p in r["pos"]).most_common(1)[0][0]
    if not args.limit:
        prefix = "pos" if not args.embeddings else run
        (VOCAB / f"{prefix}_word_vocab.json").write_text(json.dumps(word2id, ensure_ascii=False), encoding="utf-8")
        (VOCAB / f"{prefix}_char_vocab.json").write_text(json.dumps(char2id, ensure_ascii=False), encoding="utf-8")
        (LABELS / "pos_labels.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    print(f"run={run} device={device} train={len(train)} dev={len(dev)} test={len(test)} gujtb={len(gujtb)} "
          f"words={len(word2id)} chars={len(char2id)}")
    print("preparation:", json.dumps(prep))

    model = BiLSTMTagger(len(word2id), len(char2id), {"pos": len(UPOS)},
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
        gu_toks = [t for r in gujtb for t in r["tokens"]]
        print(f"embeddings: {args.embeddings} dim={kv.vector_size} frozen={not args.tune_emb} "
              f"words without FastText vector: {missing:,} "
              f"gujtb tokens in word vocab: {sum(t in word2id for t in gu_toks) / len(gu_toks):.3f}")
        del kv
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    batcher = Batcher(word2id, char2id, label2id, device, field="pos")

    history, best, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b in tqdm(batches(train, args.batch_size, True, rng), desc=f"epoch {epoch}", leave=False):
            words, chars, mask, tags = batcher.tensors([train[i] for i in b], args.word_dropout, rng)
            loss = model.loss(words, chars, mask, {"pos": tags})
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item(); n += 1
        dev_acc = score(dev, predict(model, dev, batcher, id2label), word_tags)["accuracy"]
        # GujTB tracked for the learning curve only - never used for selection
        gu_acc = score(gujtb, predict(model, gujtb, batcher, id2label), word_tags)["accuracy"]
        sched.step(dev_acc)
        rec = {"epoch": epoch, "train_loss": round(total / max(n, 1), 4), "dev_accuracy": dev_acc,
               "gujtb_accuracy_monitor_only": gu_acc, "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        hist_path.write_text(json.dumps({"args": vars(args), "history": history}, indent=2), encoding="utf-8")
        improved = dev_acc > best
        print(f"epoch {epoch}: loss {rec['train_loss']:.4f}  dev acc {dev_acc:.4f}  "
              f"(gujtb {gu_acc:.4f})  {rec['seconds']}s" + ("  *best*" if improved else ""))
        if improved:
            best, bad = dev_acc, 0
            torch.save({"state_dict": model.state_dict(), "config": model.config, "args": vars(args),
                        "epoch": epoch, "dev_accuracy": dev_acc, "word2id": word2id, "char2id": char2id,
                        "label2id": label2id}, ckpt_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no dev gain for {args.patience} epochs")
                break

    ck = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    final = {"run": run, "best_epoch": ck["epoch"], "args": vars(args), "preparation": prep,
             "tags_never_seen_in_train": sorted({p for r in gujtb for p in r["pos"]} - {p for r in train for p in r["pos"]}),
             "bilstm_crf": {}, "most_frequent_tag_baseline": {}}
    gu_preds = None
    for name, rows in (("hi2gu_dev", dev), ("hi2gu_test", test), ("gujtb_test", gujtb)):
        preds = predict(model, rows, batcher, id2label)
        if name == "gujtb_test":
            gu_preds = preds
        final["bilstm_crf"][name] = score(rows, preds, word_tags)
        final["most_frequent_tag_baseline"][name] = score(rows, mft_predict(rows, word_tags, default_tag), word_tags)
    metrics_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(metrics_path.with_name(f"{run}_gujtb_predictions.jsonl"), "w", encoding="utf-8") as fh:
        for r, p in zip(gujtb, gu_preds):
            fh.write(json.dumps({"id": r["id"], "tokens": r["tokens"], "gold": r["pos"], "pred": p},
                                ensure_ascii=False) + "\n")

    print(f"\nbest epoch {ck['epoch']}")
    print(f"{'split':<12}{'model':<8}{'acc':>8}{'macroF1':>9}{'OOV':>8}{'ambig':>8}{'unambig':>9}")
    for name in ("hi2gu_dev", "hi2gu_test", "gujtb_test"):
        for label, key_ in (("bilstm", "bilstm_crf"), ("MFT", "most_frequent_tag_baseline")):
            s = final[key_][name]
            print(f"{name:<12}{label:<8}{s['accuracy']:>8}{s['macro_f1']:>9}{str(s['oov']['accuracy']):>8}"
                  f"{str(s['ambiguous']['accuracy']):>8}{str(s['unambiguous']['accuracy']):>9}")
    g = final["bilstm_crf"]["gujtb_test"]
    print(f"\ngujtb tokens: {g['tokens']}  oov={g['oov']['tokens']} ambiguous={g['ambiguous']['tokens']}")
    print("gujtb top confusions:", g["top_confusions(gold->pred)"])
    print("tags in gujtb never seen in train:", final["tags_never_seen_in_train"])
    print(f"\nSaved -> {ckpt_dir / 'best.pt'}\n         {metrics_path}\n         {hist_path}")


if __name__ == "__main__":
    main()