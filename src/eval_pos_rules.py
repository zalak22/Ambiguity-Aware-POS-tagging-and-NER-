"""Measure the POS rules (pos_rules.py) on the real-Gujarati GujTB test set.

Run from project root:   python src/eval_pos_rules.py

Compares 4 settings with the same trained models (no retraining, no tuning on GujTB):
    raw            POS model alone
    lexicon        + closed-class word list
    ner            + NER-consistency rule (NER entity -> PROPN, NER O -> not PROPN)
    lexicon+ner    both
Writes results/metrics/pos_rules_gujtb.json
"""
import collections, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pos_rules import LEXICON, pos_constraints  # noqa: E402
from predict import DEFAULT_NER, DEFAULT_POS, Tagger  # noqa: E402
from train_ner import load_jsonl  # noqa: E402
from train_pos import GU, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = {"raw": (False, False), "lexicon": (True, False), "ner": (False, True), "lexicon+ner": (True, True)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner, pos = Tagger(DEFAULT_NER, device), Tagger(DEFAULT_POS, device)
    rows, _ = prepare(load_jsonl(GU))
    ner_tags = [ner(r["tokens"]) for r in rows]
    gold = [g for r in rows for g in r["pos"]]
    toks = [t for r in rows for t in r["tokens"]]
    res = {"tokens": len(gold), "lexicon_size": len(LEXICON),
           "tokens_covered_by_lexicon": sum(t in LEXICON for t in toks)}
    for name, (lex, use_ner) in SETTINGS.items():
        pred = [p for r, n in zip(rows, ner_tags)
                for p in pos(r["tokens"], pos_constraints(r["tokens"], n, lex, use_ner) if (lex or use_ner) else None)]
        n_propn = sum(p == "PROPN" for p in pred)
        hit_propn = sum(p == g == "PROPN" for p, g in zip(pred, gold))
        res[name] = {
            "accuracy": round(sum(p == g for p, g in zip(pred, gold)) / len(gold), 4),
            "predicted_PROPN": n_propn,
            "PROPN_precision": round(hit_propn / n_propn, 4) if n_propn else None,
            "PROPN_recall": round(hit_propn / max(sum(g == "PROPN" for g in gold), 1), 4),
            "top_confusions": {f"{g}->{p}": c for (g, p), c in
                               collections.Counter((g, p) for g, p in zip(gold, pred) if g != p).most_common(6)},
        }
    res["gold_PROPN"] = sum(g == "PROPN" for g in gold)
    (ROOT / "results/metrics/pos_rules_gujtb.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"GujTB: {res['tokens']} tokens, gold PROPN {res['gold_PROPN']}, "
          f"lexicon covers {res['tokens_covered_by_lexicon']} tokens")
    print(f"{'setting':<14}{'accuracy':>9}{'PROPN pred':>12}{'PROPN P':>9}{'PROPN R':>9}")
    for name in SETTINGS:
        s = res[name]
        print(f"{name:<14}{s['accuracy']:>9}{s['predicted_PROPN']:>12}{str(s['PROPN_precision']):>9}{s['PROPN_recall']:>9}")
    print("\nlexicon+ner top confusions:", res["lexicon+ner"]["top_confusions"])
    print("Saved -> results/metrics/pos_rules_gujtb.json")


if __name__ == "__main__":
    main()
