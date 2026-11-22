"""Try the trained models on your own Gujarati text: POS tag + named entities for each word.

Run from project root:
    python src/predict.py                                   # interactive: type sentences, empty line quits
    python src/predict.py "રાહુલ ગાંધી અમદાવાદમાં રહે છે."    # one sentence / paragraph
    python src/predict.py --file my_text.txt                # every line of a UTF-8 text file

Models used (change with --ner / --pos):
    NER  models/checkpoints/ner_bilstm_crf_train/best.pt            (test F1 0.82)
    POS  models/checkpoints/pos_bilstm_crf_hi2gu_fttuned/best.pt    (~47% on real Gujarati - weak)

Text goes through the SAME tokenize() / normalize_token() used for the training data.
POS output uses the rules in pos_rules.py (function-word list + agreement with NER);
add --no-rules to see the raw POS model.
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BiLSTMTagger  # noqa: E402
from pos_rules import pos_constraints  # noqa: E402
from preprocessing import split_sentences, tokenize  # noqa: E402
from train_ner import MAX_WORD_LEN, UNK_ID  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NER = "models/checkpoints/ner_bilstm_crf_train/best.pt"
DEFAULT_POS = "models/checkpoints/pos_bilstm_crf_hi2gu_fttuned/best.pt"


class Tagger:
    def __init__(self, ckpt_path, device):
        ck = torch.load(ROOT / ckpt_path, map_location=device, weights_only=False)
        self.model = BiLSTMTagger.from_config(ck["config"]).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.w, self.c, self.device = ck["word2id"], ck["char2id"], device
        self.label2id = ck["label2id"]
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.task = next(iter(ck["config"]["tasks"]))

    @torch.no_grad()
    def __call__(self, toks, allowed=None):
        """allowed: optional list (one per token) of sets of permitted labels, or None entries.
        Forbidden labels get a large negative score, then the CRF decodes as usual."""
        L = min(MAX_WORD_LEN, max(len(t) for t in toks))
        words = torch.tensor([[self.w.get(t, UNK_ID) for t in toks]], device=self.device)
        chars = torch.zeros(1, len(toks), L, dtype=torch.long, device=self.device)
        for j, t in enumerate(toks):
            for k, ch in enumerate(t[:L]):
                chars[0, j, k] = self.c.get(ch, UNK_ID)
        mask = torch.ones(1, len(toks), dtype=torch.bool, device=self.device)
        em = self.model.emissions(words, chars, mask)[self.task]
        if allowed is not None:
            for j, ok in enumerate(allowed):
                if ok:
                    for lab, idx in self.label2id.items():
                        if lab not in ok:
                            em[0, j, idx] = -1e4
        return [self.id2label[i] for i in self.model.crfs[self.task].decode(em, mask=mask)[0]]


def entities(toks, tags):
    """BIO tags -> [(type, 'word word')]"""
    out, cur = [], None
    for t, g in zip(toks, tags):
        if g.startswith("B-") or (g.startswith("I-") and (cur is None or cur[0] != g[2:])):
            cur = (g[2:], [t]); out.append(cur)
        elif g.startswith("I-"):
            cur[1].append(t)
        else:
            cur = None
    return [(typ, " ".join(ws)) for typ, ws in out]


def show(sentence, ner, pos, rules=True):
    for sent in split_sentences(sentence):
        toks = tokenize(sent)
        if not toks:
            continue
        ner_tags = ner(toks)
        pos_tags = pos(toks, pos_constraints(toks, ner_tags) if rules else None)
        width = max(12, max(len(t) for t in toks) + 2)
        print(f"\n{'WORD'.ljust(width)}{'POS'.ljust(8)}NER")
        print("-" * (width + 14))
        for t, p, n in zip(toks, pos_tags, ner_tags):
            print(f"{t.ljust(width)}{p.ljust(8)}{n}")
        ents = entities(toks, ner_tags)
        print("Entities: " + ("  ".join(f"[{typ}] {txt}" for typ, txt in ents) if ents else "(none)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="Gujarati text; leave empty for interactive mode")
    ap.add_argument("--file", help="UTF-8 text file, one sentence or paragraph per line")
    ap.add_argument("--ner", default=DEFAULT_NER)
    ap.add_argument("--pos", default=DEFAULT_POS)
    ap.add_argument("--no-rules", action="store_true",
                    help="raw POS model output, without the word-list + NER-consistency rules")
    args = ap.parse_args()
    rules = not args.no_rules

    for stream in (sys.stdout, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner, pos = Tagger(args.ner, device), Tagger(args.pos, device)

    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                print(f"\n>>> {line.strip()}")
                show(line, ner, pos, rules)
    elif args.text:
        show(" ".join(args.text), ner, pos, rules)
    else:
        print("Type a Gujarati sentence and press Enter (empty line to quit).")
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            show(line, ner, pos, rules)


if __name__ == "__main__":
    main()
