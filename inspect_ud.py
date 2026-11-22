"""Phase 2b - download + inspect free POS data (UD Hindi HDTB, UD Gujarati GujTB),
and create a Gujarati-script copy of the Hindi data.

Run from project root:  python inspect_ud.py

Outputs
  datasets/raw/ud/                         original .conllu files (never modified)
  datasets/processed/pos_hi2gu/{train,dev,test}.jsonl   Hindi HDTB in Gujarati script
  datasets/processed/pos_gujtb/test.jsonl  real Gujarati POS (evaluation only)
  results/reports/ud_inspection.{json,md}

Decisions (documented for the paper)
  * Transliteration: Devanagari U+0900-097F -> Gujarati U+0A80-0AFF (fixed +0x180 offset).
    Code points with no Gujarati equivalent are handled explicitly (see SPECIAL) and counted.
  * Nukta dropped (not written in standard Gujarati spelling); RRA -> RA.
  * Danda '।' -> '.', double danda '॥' -> '.', matching Naamapadam's punctuation.
  * Devanagari digits -> ASCII digits (Naamapadam uses ASCII digits).
  * GujTB multi-word tokens (e.g. તેની = તે + ની) are kept as ONE surface token, as written,
    so tokenisation matches Naamapadam. Its tag = tag of the first (head) part; the full
    part sequence is kept in `pos_parts` so the choice can be revisited.
"""
import json, collections, unicodedata, urllib.request
from pathlib import Path

ROOT = Path(".")
RAW = ROOT / "datasets/raw/ud"; RAW.mkdir(parents=True, exist_ok=True)
OUT_HI = ROOT / "datasets/processed/pos_hi2gu"; OUT_HI.mkdir(parents=True, exist_ok=True)
OUT_GU = ROOT / "datasets/processed/pos_gujtb"; OUT_GU.mkdir(parents=True, exist_ok=True)
REP = ROOT / "results/reports"; REP.mkdir(parents=True, exist_ok=True)

GH = "https://raw.githubusercontent.com/UniversalDependencies"
FILES = {
    "hi_train": (f"{GH}/UD_Hindi-HDTB/master/hi_hdtb-ud-train.conllu", "hi_hdtb-ud-train.conllu"),
    "hi_dev":   (f"{GH}/UD_Hindi-HDTB/master/hi_hdtb-ud-dev.conllu",   "hi_hdtb-ud-dev.conllu"),
    "hi_test":  (f"{GH}/UD_Hindi-HDTB/master/hi_hdtb-ud-test.conllu",  "hi_hdtb-ud-test.conllu"),
    "gu_test":  (f"{GH}/UD_Gujarati-GujTB/master/gu_gujtb-ud-test.conllu", "gu_gujtb-ud-test.conllu"),
    "hi_license": (f"{GH}/UD_Hindi-HDTB/master/LICENSE.txt", "UD_Hindi-HDTB_LICENSE.txt"),
    "gu_license": (f"{GH}/UD_Gujarati-GujTB/master/LICENSE.txt", "UD_Gujarati-GujTB_LICENSE.txt"),
}

def download():
    for key, (url, name) in FILES.items():
        p = RAW / name
        if p.exists() and p.stat().st_size > 0:
            print("exists  ", p); continue
        try:
            urllib.request.urlretrieve(url, p); print("download", p, p.stat().st_size, "bytes")
        except Exception as e:
            print("FAILED  ", url, e)

# ---------- transliteration ----------
SPECIAL = {
    "।": ".", "॥": ".",            # danda, double danda
    # Gujarati orthography rarely writes the nukta: map nukta letters to the base letter
    # and drop the combining nukta, so spellings match native Gujarati text.
    "क़": "ક", "ख़": "ખ", "ग़": "ગ", "ज़": "જ",
    "ड़": "ડ", "ढ़": "ઢ", "फ़": "ફ", "य़": "ય",
    "़": "", "ऱ": "ર",     # nukta sign dropped; RRA -> RA
}
for d in range(10):
    SPECIAL[chr(0x0966 + d)] = str(d)
unmapped = collections.Counter()

def translit(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif 0x0900 <= cp <= 0x097F:
            g = chr(cp + 0x180)
            if unicodedata.name(g, None) is None:   # no Gujarati counterpart
                unmapped[ch] += 1; out.append(ch)
            else:
                out.append(g)
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))

# ---------- CoNLL-U reading ----------
def read_conllu(path):
    sents, cur, meta, mwt = [], [], {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if cur: sents.append({"id": meta.get("sent_id"), "toks": cur})
            cur, meta, mwt = [], {}, None; continue
        if line.startswith("#"):
            if "=" in line:
                k, v = line[1:].split("=", 1); meta[k.strip()] = v.strip()
            continue
        c = line.split("\t")
        if "-" in c[0]:                      # multi-word token
            a, b = map(int, c[0].split("-"))
            mwt = {"form": c[1], "end": b, "upos": [], "xpos": []}
            continue
        if "." in c[0]: continue             # empty nodes
        idx = int(c[0])
        if mwt is not None:
            mwt["upos"].append(c[3]); mwt["xpos"].append(c[4])
            if idx == mwt["end"]:
                cur.append({"form": mwt["form"], "upos": mwt["upos"][0], "xpos": mwt["xpos"][0],
                            "pos_parts": mwt["upos"]})
                mwt = None
            continue
        cur.append({"form": c[1], "upos": c[3], "xpos": c[4], "pos_parts": [c[3]]})
    if cur: sents.append({"id": meta.get("sent_id"), "toks": cur})
    return sents

def stats(sents):
    lens = [len(s["toks"]) for s in sents]
    upos = collections.Counter(t["upos"] for s in sents for t in s["toks"])
    xpos = collections.Counter(t["xpos"] for s in sents for t in s["toks"])
    mwt = sum(len(t["pos_parts"]) > 1 for s in sents for t in s["toks"])
    word_tags = collections.defaultdict(collections.Counter)
    for s in sents:
        for t in s["toks"]:
            word_tags[t["form"]][t["upos"]] += 1
    ambiguous = {w: dict(c) for w, c in word_tags.items() if len(c) > 1}
    amb_tokens = sum(sum(c.values()) for w, c in word_tags.items() if len(c) > 1)
    texts = collections.Counter(" ".join(t["form"] for t in s["toks"]) for s in sents)
    top_amb = sorted(ambiguous.items(), key=lambda kv: -sum(kv[1].values()))[:15]
    return {
        "sentences": len(sents), "tokens": sum(lens),
        "sent_len_mean": round(sum(lens) / max(len(lens), 1), 1), "sent_len_max": max(lens, default=0),
        "duplicate_sentences": sum(c - 1 for c in texts.values() if c > 1),
        "multiword_tokens": mwt, "upos_distribution": dict(upos.most_common()),
        "xpos_distribution": dict(xpos.most_common()) if any(k != "_" for k in xpos) else "not available",
        "vocab_size": len(word_tags), "ambiguous_word_types": len(ambiguous),
        "ambiguous_token_share": round(amb_tokens / max(sum(lens), 1), 3),
        "top_ambiguous_words": dict(top_amb),
    }

def write_jsonl(sents, path):
    with open(path, "w", encoding="utf-8") as fh:
        for s in sents:
            fh.write(json.dumps({"id": s["id"], "tokens": [t["form"] for t in s["toks"]],
                                 "upos": [t["upos"] for t in s["toks"]],
                                 "xpos": [t["xpos"] for t in s["toks"]],
                                 "pos_parts": [t["pos_parts"] for t in s["toks"]]},
                                ensure_ascii=False) + "\n")

def main():
    download()
    report = {"files": {}}
    splits = {}
    for split in ("train", "dev", "test"):
        hi = read_conllu(RAW / f"hi_hdtb-ud-{split}.conllu")
        for s in hi:
            for t in s["toks"]:
                t["form"] = translit(t["form"])
        splits[split] = hi
        write_jsonl(hi, OUT_HI / f"{split}.jsonl")
        report["files"][f"hi2gu_{split}"] = stats(hi)
    gu = read_conllu(RAW / "gu_gujtb-ud-test.conllu")
    write_jsonl(gu, OUT_GU / "test.jsonl")
    report["files"]["gujtb_test"] = stats(gu)

    # leakage check across Hindi splits
    key = lambda s: " ".join(t["form"] for t in s["toks"])
    sets = {k: {key(s) for s in v} for k, v in splits.items()}
    report["hi_split_overlap"] = {"train&dev": len(sets["train"] & sets["dev"]),
                                  "train&test": len(sets["train"] & sets["test"]),
                                  "dev&test": len(sets["dev"] & sets["test"])}
    # vocabulary overlap: how much real-Gujarati vocabulary the transliterated Hindi covers
    hi_vocab = {t["form"] for s in splits["train"] for t in s["toks"]}
    gu_toks = [t["form"] for s in gu for t in s["toks"]]
    report["gujtb_token_coverage_by_hi2gu_train_vocab"] = round(
        sum(w in hi_vocab for w in gu_toks) / max(len(gu_toks), 1), 3)
    report["unmapped_devanagari_chars"] = {f"U+{ord(c):04X} {unicodedata.name(c, '?')}": n
                                           for c, n in unmapped.most_common()}
    report["upos_tagset_hi_minus_gu"] = sorted(set(report["files"]["hi2gu_train"]["upos_distribution"])
                                               - set(report["files"]["gujtb_test"]["upos_distribution"]))
    report["upos_tagset_gu_minus_hi"] = sorted(set(report["files"]["gujtb_test"]["upos_distribution"])
                                               - set(report["files"]["hi2gu_train"]["upos_distribution"]))
    report["samples"] = {
        "hi2gu_train": [" ".join(f"{t['form']}/{t['upos']}" for t in s["toks"]) for s in splits["train"][:3]],
        "gujtb_test": [" ".join(f"{t['form']}/{t['upos']}" for t in s["toks"]) for s in gu[:3]],
    }

    (REP / "ud_inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(REP / "ud_inspection.md", "w", encoding="utf-8") as fh:
        fh.write("# UD POS data inspection\n\n")
        for name, st in report["files"].items():
            fh.write(f"## {name}\n")
            for k in ("sentences", "tokens", "sent_len_mean", "sent_len_max", "duplicate_sentences",
                      "multiword_tokens", "vocab_size", "ambiguous_word_types", "ambiguous_token_share"):
                fh.write(f"- {k}: {st[k]}\n")
            fh.write(f"- UPOS: {st['upos_distribution']}\n\n")
        for k in ("hi_split_overlap", "gujtb_token_coverage_by_hi2gu_train_vocab",
                  "unmapped_devanagari_chars", "upos_tagset_hi_minus_gu", "upos_tagset_gu_minus_hi"):
            fh.write(f"**{k}**: {report[k]}\n\n")
        fh.write("## Samples\n")
        for k, v in report["samples"].items():
            fh.write(f"### {k}\n" + "\n".join(f"- {x}" for x in v) + "\n")
    summary = {k: v for k, v in report.items() if k not in ("files", "samples")}
    summary["sizes"] = {k: (v["sentences"], v["tokens"]) for k, v in report["files"].items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSamples:"); [print(" ", x) for v in report["samples"].values() for x in v]
    print("\nSaved -> results/reports/ud_inspection.{json,md}")

if __name__ == "__main__":
    main()
