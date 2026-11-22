"""Phase 2 - download + inspect the Gujarati subset of ai4bharat/naamapadam.
Run: python scripts/inspect_naamapadam.py
- Downloads ONLY the Gujarati zip into datasets/raw/naamapadam/ (originals never modified).
- Does not use load_dataset(): the repo relies on a loading script, which recent
  `datasets` versions no longer execute. We read the raw files directly instead.
Writes results/reports/naamapadam_gu_inspection.json + .md"""
import json, zipfile, collections, hashlib, statistics, sys
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "ai4bharat/naamapadam"
RAW = Path("datasets/raw/naamapadam"); RAW.mkdir(parents=True, exist_ok=True)
# Label order as declared in the repo's naamapadam.py ClassLabel -- verified against data below.
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

files = list_repo_files(REPO, repo_type="dataset")
print("Repo files:", files)
gu = [f for f in files if Path(f).name.lower().startswith("gu")]
if not gu:
    sys.exit("No Gujarati file found in repo listing above - report this.")
extracted = RAW / "gu_extracted"
for f in gu:
    p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW)
    print("Downloaded", p)
    if p.endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            z.extractall(extracted)

def read_records(path):
    txt = path.read_text(encoding="utf-8")
    try:  # whole-file JSON
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]

report = {"files": {}, "label_order_assumed": LABELS}
split_sentences = {}
for path in sorted(p for p in extracted.rglob("*") if p.is_file()):
    try:
        recs = read_records(path)
    except Exception as e:
        report["files"][str(path)] = f"unreadable as JSON: {e}"; continue
    keys = collections.Counter(k for r in recs for k in r.keys())
    tok_key = next(k for k in ("tokens", "words") if k in keys)
    lab_key = next(k for k in ("ner_tags", "ner", "tags") if k in keys)
    lens, mismatch, empty, bad_bio, tag_counts, ent_counts = [], 0, 0, 0, collections.Counter(), collections.Counter()
    hashes = collections.Counter(); raw_label_values = collections.Counter()
    for r in recs:
        toks, labs = r[tok_key], r[lab_key]
        if not toks: empty += 1; continue
        if len(toks) != len(labs): mismatch += 1; continue
        lens.append(len(toks))
        names = [LABELS[l] if isinstance(l, int) else l for l in labs]
        raw_label_values.update(map(str, labs))
        tag_counts.update(names)
        prev = "O"
        for n in names:
            if n.startswith("B-"): ent_counts[n[2:]] += 1
            if n.startswith("I-") and prev[2:] != n[2:]: bad_bio += 1
            prev = n
        hashes[hashlib.md5(" ".join(toks).encode()).hexdigest()] += 1
    split_sentences[path.name] = set(hashes)
    report["files"][str(path)] = {
        "records": len(recs), "fields": dict(keys), "token_field": tok_key, "label_field": lab_key,
        "empty": empty, "token_label_length_mismatch": mismatch,
        "exact_duplicate_sentences": sum(c - 1 for c in hashes.values() if c > 1),
        "tokens_total": sum(lens),
        "sent_len": {"mean": round(statistics.mean(lens), 1), "median": statistics.median(lens),
                     "max": max(lens), "p95": sorted(lens)[int(.95 * len(lens))]} if lens else None,
        "raw_label_values": dict(raw_label_values), "tag_distribution": dict(tag_counts),
        "entities_by_type": dict(ent_counts), "invalid_BIO_transitions(I-without-B)": bad_bio,
        "samples": [{"tokens": r[tok_key], "labels": [LABELS[l] if isinstance(l, int) else l for l in r[lab_key]]}
                    for r in recs[:5]],
    }
names = list(split_sentences)
report["cross_file_overlap"] = {f"{a} & {b}": len(split_sentences[a] & split_sentences[b])
                                for i, a in enumerate(names) for b in names[i + 1:]}
out = Path("results/reports"); out.mkdir(parents=True, exist_ok=True)
(out / "naamapadam_gu_inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
with open(out / "naamapadam_gu_inspection.md", "w", encoding="utf-8") as fh:
    for f, info in report["files"].items():
        fh.write(f"## {f}\n```\n{json.dumps({k: v for k, v in info.items() if k != 'samples'} if isinstance(info, dict) else info, indent=2, ensure_ascii=False)}\n```\n")
        if isinstance(info, dict):
            for s in info["samples"][:3]:
                fh.write("  ".join(f"{t}/{l}" for t, l in zip(s["tokens"], s["labels"])) + "\n\n")
    fh.write(f"## Cross-file overlap\n{report['cross_file_overlap']}\n")
print(json.dumps({f: (i if not isinstance(i, dict) else {k: v for k, v in i.items() if k != 'samples'})
                  for f, i in report["files"].items()}, indent=2, ensure_ascii=False))
print("Overlap:", report["cross_file_overlap"])
print("Saved -> results/reports/naamapadam_gu_inspection.{json,md}")
