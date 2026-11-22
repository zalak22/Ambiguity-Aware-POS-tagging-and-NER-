# Ambiguity-Aware Joint Gujarati POS Tagging and NER using BiLSTM-CRF

A research project for **Part-of-Speech (POS) tagging** and **Named Entity Recognition (NER)** on Gujarati text using a BiLSTM-CRF architecture with FastText embeddings.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Setup from Scratch](#setup-from-scratch)
- [Dataset Download](#dataset-download)
- [Run Order](#run-order)
- [Results](#results)
- [Try it on Your Own Text](#try-it-on-your-own-text)
- [Known Issues](#known-issues)

---

## Project Overview

| Component | Description |
|-----------|-------------|
| **NER** | BiLSTM-CRF trained on [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) (Gujarati), entities: PER, ORG, LOC |
| **POS** | BiLSTM-CRF trained on Hindi HDTB transliterated to Gujarati script, evaluated on real Gujarati (GujTB) |
| **Embeddings** | FastText 100-d (skip-gram, char 2-5 grams) pre-trained on Gujarati corpus |
| **Rules** | Post-decode constraints: closed-class word list + NER consistency |

---

## Setup from Scratch

### 1. Clone the repository
```bash
git clone https://github.com/zalak22/Ambiguity-Aware-POS-tagging-and-NER-.git
cd Ambiguity-Aware-POS-tagging-and-NER-
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch (do this FIRST, before requirements.txt)
Go to [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and pick the right command for your system.

**Example — CUDA 12.8 (NVIDIA GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
**Example — CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify environment
```bash
python env_check.py
```
→ Writes `results/reports/env_report.json`

---

## Dataset Download

> ⚠️ Large dataset files are **not included** in this repo (GitHub's 100 MB limit). You must download them manually.

### NER — Naamapadam (Gujarati)
```bash
# via Hugging Face datasets library
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('ai4bharat/naamapadam', 'gu')
ds.save_to_disk('datasets/raw/naamapadam')
"
```
Or download directly from: https://huggingface.co/datasets/ai4bharat/naamapadam

Place the raw files under: `datasets/raw/naamapadam/`

### POS — Universal Dependencies (Hindi HDTB + GujTB)
- **Hindi HDTB** (transliterated to Gujarati): https://universaldependencies.org/treebanks/hi_hdtb/
- **GujTB** (real Gujarati test set): https://universaldependencies.org/treebanks/gu_gtb/

Place files under: `datasets/raw/`

### FastText Model (pre-trained)
The FastText model (`fasttext_gu_100d.model`) is too large for GitHub (~270 MB).
Download it and place it at:
```
models/embeddings/fasttext_gu_100d.model
models/embeddings/fasttext_gu_100d.model.wv.vectors_ngrams.npy
```
> Contact the repo owner, or re-train from scratch:
> ```bash
> # Only if you don't have the model files — this will train FastText
> python src/train_embeddings.py
> ```

---

## Run Order

Run all commands from the **project root folder**.

### Step 1 — Inspect raw datasets
```bash
python inspect_naamapadam.py    # → results/reports/naamapadam_gu_inspection.{json,md}
python inspect_ud.py            # → datasets/processed/pos_hi2gu/, pos_gujtb/
```

### Step 2 — Preprocess NER data
```bash
python src/preprocessing.py
```
Generates:
- `datasets/processed/ner/` (train / val / test / train_150k .jsonl)
- `models/vocabulary/` (word + char vocabs)
- `models/label_mappings/`
- `results/reports/ner_preprocessing_report.json`

### Step 3 — Build FastText KeyedVectors file
```bash
python src/train_embeddings.py
```
Loads `models/embeddings/fasttext_gu_100d.model` and exports `models/embeddings/fasttext_gu_100d.kv`.
> This does **not** re-train the model. It converts the existing FastText model to a `.kv` file for faster loading.

### Step 4 — Train NER model
```bash
# Full dataset (~469k sentences, recommended)
python src/train_ner.py --train train

# Smaller 150k subset (faster)
python src/train_ner.py --train train_150k

# With FastText embeddings (better OOV handling)
python src/train_ner.py --train train --embeddings models/embeddings/fasttext_gu_100d.kv

# Smoke test (2000 rows, 1 epoch — just to verify it runs)
python src/train_ner.py --limit 2000 --epochs 1
```
→ Saves best checkpoint to `models/checkpoints/<run>/best.pt`

### Step 5 — Train POS model
```bash
# Baseline (random init)
python src/train_pos.py

# With FastText embeddings, frozen
python src/train_pos.py --embeddings models/embeddings/fasttext_gu_100d.kv

# With FastText embeddings, fine-tuned (best results)
python src/train_pos.py --embeddings models/embeddings/fasttext_gu_100d.kv --tune-emb
```

### Step 6 — Evaluate post-decode rules
```bash
python src/eval_pos_rules.py
```

---

## Results

### NER (entity-level F1, seqeval)
| Run | Train sents | Val F1 | Test F1 | Test-dedup F1 |
|-----|------------|--------|---------|----------------|
| `ner_bilstm_crf_train_150k` | 150,000 | 0.766 | 0.801 | 0.806 |
| `ner_bilstm_crf_train` | 468,881 | 0.780 | 0.824 | 0.830 |

### POS (accuracy on GujTB — real Gujarati)
| Setting | GujTB acc | GujTB macro-F1 | GujTB OOV acc |
|---------|-----------|----------------|---------------|
| Most-frequent-tag baseline | 0.432 | 0.236 | 0.250 |
| BiLSTM-CRF (random init) | 0.372 | 0.232 | 0.203 |
| + FastText frozen | 0.415 | 0.261 | 0.257 |
| + FastText fine-tuned | **0.465** | **0.292** | **0.323** |
| + FastText fine-tuned + rules | **0.687** | — | — |

---

## Try it on Your Own Text

```bash
# Interactive mode (type a sentence, press Enter; empty line to quit)
python src/predict.py

# Single sentence
python src/predict.py "રાહુલ ગાંધી અમદાવાદમાં રહે છે."

# From a text file (one sentence per line, UTF-8)
python src/predict.py --file my_text.txt
```

Prints POS + NER per word and the named entities found.
Uses the full-data NER model and FastText fine-tuned POS model by default.
Use `--ner` / `--pos` flags to point to other checkpoints.

---

## Known Issues

- **Windows + CUDA crash on exit**: On Windows with `torch 2.x + cuDNN 9.x`, using `nn.LSTM(num_layers>1, dropout>0)` on CUDA crashes the process at exit (`0xC0000409`). Fixed in `src/model.py` by stacking single-layer BiLSTMs with dropout between them.
- **GujTB POS accuracy is low (0.465 raw)**: The model is trained on Hindi transliterated to Gujarati, not real Gujarati. The rule-based post-processing raises it to **0.687**. Real Gujarati training data (LDC-IL) would significantly improve performance.

---

## Project Structure

```
├── src/
│   ├── preprocessing.py       # NER data preprocessing
│   ├── train_embeddings.py    # FastText → KeyedVectors export
│   ├── train_ner.py           # BiLSTM-CRF NER training
│   ├── train_pos.py           # BiLSTM-CRF POS training
│   ├── model.py               # BiLSTMTagger model definition
│   ├── predict.py             # Inference on new Gujarati text
│   ├── pos_rules.py           # Post-decode constraint rules
│   └── eval_pos_rules.py      # Rule evaluation script
├── datasets/
│   ├── raw/                   # Raw downloaded datasets (NOT in repo)
│   └── processed/             # Preprocessed .jsonl files (large files NOT in repo)
├── models/
│   ├── embeddings/            # FastText model + .kv file (NOT in repo — too large)
│   ├── checkpoints/           # Trained model .pt files (NOT in repo)
│   ├── vocabulary/            # word2id / char2id JSON files
│   └── label_mappings/        # label2id JSON files
├── results/
│   ├── metrics/               # Per-run evaluation JSON
│   ├── training_history/      # Loss / accuracy per epoch JSON
│   └── reports/               # Dataset inspection reports
├── requirements.txt
└── env_check.py
```

---

## Datasets Credit
- **NER**: [ai4bharat/naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) — AI4Bharat
- **POS (Hindi)**: [UD Hindi HDTB](https://universaldependencies.org/treebanks/hi_hdtb/) — Universal Dependencies
- **POS (Gujarati)**: [UD Gujarati GTB](https://universaldependencies.org/treebanks/gu_gtb/) — Universal Dependencies
