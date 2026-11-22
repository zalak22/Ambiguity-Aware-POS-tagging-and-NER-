# Ambiguity-Aware Joint Gujarati POS Tagging and NER using BiLSTM

Status: Phase 1-3 done (environment, dataset inspection, NER preprocessing). Phase 4 (NER BiLSTM-CRF baseline) done.

## NER baseline results (entity-level F1, seqeval)
| Run | Train sents | Val F1 | Test F1 | Test-dedup F1 |
|---|---|---|---|---|
| ner_bilstm_crf_train_150k | 150,000 | 0.766 | 0.801 | 0.806 |
| ner_bilstm_crf_train      | 468,881 | 0.780 | 0.824 | 0.830 |
Both hit the 15-epoch cap with val F1 still creeping up; ORG is the weakest class (test F1 0.72).

## POS results (`python src/train_pos.py`, trained on Hindi HDTB transliterated to Gujarati script)
| Split | BiLSTM-CRF acc | Most-frequent-tag acc | BiLSTM OOV acc |
|---|---|---|---|
| hi2gu test (Hindi)     | 0.969 | 0.915 | 0.810 |
| GujTB test (real Gujarati, 1,801 tokens) | **0.372** | **0.432** | 0.203 |
Transfer to real Gujarati fails: 61% of GujTB tokens are OOV and the model labels unknown Gujarati
words as PROPN (NOUN->PROPN 243, VERB->PROPN 87). GujTB accuracy peaked at epoch 1 (0.44) and fell
as the model fitted Hindi. Real Gujarati POS training data (LDC-IL) is needed.

### + Gujarati fastText embeddings (`python src/train_embeddings.py`, then `--embeddings models/embeddings/fasttext_gu_100d.model`)
fastText (100d, skip-gram, char 2-5-grams) on Naamapadam train + hi2gu train; GujTB vocab coverage 39% -> 94%.
| Run | Hindi test acc | GujTB acc | GujTB macro-F1 | GujTB OOV acc | GujTB ambiguous acc |
|---|---|---|---|---|---|
| Most-frequent-tag baseline         | 0.915 | 0.432 | 0.236 | 0.250 | 0.514 |
| BiLSTM-CRF (random init)           | 0.969 | 0.372 | 0.232 | 0.203 | 0.416 |
| + fastText, frozen                 | 0.963 | 0.415 | 0.261 | 0.257 | 0.444 |
| + fastText, fine-tuned             | 0.970 | **0.465** | **0.292** | **0.323** | 0.458 |
Fine-tuned embeddings are the first model to beat the most-frequent-tag baseline on real Gujarati (+3.3 pts).
Main remaining error: Gujarati nouns/verbs labelled PROPN (NOUN->PROPN 154).

### + decoding rules, no retraining (`src/pos_rules.py`, measured by `python src/eval_pos_rules.py`)
Constraints applied to the CRF decode of the fine-tuned model: (1) a closed-class Gujarati function-word
list (~150 words, written from grammar), (2) NER consistency: NER entity -> PROPN, NER O -> not PROPN.
| Setting | GujTB acc | PROPN precision | PROPN recall |
|---|---|---|---|
| raw model            | 0.465 | 0.238 | 0.752 |
| + word list          | 0.614 | 0.278 | 0.766 |
| + NER consistency    | 0.548 | 0.758 | 0.800 |
| + both               | **0.687** | 0.758 | 0.800 |
Caveat: the word list was written by hand; three GujTB sample sentences had been seen in
inspect_ud.py output beforehand, so confirm on a second Gujarati test set (e.g. ILCI) before
reporting the word-list gain as final. The NER-consistency rule uses no hand-picked words.

## Run order (from this folder)
1. Install PyTorch with the CUDA wheel matching your GPU, then `pip install -r requirements.txt`
2. `python env_check.py`            -> results/reports/env_report.json
3. `python inspect_naamapadam.py`   -> results/reports/naamapadam_gu_inspection.{json,md}
4. `python inspect_ud.py`           -> datasets/processed/pos_hi2gu/, pos_gujtb/, results/reports/ud_inspection.{json,md}
5. `python src/preprocessing.py`    -> datasets/processed/ner/, models/vocabulary/, models/label_mappings/,
                                       results/reports/ner_preprocessing_report.json
6. `python src/train_ner.py`        -> NER BiLSTM-CRF (word + char-CNN) on the 150k subset;
                                       add `--train train` for the full set, `--limit 2000 --epochs 1` to smoke-test.
                                       Writes models/checkpoints/<run>/best.pt, results/training_history/<run>.json,
                                       results/metrics/<run>.json (+ test predictions for error analysis)

## Try it on your own text
    python src/predict.py                                   # interactive: type a sentence, Enter; empty line quits
    python src/predict.py "રાહુલ ગાંધી અમદાવાદમાં રહે છે."    # one sentence / paragraph
    python src/predict.py --file my_text.txt                # each line of a UTF-8 file
Prints POS + NER per word and the entities found. Uses the full-data NER model and the
fastText fine-tuned POS model by default (`--ner` / `--pos` to pick other checkpoints).

## Known issue
On this machine (Windows, torch 2.11 + cuDNN 9.19), `nn.LSTM(num_layers>1, dropout>0)` on CUDA
crashes the process at exit (0xC0000409). `src/model.py` therefore stacks single-layer BiLSTMs
with dropout applied between them, which computes the same thing.

## Datasets
- NER: ai4bharat/naamapadam (Gujarati only). Raw copy kept untouched in datasets/raw/naamapadam/.
- POS: LDC-IL Gujarati POS Annotated Corpus (request via data.ldcil.org) -> datasets/raw/ldcil_pos/.
Processed copies go in datasets/processed/; originals are never modified.
