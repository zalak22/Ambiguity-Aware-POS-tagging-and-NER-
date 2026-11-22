"""Post-hoc constraints for the POS tagger (no retraining).

Two general rules, applied as CONSTRAINTS on the CRF decode (the CRF still picks the best
full tag sequence, it just cannot use forbidden tags):

1. Closed-class lexicon: frequent Gujarati function words whose tag is fixed by grammar
   (auxiliaries, pronouns, postpositions, conjunctions, ...). The Hindi-trained model never
   saw most of them (Gujarati છે vs Hindi है -> હૈ), so it guesses badly on them.
   Written from Gujarati grammar - NOT derived from the GujTB test set.

2. NER consistency (joint-task agreement): the NER model is trained on real Gujarati.
     NER says entity (B-/I-PER/ORG/LOC)  ->  POS must be PROPN
     NER says O                          ->  POS may not be PROPN
   This targets the main transfer error: ordinary Gujarati words labelled PROPN.

Lexicon wins over the NER rule. Measure the effect with src/eval_pos_rules.py.
"""
import re

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
        "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

_LEX = {
    "AUX": "છે છું છો છીએ હતો હતી હતું હતા હતાં નથી હોય હશે",
    "PRON": "હું મને મારું મારી મારો મારા અમે અમને અમારું અમારી અમારો અમારા આપણે આપણને આપણું "
            "આપણી આપણો આપણા તું તને તારું તારી તારો તારા તમે તમને તમારું તમારી તમારો તમારા "
            "તે તેને તેનું તેની તેનો તેના તેણે તેઓ તેમને તેમનું તેમની તેમનો તેમના તેમણે "
            "કોઈ કંઈ શું કોણ જે જેને જેનું જેની જેનો જેના પોતે પોતાનું પોતાની પોતાનો પોતાના",
    "DET": "આ દરેક",
    "ADP": "માટે પર ઉપર સાથે વિશે દ્વારા તરફ પાસે વગર સુધી બાદ વચ્ચે અંગે",
    "CCONJ": "અને અથવા તથા તેમજ પરંતુ પણ",
    "SCONJ": "કે જો જ્યારે",
    "PART": "જ નહીં ન",
    "ADV": "આજે કાલે હવે ત્યાં અહીં ક્યાં ક્યારે ત્યારે સવારે સાંજે રાત્રે હંમેશા ફરી ખૂબ બહુ વધુ પહેલાં પછી",
    "NUM": "એક બે ત્રણ ચાર પાંચ સાત આઠ નવ દસ સો હજાર લાખ કરોડ",
}
LEXICON = {w: tag for tag, words in _LEX.items() for w in words.split()}
_NUM_RE = re.compile(r"[0-9]+([.,][0-9]+)*")
_ALL_BUT_PROPN = frozenset(UPOS) - {"PROPN"}


def pos_constraints(toks, ner_tags=None, use_lexicon=True, use_ner=True):
    """Allowed POS tags per token: a set, or None for 'no constraint'."""
    out = []
    for j, t in enumerate(toks):
        if use_lexicon and t in LEXICON:
            out.append({LEXICON[t]})
        elif use_lexicon and _NUM_RE.fullmatch(t):
            out.append({"NUM"})
        elif use_ner and ner_tags is not None:
            out.append({"PROPN"} if ner_tags[j] != "O" else set(_ALL_BUT_PROPN))
        else:
            out.append(None)
    return out
