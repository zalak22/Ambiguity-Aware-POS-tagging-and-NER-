# UD POS data inspection

## hi2gu_train
- sentences: 13306
- tokens: 281057
- sent_len_mean: 21.1
- sent_len_max: 116
- duplicate_sentences: 11
- multiword_tokens: 0
- vocab_size: 16544
- ambiguous_word_types: 2037
- ambiguous_token_share: 0.531
- UPOS: {'NOUN': 62191, 'ADP': 58463, 'PROPN': 34288, 'VERB': 29008, 'AUX': 18947, 'PUNCT': 18668, 'ADJ': 17279, 'PRON': 11857, 'DET': 6074, 'PART': 5610, 'SCONJ': 5389, 'NUM': 5332, 'CCONJ': 5110, 'ADV': 2711, 'X': 127, 'INTJ': 3}

## hi2gu_dev
- sentences: 1659
- tokens: 35217
- sent_len_mean: 21.2
- sent_len_max: 68
- duplicate_sentences: 3
- multiword_tokens: 0
- vocab_size: 5291
- ambiguous_word_types: 479
- ambiguous_token_share: 0.373
- UPOS: {'NOUN': 7928, 'ADP': 7288, 'PROPN': 4214, 'VERB': 3490, 'AUX': 2417, 'PUNCT': 2367, 'ADJ': 2246, 'PRON': 1473, 'PART': 722, 'NUM': 715, 'DET': 697, 'CCONJ': 682, 'SCONJ': 682, 'ADV': 292, 'X': 4}

## hi2gu_test
- sentences: 1684
- tokens: 35430
- sent_len_mean: 21.0
- sent_len_max: 66
- duplicate_sentences: 1
- multiword_tokens: 0
- vocab_size: 5228
- ambiguous_word_types: 485
- ambiguous_token_share: 0.416
- UPOS: {'NOUN': 8036, 'ADP': 7464, 'PROPN': 4438, 'VERB': 3461, 'PUNCT': 2420, 'AUX': 2392, 'ADJ': 2130, 'PRON': 1372, 'DET': 744, 'NUM': 693, 'PART': 677, 'SCONJ': 655, 'CCONJ': 635, 'ADV': 304, 'X': 9}

## gujtb_test
- sentences: 187
- tokens: 1801
- sent_len_mean: 9.6
- sent_len_max: 37
- duplicate_sentences: 0
- multiword_tokens: 84
- vocab_size: 1013
- ambiguous_word_types: 48
- ambiguous_token_share: 0.135
- UPOS: {'NOUN': 423, 'PUNCT': 250, 'VERB': 214, 'AUX': 184, 'PROPN': 145, 'PRON': 133, 'ADJ': 133, 'ADP': 70, 'ADV': 60, 'CCONJ': 50, 'PART': 43, 'NUM': 40, 'DET': 23, 'INTJ': 14, 'SCONJ': 13, 'X': 3, 'SYM': 3}

**hi_split_overlap**: {'train&dev': 2, 'train&test': 10, 'dev&test': 0}

**gujtb_token_coverage_by_hi2gu_train_vocab**: 0.389

**unmapped_devanagari_chars**: {}

**upos_tagset_hi_minus_gu**: []

**upos_tagset_gu_minus_hi**: ['SYM']

## Samples
### hi2gu_train
- યહ/DET એશિયા/PROPN કી/ADP સબસે/ADV બડી/ADJ મસ્જિદોં/NOUN મેં/ADP સે/ADP એક/NUM હૈ/AUX ./PUNCT
- ઇસે/PRON નવાબ/NOUN શાહજેહન/PROPN ને/ADP બનવાયા/VERB થા/AUX ./PUNCT
- ઇસકા/PRON પ્રવેશ/NOUN દ્વાર/NOUN દો/NUM મંજિલા/ADJ હૈ/AUX ./PUNCT
### gujtb_test
- છોક્રીએ/NOUN તેની/PRON બહેનપણીને/NOUN પત્ર/NOUN લખ્યો/VERB ./PUNCT
- મને/PRON લાગે/VERB છે/AUX કે/SCONJ વરસાદ/NOUN વરસી/VERB રહ્યો/AUX છે/AUX ./PUNCT
- તેણે/PRON ધુમર્પાન/NOUN અને/CCONJ મદિરાપાન/NOUN છોડવાની/VERB કોશિશ/NOUN કરી/VERB ./PUNCT
