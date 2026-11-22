## datasets\raw\naamapadam\gu_extracted\gu_test.json
```
{
  "records": 1076,
  "fields": {
    "words": 1076,
    "ner": 1076
  },
  "token_field": "words",
  "label_field": "ner",
  "empty": 0,
  "token_label_length_mismatch": 0,
  "exact_duplicate_sentences": 47,
  "tokens_total": 13617,
  "sent_len": {
    "mean": 12.7,
    "median": 10.0,
    "max": 100,
    "p95": 30
  },
  "raw_label_values": {
    "O": 10794,
    "B-PER": 673,
    "I-PER": 579,
    "B-ORG": 419,
    "B-LOC": 645,
    "I-ORG": 401,
    "I-LOC": 106
  },
  "tag_distribution": {
    "O": 10794,
    "B-PER": 673,
    "I-PER": 579,
    "B-ORG": 419,
    "B-LOC": 645,
    "I-ORG": 401,
    "I-LOC": 106
  },
  "entities_by_type": {
    "PER": 673,
    "ORG": 419,
    "LOC": 645
  },
  "invalid_BIO_transitions(I-without-B)": 5
}
```
તેજસ્વી/O  રંગીન/O  શાકભાજી/O

સુશાંતસિંહ/B-PER  રાજપૂત/I-PER  રિહા/B-PER  ચક્રવર્તી/I-PER  સાથે/O  ડેટ/O  કરી/O  રહી/O  હોવાની/O  ચર્ચા/O  ચાલી/O  રહી/O  છે/O

મંત્રાલયનો/O  ડેટા/O  માત્ર/O  સરકારી/O  ડોક્ટરો/O  માટે/O  જ/O  છે/O  ./O

## datasets\raw\naamapadam\gu_extracted\gu_test.tsv
```
"unreadable as JSON: Expecting value: line 1 column 1 (char 0)"
```
## datasets\raw\naamapadam\gu_extracted\gu_train.json
```
{
  "records": 472845,
  "fields": {
    "words": 472845,
    "ner": 472845
  },
  "token_field": "words",
  "label_field": "ner",
  "empty": 0,
  "token_label_length_mismatch": 0,
  "exact_duplicate_sentences": 3548,
  "tokens_total": 6218281,
  "sent_len": {
    "mean": 13.2,
    "median": 11,
    "max": 100,
    "p95": 31
  },
  "raw_label_values": {
    "O": 4923455,
    "B-LOC": 238145,
    "I-LOC": 56196,
    "B-PER": 321674,
    "B-ORG": 205652,
    "I-ORG": 190116,
    "I-PER": 283043
  },
  "tag_distribution": {
    "O": 4923455,
    "B-LOC": 238145,
    "I-LOC": 56196,
    "B-PER": 321674,
    "B-ORG": 205652,
    "I-ORG": 190116,
    "I-PER": 283043
  },
  "entities_by_type": {
    "LOC": 238145,
    "PER": 321674,
    "ORG": 205652
  },
  "invalid_BIO_transitions(I-without-B)": 1725
}
```
લક્ઝરી/O  સેન્ટ/B-LOC  એન્ડ્રુ/I-LOC  માતાનો/I-LOC  ચર્ચ/I-LOC

આ/O  દિવસે/O  ભગવાન/O  ગણેશની/B-PER  પૂજાનું/O  વધારે/O  મહત્ત્વ/O  છે/O  ./O

આદુ/O  પાવડર/O  -/O  1/O  ટીસ્પૂન/O

## datasets\raw\naamapadam\gu_extracted\gu_val.json
```
{
  "records": 2389,
  "fields": {
    "words": 2389,
    "ner": 2389
  },
  "token_field": "words",
  "label_field": "ner",
  "empty": 0,
  "token_label_length_mismatch": 0,
  "exact_duplicate_sentences": 0,
  "tokens_total": 31501,
  "sent_len": {
    "mean": 13.2,
    "median": 11,
    "max": 83,
    "p95": 31
  },
  "raw_label_values": {
    "B-ORG": 1056,
    "O": 24957,
    "B-PER": 1612,
    "I-PER": 1405,
    "B-LOC": 1207,
    "I-LOC": 305,
    "I-ORG": 959
  },
  "tag_distribution": {
    "B-ORG": 1056,
    "O": 24957,
    "B-PER": 1612,
    "I-PER": 1405,
    "B-LOC": 1207,
    "I-LOC": 305,
    "I-ORG": 959
  },
  "entities_by_type": {
    "ORG": 1056,
    "PER": 1612,
    "LOC": 1207
  },
  "invalid_BIO_transitions(I-without-B)": 8
}
```
રેડમી/B-ORG  નોટ/O  8/O  પ્રોનાં/O  બેઝિક/O  સ્પેસિફિકેશન/O

કોંગ્રેસ/B-ORG  વાળા/O  આ/O  વિષયને/O  ભટકાવવા/O  માંગે/O  છે/O  ./O

પરીક્ષાની/O  આગામી/O  તારીખ/O  ટૂંક/O  સમયમાં/O  જ/O  રજૂ/O  કરવામાં/O  આવશે/O  ./O

## Cross-file overlap
{'gu_test.json & gu_train.json': 9, 'gu_test.json & gu_val.json': 0, 'gu_train.json & gu_val.json': 33}
