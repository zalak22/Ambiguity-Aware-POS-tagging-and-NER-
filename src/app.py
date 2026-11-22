from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import torch

# Project paths (this file lives in src/, so the project root is one level up)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import Tagger, entities
from preprocessing import split_sentences, tokenize
from pos_rules import pos_constraints

"""
cd D:\Learning\NLP\Gujarati-NLP-Research_Project
.\.venv\Scripts\streamlit.exe run src\app.py
"""
NER_CHECKPOINT = ROOT / "models/checkpoints/ner_bilstm_crf_train/best.pt"
POS_CHECKPOINT = ROOT / "models/checkpoints/pos_bilstm_crf_hi2gu_fttuned/best.pt"


st.set_page_config(
    page_title="Gujarati NLP Analyzer",
    page_icon="🔤",
    layout="wide"
)


# Load the trained models only once
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ner_model = Tagger(NER_CHECKPOINT, device)
    pos_model = Tagger(POS_CHECKPOINT, device)

    return ner_model, pos_model, device


def show_table(df):
    """Full-width table that works on both old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)          # Streamlit 1.45+
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)  # older Streamlit


def analyze_text(text):
    ner_model, pos_model, device = load_models()

    all_rows = []
    all_entities = []

    sentences = split_sentences(text)

    for sentence in sentences:
        tokens = tokenize(sentence)

        if not tokens:
            continue

        ner_tags = ner_model(tokens)

        constraints = pos_constraints(tokens, ner_tags)
        pos_tags = pos_model(tokens, constraints)

        extracted = entities(tokens, ner_tags)

        for word, pos, ner in zip(tokens, pos_tags, ner_tags):
            all_rows.append({
                "Word": word,
                "POS Tag": pos,
                "NER Tag": ner
            })

        for entity_type, entity_text in extracted:
            all_entities.append({
                "Entity": entity_text,
                "Entity Type": entity_type
            })

    return all_rows, all_entities


# Header
st.title("🔤 Gujarati NLP Analyzer")

st.markdown(
    "### Ambiguity-Aware Gujarati POS Tagging and Named Entity Recognition"
)

st.write(
    "Analyze Gujarati text using trained BiLSTM-CRF models "
    "for Part-of-Speech tagging and Named Entity Recognition."
)

st.divider()

# Input section
st.subheader("Enter Gujarati Text")

sample_text = "રાહુલ ગાંધી અમદાવાદમાં રહે છે."

user_text = st.text_area(
    "Enter a Gujarati sentence or paragraph:",
    value=sample_text,
    height=130,
    placeholder="અહીં ગુજરાતી લખાણ દાખલ કરો..."
)

if st.button("🔍 Analyze Text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some Gujarati text.")
    else:
        try:
            with st.spinner("Loading models and analyzing text..."):
                rows, extracted_entities = analyze_text(user_text)

            if not rows:
                st.warning("No words could be extracted. Please try another sentence.")
            else:
                st.success("Analysis completed successfully!")

                # POS and NER results
                st.subheader("📊 Word-Level Analysis")

                df = pd.DataFrame(rows)
                show_table(df)

                # Entity extraction
                st.subheader("🏷️ Named Entities")

                if extracted_entities:
                    entity_df = pd.DataFrame(extracted_entities)
                    show_table(entity_df)
                else:
                    st.info("No named entities detected.")

                # Summary
                st.subheader("📈 Analysis Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Words", len(rows))

                with col2:
                    st.metric("Named Entities", len(extracted_entities))

                with col3:
                    st.metric(
                        "Unique POS Tags",
                        df["POS Tag"].nunique()
                    )

        except Exception as error:
            st.error(f"An error occurred: {error}")
            st.info(
                "Please check that both model checkpoints and project "
                "dependencies are available."
            )


st.divider()

# Project information
with st.expander("ℹ️ About the Project"):
    st.write("""
    **Project:** Ambiguity-Aware Joint Gujarati POS Tagging and NER using BiLSTM

    **Components:**
    - BiLSTM-CRF based POS tagging
    - BiLSTM-CRF based Named Entity Recognition
    - POS–NER consistency constraints during decoding
    - BIO-based named entity extraction

    **Entity Categories:**
    - PER – Person
    - LOC – Location
    - ORG – Organization

    **Processing Pipeline:**
    1. Gujarati text input
    2. Sentence splitting and tokenization
    3. POS and NER model prediction
    4. POS–NER consistency constraints
    5. Named entity extraction
    """)