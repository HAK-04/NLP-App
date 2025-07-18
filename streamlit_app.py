import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

import spacy
nlp = spacy.load("en_core_web_sm")

from app import load_file, process_columns

# streamlit config
st.set_page_config(page_title="Topic Modeling & Summarization", layout="wide")

st.title("NLP Column Analyzer")
st.markdown("Upload a text-rich dataset and select columns for **topic modeling and summarization**.")

# file upload
st.markdown("#### Upload Dataset")
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=["csv", "xls", "xlsx"],
    help="Accepted formats: .csv, .xls, .xlsx"
)

file_error = None
df = None

if uploaded_file:
    try:
        df = load_file(uploaded_file)
    except Exception as e:
        file_error = f"❌ Error reading file: {e}"
else:
    file_error = "⚠️ Please upload a file to continue."

if file_error and not df:
    st.markdown(f"<span style='color: red;'>{file_error}</span>", unsafe_allow_html=True)
    st.stop()

# select columns
st.markdown("#### Select Text Columns for NLP Processing")

# detect columns
text_columns = [col for col in df.select_dtypes(include='object').columns if is_textual_column(df[col])]

if not text_columns:
    st.warning("No suitable text columns found in this file.")
    st.stop()

select_all = st.checkbox("Select All Text Columns")
selected_columns = st.multiselect(
    "Choose one or more columns:",
    options=text_columns,
    default=text_columns if select_all else []
)

if not selected_columns:
    st.error("⚠️ Please select at least one text column.")
    st.stop()

# topic slider
st.markdown("#### Choose Number of Topics to Extract (per column)")
num_topics = st.slider("Topics per column", min_value=1, max_value=5, value=3)

# topic count to processing function
def custom_process_columns(df, selected_cols, topic_count):
    from app import generate_topic_modeling, summarize_column_with_equal_weight
    results = {}
    for col in selected_cols:
        entries = df[col].dropna().astype(str).tolist()
        topics, wordcloud = generate_topic_modeling(entries, col)
        summary = summarize_column_with_equal_weight(entries)
        # Slice topics as per slider
        topics = topics[:topic_count]
        results[col] = {
            "topics": topics,
            "summary": summary,
            "wordcloud": wordcloud
        }
    return results

st.markdown("----")

# run app.py
if st.button("Run NLP Processing"):
    with st.spinner("Processing... This may take a few minutes."):
        output = custom_process_columns(df, selected_columns, num_topics)

    if len(selected_columns) <= 10:
        st.success("✅ Processing complete! See results below.")

        for col in selected_columns:
            with st.expander(f"📌 Column: {col}"):
                st.markdown(f"**Summary:**\n\n{output[col]['summary']}")
                
                st.markdown("**Topics Identified:**")
                for topic_name, keywords in output[col]["topics"]:
                    st.markdown(f"- **{topic_name}**: {', '.join(keywords)}")

                st.markdown("**WordCloud:**")
                if output[col]["wordcloud"]:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(output[col]["wordcloud"], interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)

    else:
        # display limit
        st.info("More than 10 columns selected. Results are provided as a downloadable file.")
        results_text = io.StringIO()
        for col in selected_columns:
            results_text.write(f"=== {col} ===\n")
            results_text.write(f"\nSummary:\n{output[col]['summary']}\n")
            results_text.write("Topics:\n")
            for topic_name, keywords in output[col]["topics"]:
                results_text.write(f"- {topic_name}: {', '.join(keywords)}\n")
            results_text.write("\n\n")

        st.download_button(
            label="📥 Download Results",
            data=results_text.getvalue(),
            file_name="nlp_analysis_results.txt",
            mime="text/plain"
        )


def is_textual_column(series, min_word_count=5, min_alpha_ratio=0.6, max_propn_ratio=0.5, sample_size=30):
    """Determine if a column contains natural language text based on heuristics."""

    samples = series.dropna().astype(str)
    if samples.empty:
        return False

    samples = samples.sample(n=min(sample_size, len(samples)), random_state=42)

    total_words = 0
    total_alpha_ratios = []
    proper_noun_ratios = []

    for text in samples:
        words = text.split()
        total_words += len(words)

        # alpha to alphanumeric ratio
        alpha_chars = sum(c.isalpha() for c in text)
        alphanum_chars = sum(c.isalnum() for c in text)
        if alphanum_chars > 0:
            total_alpha_ratios.append(alpha_chars / alphanum_chars)

        # proper noun detection with spaCy
        doc = nlp(text)
        if len(doc) > 0:
            propn_count = sum(1 for token in doc if token.pos_ == "PROPN")
            proper_noun_ratios.append(propn_count / len(doc))

    avg_word_count = total_words / len(samples)
    avg_alpha_ratio = sum(total_alpha_ratios) / len(total_alpha_ratios) if total_alpha_ratios else 0
    avg_propn_ratio = sum(proper_noun_ratios) / len(proper_noun_ratios) if proper_noun_ratios else 0

    return (
        avg_word_count >= min_word_count and
        avg_alpha_ratio >= min_alpha_ratio and
        avg_propn_ratio <= max_propn_ratio
    )
