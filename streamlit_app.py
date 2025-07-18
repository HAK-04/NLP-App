import streamlit as st
import pandas as pd
import io

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
text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 5]

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
                    st.pyplot(output[col]["wordcloud"].figure)
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
