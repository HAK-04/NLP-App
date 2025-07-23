import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import zipfile
import os
    
import warnings
warnings.filterwarnings('ignore')

from app import load_file, process_columns, analyze_sentiment, get_sentiment_label, summarize_sentiment_entries, summarize_chunk

import multiprocessing
try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# st config
st.set_page_config(page_title="Topic Modeling & Summarization", layout="wide")

st.title("NLP Column Analyzer")
st.markdown("Upload a text-rich dataset and select columns for **topic modeling, summarization, and sentiment analysis**.")

# File Upload
st.markdown("#### Upload Dataset")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xls", "xlsx"],
    help="Accepted formats: .csv, .xls, .xlsx"
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_error' not in st.session_state:
    st.session_state.file_error = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'output' not in st.session_state:
    st.session_state.output = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_columns_for_display' not in st.session_state:
    st.session_state.selected_columns_for_display = []
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'num_topics' not in st.session_state:
    st.session_state.num_topics = 3
if 'nlp_started' not in st.session_state:
    st.session_state.nlp_started = False


if uploaded_file is not st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.df = None
    st.session_state.file_error = None
    st.session_state.output = None
    st.session_state.processing_complete = False
    st.session_state.selected_columns_for_display = []
    st.session_state.selected_columns = []
    st.session_state.nlp_started = False

    if uploaded_file:
        try:
            st.session_state.df = load_file(uploaded_file)
        except Exception as e:
            st.session_state.file_error = f"❌ Error reading file: {e}"
            st.session_state.df = None
    else:
        st.session_state.file_error = "⚠️ Please upload a file to continue."


# columns selection
st.markdown("#### Select Text Columns for NLP")

available_text_like_columns = []
if st.session_state.df is not None:
    available_text_like_columns = [col for col in st.session_state.df.columns if st.session_state.df[col].dtype == 'object']

if st.session_state.df is not None:
    if not available_text_like_columns:
        st.warning("No object-type columns found in this file that could contain text data.")
        st.session_state.selected_columns = []
    else:
        current_selection_is_all = set(st.session_state.selected_columns) == set(available_text_like_columns) and len(available_text_like_columns) > 0
        select_all = st.checkbox("Select All Object Columns", value=current_selection_is_all)

        if select_all:
            default_multiselect_value = available_text_like_columns
        elif 'selected_columns' in st.session_state:
            default_multiselect_value = st.session_state.selected_columns
        else:
            default_multiselect_value = []

        selected_columns = st.multiselect(
            "Choose one or more columns:",
            options=available_text_like_columns,
            default=default_multiselect_value
        )
        st.session_state.selected_columns = selected_columns
else:
    st.session_state.selected_columns = []
    st.info("Upload a file above to select columns.")

if not st.session_state.nlp_started:
    st.warning("⚠️ Only select columns viable for NLP.")


# Topic Slider
st.markdown("#### Choose Number of Topics to Extract (per column)")
num_topics = st.slider("Topics per column", min_value=1, max_value=5, value=st.session_state.num_topics)
st.session_state.num_topics = num_topics

st.markdown("----")

# Run NLP Button
if st.button("Run NLP"):
    st.session_state.nlp_started = True
    if st.session_state.df is None:
        st.error("❌ Please upload a file before running NLP.")
        st.session_state.processing_complete = False
        st.stop()

    viable_selected_columns = []
    for col in st.session_state.selected_columns:
        if st.session_state.df[col].dtype == 'object' and st.session_state.df[col].nunique() > 5:
            viable_selected_columns.append(col)
        else:
            st.warning(f"Column '{col}' was selected but is not viable for NLP (not object type or too few unique values). It will be skipped.")

    if not viable_selected_columns:
        st.error("❌ No viable text columns were selected or found in the uploaded file after filtering.")
        st.session_state.processing_complete = False
        st.stop()

    with st.spinner("Processing NLP and preparing results... This may take a few minutes."):
        output = process_columns(st.session_state.df, viable_selected_columns, topic_count=st.session_state.num_topics)

        st.session_state.output = output
        st.session_state.processing_complete = True
        st.session_state.selected_columns_for_display = viable_selected_columns

        # results preparation and display in spinner's context 
        output = st.session_state.output
        selected_columns = st.session_state.selected_columns_for_display

        if not selected_columns: # If no columns were successfully processed
            st.info("No viable columns were processed for NLP. Please check your column selections and file content.")
        else:
            results_text_buffer = io.StringIO()
            image_buffers = {}

            for col in selected_columns: # iterate processed col
                col_safe_name = "".join([c for c in col if c.isalnum() or c in (' ', '_')]).replace(' ', '_')[:50]

                results_text_buffer.write(f"=== Column: {col} ===\n\n")

                summary_text = output[col].get('summary', 'Summary not available.')
                results_text_buffer.write(f"Summary:\n{summary_text}\n\n")

                topics_data = output[col].get('topics', [])
                results_text_buffer.write("Topics Identified:\n")
                if topics_data:
                    for topic_name, keywords in topics_data:
                        results_text_buffer.write(f"- {topic_name}: {', '.join(keywords)}\n")
                else:
                    results_text_buffer.write("No topics identified or an error occurred during topic modeling.\n")
                results_text_buffer.write("\n")

                wordcloud_data = output[col].get('wordcloud')
                if wordcloud_data:
                    wordcloud_filename = f"WordCloud_{col_safe_name}.png"
                    wordcloud_buffer = io.BytesIO()
                    wordcloud_data.to_image().save(wordcloud_buffer, format='png')
                    image_buffers[wordcloud_filename] = wordcloud_buffer
                    results_text_buffer.write(f"WordCloud: See {wordcloud_filename}\n\n")
                else:
                    results_text_buffer.write("WordCloud: Not generated.\n\n")

                sentiment_scores = output[col].get('sentiment_scores')
                if sentiment_scores:
                    neg_scores = [s['neg'] for s in sentiment_scores]
                    neu_scores = [s['neu'] for s in sentiment_scores]
                    pos_scores = [s['pos'] for s in sentiment_scores]
                    compound_scores = [s['compound'] for s in sentiment_scores]

                    avg_neg = np.mean(neg_scores)
                    avg_neu = np.mean(neu_scores)
                    avg_pos = np.mean(pos_scores)
                    avg_compound = np.mean(compound_scores)

                    results_text_buffer.write("Sentiment Analysis:\n")
                    results_text_buffer.write(f"    Average Negative: {avg_neg:.4f}\n")
                    results_text_buffer.write(f"    Average Neutral: {avg_neu:.4f}\n")
                    results_text_buffer.write(f"    Average Positive: {avg_pos:.4f}\n")
                    results_text_buffer.write(f"    Average Compound: {avg_compound:.4f}\n")
                    results_text_buffer.write(f"    Overall Sentiment: {get_sentiment_label(avg_compound)}\n\n")

                    sentiment_chart_filename = f"SentimentChart_{col_safe_name}.png"
                    sentiment_buffer = io.BytesIO()
                    sentiment_labels = ['Negative', 'Neutral', 'Positive']
                    average_sentiments = [avg_neg, avg_neu, avg_pos]
                    fig_sentiment, ax_sentiment = plt.subplots(figsize=(5, 3))
                    ax_sentiment.bar(sentiment_labels, average_sentiments, color=['red', 'gray', 'green'])
                    ax_sentiment.set_title(f'Sentiment Distribution for {col}')
                    ax_sentiment.set_ylabel('Average Score')
                    ax_sentiment.set_ylim(0, 1)
                    fig_sentiment.tight_layout()
                    fig_sentiment.savefig(sentiment_buffer, format='png')
                    plt.close(fig_sentiment)
                    image_buffers[sentiment_chart_filename] = sentiment_buffer
                    results_text_buffer.write(f"Sentiment Distribution Chart: See {sentiment_chart_filename}\n\n")

                    results_text_buffer.write("Sentiment Summaries:\n")
                    original_entries = output[col].get('entries', [])
                    positive_summary = summarize_sentiment_entries(original_entries, sentiment_scores, "Positive", summarize_chunk)
                    results_text_buffer.write(f"    Positive Summary: {positive_summary}\n")
                    negative_summary = summarize_sentiment_entries(original_entries, sentiment_scores, "Negative", summarize_chunk)
                    results_text_buffer.write(f"    Negative Summary: {negative_summary}\n")
                    neutral_summary = summarize_sentiment_entries(original_entries, sentiment_scores, "Neutral", summarize_chunk)
                    results_text_buffer.write(f"    Neutral Summary: {neutral_summary}\n")
                else:
                    results_text_buffer.write("Sentiment Analysis: Not available.\n")
                    results_text_buffer.write("Sentiment Summaries: Not available.\n")
                results_text_buffer.write("\n\n")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('nlp_analysis_report.txt', results_text_buffer.getvalue())
                for filename, img_buffer in image_buffers.items():
                    img_buffer.seek(0)
                    zf.writestr(f'visualizations/{filename}', img_buffer.getvalue())

            zip_buffer.seek(0)

            st.success("✅ Processing complete! See results below.") # Move success message here

            st.download_button(
                label="📥 Download All Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="nlp_analysis_results.zip",
                mime="application/zip"
            )

            if len(selected_columns) <= 10:
                st.markdown("### Interactive Results Display")
                for col in selected_columns:
                    with st.expander(f"📌 Column: {col}"):
                        summary_text = output[col].get('summary', 'Summary not available.')
                        st.markdown(f"**Summary:**\n\n{summary_text}")

                        topics_data = output[col].get('topics', [])
                        st.markdown("**Topics Identified:**")
                        if topics_data:
                            for topic_name, keywords in topics_data:
                                st.markdown(f"- **{topic_name}**: {', '.join(keywords)}")
                        else:
                            st.info("No topics identified or an error occurred during topic modeling.")

                        wordcloud_data = output[col].get('wordcloud')
                        st.markdown("**WordCloud:**")
                        if wordcloud_data:
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.imshow(wordcloud_data, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("No WordCloud generated.")

                        st.markdown("### Sentiment Analysis")
                        sentiment_scores = output[col].get('sentiment_scores')
                        if sentiment_scores:
                            neg_scores = [s['neg'] for s in sentiment_scores]
                            neu_scores = [s['neu'] for s in sentiment_scores]
                            pos_scores = [s['pos'] for s in sentiment_scores]
                            compound_scores = [s['compound'] for s in sentiment_scores]

                            avg_neg = np.mean(neg_scores)
                            avg_neu = np.mean(neu_scores)
                            avg_pos = np.mean(pos_scores)
                            avg_compound = np.mean(compound_scores)

                            st.markdown(f"**Average Negative:** {avg_neg:.4f}")
                            st.markdown(f"**Average Neutral:** {avg_neu:.4f}")
                            st.markdown(f"**Average Positive:** {avg_pos:.4f}")
                            st.markdown(f"**Average Compound:** {avg_compound:.4f}")
                            st.markdown(f"**Overall Sentiment:** {get_sentiment_label(avg_compound)}")

                            st.markdown("**Sentiment Distribution:**")
                            sentiment_labels = ['Negative', 'Neutral', 'Positive']
                            average_sentiments = [avg_neg, avg_neu, avg_pos]

                            fig_sentiment, ax_sentiment = plt.subplots(figsize=(5, 3))
                            ax_sentiment.bar(sentiment_labels, average_sentiments, color=['red', 'gray', 'green'])
                            ax_sentiment.set_title(f'Sentiment Distribution for {col}')
                            ax_sentiment.set_ylabel('Average Score')
                            ax_sentiment.set_ylim(0, 1)
                            fig_sentiment.tight_layout()
                            st.pyplot(fig_sentiment)
                            plt.close(fig_sentiment)

                        st.markdown("### Sentiment Summaries")
                        if sentiment_scores:
                            original_entries = output[col].get('entries', [])
                            st.markdown("**Summary of Positive Sentiments:**")
                            positive_summary = summarize_sentiment_entries(original_entries, sentiment_scores, "Positive", summarize_chunk)
                            st.write(positive_summary)
                            st.markdown("**Summary of Negative Sentiments:**")
                            negative_summary = summarize_sentiment_entries(original_entries, sentiment_scores, "Negative", summarize_chunk)
                            st.write(negative_summary)
                            st.markdown("**Summary of Neutral Sentiments:**")
                            neutral_summary = summarize_sentiment_entries(original_entries, sentiment_scores, "Neutral", summarize_chunk)
                            st.write(neutral_summary)
                        else:
                            st.info("No sentiment data available for summarization.")
            else:
                st.info("Interactive display is limited to 10 columns. All results are available for download.")
