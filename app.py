import streamlit as st
HF_TOKEN = st.secrets["HF_TOKEN"]

import os
import pandas as pd
import nltk
import spacy
import torch

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams, trigrams, FreqDist
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from transformers import BartTokenizer, BartForConditionalGeneration

# nltk_data path
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path = [nltk_data_path] + nltk.data.path

# nltk take 2
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    stop_words = set(stopwords.words('english'))

try:
    _ = sent_tokenize("test")
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# BART summarization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = BartForConditionalGeneration.from_pretrained(model_name, use_auth_token=HF_TOKEN).to(device)

# constants
max_tokens = 1024
chunk_max_len = 150
entry_summary_max_len = 60

def load_file(file_obj):
    ext = os.path.splitext(file_obj.name)[-1].lower()
    if ext == '.csv':
        return pd.read_csv(file_obj)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_obj)
    else:
        raise ValueError("Unsupported file format.")

def preprocess_text(text):
    token_words = word_tokenize(text.lower())
    token_words = [t for t in token_words if t.isalpha() and t not in stop_words]
    doc = nlp(' '.join(token_words))
    lemmatized = [token.lemma_ for token in doc]

    bigram_list = ['_'.join(b) for b in bigrams(lemmatized)]
    trigram_list = ['_'.join(t) for t in trigrams(lemmatized)]

    all_tokens = lemmatized + bigram_list + trigram_list
    return ' '.join(all_tokens)

def generate_topic_modeling(texts, col_name, n_topics=5):
    processed_docs = [preprocess_text(str(entry)) for entry in texts if pd.notnull(entry)]

    if not processed_docs:
        return [], None

    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(processed_docs)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
        topics.append((f"Topic {topic_idx + 1}", top_features))

    flat_tokens = ' '.join(processed_docs).split()
    freq = FreqDist(flat_tokens)
    top_words = [word for word, _ in freq.most_common(30)]
    joined_words = ' '.join(top_words)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(joined_words)

    return topics, wordcloud

def chunk_text(sentences, max_tokens=max_tokens):
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_len + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(tokens)
        else:
            current_chunk.append(sentence)
            current_len += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_chunk(text, max_summary_length=chunk_max_len):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_tokens
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_summary_length,
        min_length=20,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_entry(entry_text):
    if not entry_text.strip():
        return ""
    sentences = sent_tokenize(entry_text)
    chunks = chunk_text(sentences)
    chunk_summaries = [summarize_chunk(chunk, max_summary_length=entry_summary_max_len) for chunk in chunks]
    return " ".join(chunk_summaries)

def summarize_column_with_equal_weight(entries):
    entry_summaries = [
        summarize_entry(str(entry).strip())
        for entry in entries
        if pd.notnull(entry) and str(entry).strip()
    ]
    combined_summary_input = " ".join(entry_summaries)
    final_summary = summarize_chunk(combined_summary_input, max_summary_length=chunk_max_len)
    return final_summary

def process_columns(df, selected_cols, topic_count=5):
    results = {}
    for col in selected_cols:
        entries = df[col].dropna().astype(str).tolist()
        topics, wordcloud = generate_topic_modeling(entries, col, n_topics=topic_count)
        summary = summarize_column_with_equal_weight(entries)
        results[col] = {
            "topics": topics,
            "summary": summary,
            "wordcloud": wordcloud
        }
    return results
