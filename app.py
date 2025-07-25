import os
import pandas as pd
import numpy as np
import nltk
import spacy
import torch
import matplotlib.pyplot as plt
import gc
import logging      # termnal error logs
import streamlit as st # Import streamlit

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams, trigrams, FreqDist
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from transformers import BartTokenizer, BartForConditionalGeneration

from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    _ = sent_tokenize("test")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    _ = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    logging.info("Downloading en_core_web_sm SpaCy model...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device} for BART model.")

model_name = "facebook/bart-large-cnn" # ssleifer/distilbart-cnn-12-6 for moar speed.

# Fetch HF_TOKEN from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None

# Load tokenizer and model once globally
try:
    tokenizer = BartTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = BartForConditionalGeneration.from_pretrained(model_name, use_auth_token=HF_TOKEN).to(device)
    logging.info(f"Successfully load BART model: {model_name}")
except Exception as e:
    logging.error(f"Error loading BART model or tokenizer: {e}")
    raise

max_tokens = 1024
chunk_max_len = 150

def load_file(file_object):
    file_name = file_object.name
    ext = os.path.splitext(file_name)[-1].lower()

    if ext == '.csv':
        return pd.read_csv(file_object)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_object)
    else:
        raise ValueError("Unsupported file format.")

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        token_words = word_tokenize(text.lower())
        token_words = [t for t in token_words if t.isalpha() and t not in stop_words]
        
        if not token_words: # text becomes empty after filtering
            return ""

        doc = nlp(' '.join(token_words)) # spaCy processing
        lemmatized = [token.lemma_ for token in doc]
        
        # Ensure there are enough tokens for bigrams/trigrams
        bigram_list = ['_'.join(b) for b in bigrams(lemmatized)] if len(lemmatized) >= 2 else []
        trigram_list = ['_'.join(t) for t in trigrams(lemmatized)] if len(lemmatized) >= 3 else []
        
        all_tokens = lemmatized + bigram_list + trigram_list
        return ' '.join(all_tokens)
    except Exception as e:
        logging.error(f"Error in preprocess_text for: '{text[:100]}' - {e}")
        return ""

def summary_preprocess(text, num_sentences=5):
    if not isinstance(text, str) or not text.strip():
        return ""

    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(stop_words='english', norm='l2')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).ravel()
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_sentence_indices.sort() # sort to maintain original sentence order
        return " ".join([sentences[i] for i in top_sentence_indices])
    except ValueError as ve:
        logging.warning(f"ValueError in summary_preprocess: {ve}. Returning first {num_sentences} sentences.")  # insufficient data
        return " ".join(sentences[:num_sentences])
    except Exception as e:
        logging.error(f"Error in summary_preprocess: '{text[:100]}' - {e}")
        return " ".join(sentences[:num_sentences]) # Fallback


def generate_topic_modeling(texts, col_name, executor, n_topics=5):
    # parallelize preprocessing
    valid_texts = [str(entry) for entry in texts if pd.notnull(entry) and str(entry).strip()]
    
    processed_docs = list(executor.map(preprocess_text, valid_texts))

    processed_docs = [doc for doc in processed_docs if doc.strip()]

    if not processed_docs:
        logging.info(f"No valid processed documents: {col_name}")
        return [], None

    try:
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(processed_docs)

        if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
            logging.warning(f"TF-IDF matrix is empty for: {col_name}")
            return [], None

        # insufficient topics
        effective_n_topics = min(n_topics, tfidf_matrix.shape[0], tfidf_matrix.shape[1] if tfidf_matrix.shape[1] > 0 else 1)
        if effective_n_topics < 1:
            logging.warning(f"Number of topics is less than 1 for column: {col_name}")
            return [], None

        lda = LatentDirichletAllocation(n_components=effective_n_topics, random_state=42)
        lda.fit(tfidf_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_indices = topic.argsort()[-min(5, len(feature_names)):][::-1]
            top_features = [feature_names[i] for i in top_features_indices]
            topics.append((f"Topic {topic_idx + 1}", top_features))

        flat_tokens = ' '.join(processed_docs).split()
        if not flat_tokens:
            wordcloud = None
        else:
            freq = FreqDist(flat_tokens)
            top_words = [word for word, _ in freq.most_common(30)]
            joined_words = ' '.join(top_words)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(joined_words)

        return topics, wordcloud
    except Exception as e:
        logging.error(f"Error generating topic modeling for {col_name}: {e}")
        return [], None


def summarize_chunk(text, max_summary_length=chunk_max_len):
    if not isinstance(text, str) or not text.strip():
        return "Not enough content to generate summary."
    try:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens
        ).to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_summary_length,
            min_length=30,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except RuntimeError as re: # Catch CUDA out of memory or similar
        logging.error(f"RuntimeError during summarization: {re}. Text length: {len(text)}")
        # Attempt to summarize on CPU
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to("cpu")
        model_cpu = model.to("cpu")
        summary_ids = model_cpu.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_summary_length,
            min_length=30,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        model.to(device)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during summarization for: '{text[:100]}' - {e}")
        return "Error generating summary."

def analyze_sentiment(text, analyzer):
    if not isinstance(text, str) or not text.strip():
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    try:
        return analyzer.polarity_scores(text)
    except Exception as e:
        logging.error(f"Error analyzing sentiment for: '{text[:100]}' - {e}")
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def summarize_sentiment_entries(entries, sentiment_scores, target_sentiment, summarization_func):
    filtered_entries = [
        entries[i] for i in range(len(entries))
        if get_sentiment_label(sentiment_scores[i]['compound']) == target_sentiment
    ]
    if not filtered_entries:
        return f"No entries classified as {target_sentiment}."

    combined_text = " ".join(filtered_entries)
    if len(combined_text.split()) > 50:
        return summarization_func(combined_text)
    else:
        return "Not enough content to generate a summary."

def process_columns(df, selected_cols, topic_count=5):
    results = {}
    sid = SentimentIntensityAnalyzer() # VADER

    # Explicitly control max_workers
    max_workers = min(os.cpu_count() or 1, 24)
    logging.info(f"Using {max_workers} worker processes")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for col in selected_cols:
            logging.info(f"Processing column: {col}...")

            entries = df[col].dropna().astype(str).tolist()

            #logging.info(f"Starting summary_preprocess for column {col} with {len(entries)} entries...")
            normalized_entries = list(executor.map(summary_preprocess, entries))
            normalized_entries = [ne for ne in normalized_entries if ne.strip()]
            #logging.info(f"Finished summary_preprocess for column {col}.")

            # pass the executor to the topic modeling
            #logging.info(f"Starting topic modeling for column {col}...")
            topics, wordcloud = generate_topic_modeling(normalized_entries, col, executor, n_topics=topic_count)
            logging.info(f"Finished topic modeling for column {col}.")

            #logging.info(f"Starting sentiment analysis for column {col}...")
            sentiment_scores = [analyze_sentiment(entry, sid) for entry in entries]
            logging.info(f"Finished sentiment analysis for column {col}.")

            combined_text = " ".join(normalized_entries)
            final_summary = "Not enough content to generate a summary."

            if len(combined_text.split()) > 50:
                #logging.info(f"Starting summarization for column: {col}")
                final_summary = summarize_chunk(combined_text, max_summary_length=chunk_max_len)
                logging.info(f"Finished summarization for column: {col}")
            else:
                logging.info(f"Skipping summarization for column {col}: Not enough content.")

            results[col] = {
                "entries": entries, # keep original entries for sentiment summarization
                "sentiment_scores": sentiment_scores,
                "topics": topics,
                "summary": final_summary,
                "wordcloud": wordcloud,
            }
            gc.collect()

    logging.info("NLP complete")
    return results
