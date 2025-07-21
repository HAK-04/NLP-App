import os
import pandas as pd
import numpy as np
import nltk
import spacy
import torch
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams, trigrams, FreqDist
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from transformers import BartTokenizer, BartForConditionalGeneration

from concurrent.futures import ProcessPoolExecutor

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
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/bart-large-cnn" # ssleifer/distilbart-cnn-12-6 for moar speed.

HF_TOKEN = os.environ.get("HF_TOKEN", None) #IMPORTANT

tokenizer = BartTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = BartForConditionalGeneration.from_pretrained(model_name, use_auth_token=HF_TOKEN).to(device)

max_tokens = 1024
chunk_max_len = 150

def load_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
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

def generate_topic_modeling(texts, col_name, executor, n_topics=5):

    # parallelize preprocessing
    processed_docs = list(executor.map(preprocess_text, [str(entry) for entry in texts if pd.notnull(entry)]))

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

def summarize_chunk(text, max_summary_length=chunk_max_len):
    """Generates a summary for a single block of text (used only once at the end)."""
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

def extractive_summarize_entry(text, num_sentences=5):

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
    except ValueError:
        return " ".join(sentences[:num_sentences])

def analyze_sentiment(text, analyzer):
    """Applies VADER sentiment analysis to a given text."""
    if not isinstance(text, str) or not text.strip():
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    return analyzer.polarity_scores(text)

def process_columns(df, selected_cols, topic_count=5):
    """
    MODIFIED: Uses a ProcessPoolExecutor to parallelize the most intensive,
    row-by-row text processing tasks. Includes sentiment analysis.
    """
    results = {}
    sid = SentimentIntensityAnalyzer() #VADER

    max_workers = os.cpu_count() or 1
    print(f"Using {max_workers} worker processes for parallelization.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for col in selected_cols:
            print(f"Processing column: {col}...")

            entries = df[col].dropna().astype(str).tolist()


            normalized_entries = list(executor.map(extractive_summarize_entry, entries))

            sentiment_scores = [analyze_sentiment(entry, sid) for entry in entries]
            # pass the executor to the topic modeling
            topics, wordcloud = generate_topic_modeling(normalized_entries, col, executor, n_topics=topic_count)

            combined_text = " ".join(normalized_entries)
            final_summary = "Not enough unique content to generate a summary."

            if len(combined_text.split()) > 50:
                print(f"Starting BART summarization for column: {col}")
                final_summary = summarize_chunk(combined_text, max_summary_length=chunk_max_len)

            results[col] = {
                "topics": topics,
                "summary": final_summary,
                "wordcloud": wordcloud,
                "sentiment_scores": sentiment_scores
            }

    print("Processing complete!")
    return results

def main():
    """Main function to handle user input and run the analysis."""

    file_path = input("Enter the path to your CSV or Excel file: ")

    # Load the file
    try:
        df = load_file(file_path)
        print("File loaded successfully.")
        print("Available columns with index:")
        for i, col in enumerate(df.columns.tolist()):
            print(f"{i}: {col}")

        selected_columns_input= input("Enter the column number(s) to process (comma-separated): ")
        selected_indices_str = [idx.strip() for idx in selected_columns_input.split(',')]
        selected_indices = []
        valid_columns = []
        invalid_inputs = []

        for idx_str in selected_indices_str:
            try:
                idx = int(idx_str)
                if 0 <= idx < len(df.columns):
                    selected_indices.append(idx)
                    valid_columns.append(df.columns[idx])
                else:
                    invalid_inputs.append(idx_str)
            except ValueError:
                invalid_inputs.append(idx_str)


        if invalid_inputs:
            print(f"Warning: The following inputs are not valid column numbers and will be ignored: {', '.join(invalid_inputs)}")

        if valid_columns:
            analysis_results = process_columns(df, valid_columns)

            for col, result in analysis_results.items():
                print(f"\n--- Results for {col} ---")
                print("Summary:")
                print(result['summary'])
                print("\nTopics:")
                for topic_name, features in result['topics']:
                    print(f"{topic_name}: {', '.join(features)}")

                # sentiment results
                if result['sentiment_scores']:
                    neg_scores = [s['neg'] for s in result['sentiment_scores']]
                    neu_scores = [s['neu'] for s in result['sentiment_scores']]
                    pos_scores = [s['pos'] for s in result['sentiment_scores']]
                    compound_scores = [s['compound'] for s in result['sentiment_scores']]

                    avg_neg = np.mean(neg_scores)
                    avg_neu = np.mean(neu_scores)
                    avg_pos = np.mean(pos_scores)
                    avg_compound = np.mean(compound_scores)

                    print("\nSentiment Analysis:")
                    print(f"  Average Negative: {avg_neg:.4f}")
                    print(f"  Average Neutral: {avg_neu:.4f}")
                    print(f"  Average Positive: {avg_pos:.4f}")
                    print(f"  Average Compound: {avg_compound:.4f}")

                    if avg_compound >= 0.05:
                        overall_sentiment = "Positive"
                    elif avg_compound <= -0.05:
                        overall_sentiment = "Negative"
                    else:
                        overall_sentiment = "Neutral"
                    print(f"  Overall Sentiment: {overall_sentiment}")

                    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
                    fig.suptitle(f'Analysis for Column: {col}', fontsize=16)

                    sentiment_labels = ['Negative', 'Neutral', 'Positive']
                    average_sentiments = [avg_neg, avg_neu, avg_pos]

                    axes[0].bar(sentiment_labels, average_sentiments, color=['red', 'gray', 'green'])
                    axes[0].set_title('Sentiment Distribution')
                    axes[0].set_ylabel('Average Score')
                    axes[0].set_ylim(0, 1)

                    if result['wordcloud']:
                        axes[1].imshow(result['wordcloud'], interpolation='bilinear')
                        axes[1].axis("off")
                        axes[1].set_title('Word Cloud')

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.show()

        else:
            print("No valid columns selected for processing.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except ValueError as e:
        print(f"Error loading file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
