import streamlit as st
import numpy as np
import nltk
import networkx as nx
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_score, recall_score, f1_score
from summarizer import Summarizer

nltk.download('vader_lexicon')
nltk.download('punkt')
st.title("Text Summarizer")

text = st.text_area("Enter your text here:")

# Allow user to set summary ratio
summary_ratio = st.sidebar.slider("Select summary ratio:", min_value=0.1, max_value=1.0, value=0.2, step=0.1)

# Tokenize input text into sentences using NLTK
def read_article(text):
    sentences = nltk.sent_tokenize(text)
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip().lower()
    return sentences

# Compute the similarity between two sentences
def sentence_similarity(sent1, sent2):
    words1 = nltk.word_tokenize(sent1)
    words2 = nltk.word_tokenize(sent2)
    vocab = set(words1 + words2)
    vec1 = np.zeros(len(vocab))
    vec2 = np.zeros(len(vocab))
    for word in words1:
        vec1[list(vocab).index(word)] += 1
    for word in words2:
        vec2[list(vocab).index(word)] += 1
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Build similarity matrix between all sentences
def build_similarity_matrix(sentences):
    if len(sentences) < 2:
        return np.zeros((1, 1))
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    return similarity_matrix

# Generate summary of input text
def generate_summary(text, ratio):
    # Read input text and tokenize sentences
    sentences = read_article(text)
    
    # Generate similarity matrix between sentences
    similarity_matrix = build_similarity_matrix(sentences)
    
    # Apply TextRank algorithm to get most important sentences
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    # Get summary of input text
    summary = ""
    num_sentences = max(1, int(len(sentences) * ratio))
    if len(ranked_sentences) < num_sentences:
        num_sentences = len(ranked_sentences)
    for i in range(num_sentences):
        summary += ranked_sentences[i][1] + " "
    
    return summary

# Function to calculate summary metrics
def calculate_summary_metrics(original_text, summary_text):
    original_words = original_text.split()
    summary_words = summary_text.split()
    
    # Make sure the number of samples is the same
    min_length = min(len(original_words), len(summary_words))
    original_words = original_words[:min_length]
    summary_words = summary_words[:min_length]

    # Calculate precision score for each tuple of original and summary words
    precision_scores = []
    for original_word, summary_word in zip(original_words, summary_words):
        if original_word == summary_word:
            precision_scores.append(1)
        else:
            precision_scores.append(0)

    # Calculate summary metrics
    original_length = len(original_words)
    summary_length = len(summary_words)
    if len(precision_scores) == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = sum(precision_scores) / len(precision_scores)
        recall = sum(precision_scores) / original_length
        f1 = 2 * (precision * recall) / (precision + recall)

    return original_length, summary_length, precision, recall, f1

# Function to perform exploratory data analysis
def perform_eda(text, summary_ratio):
    if not text.strip():
        st.error("Please enter some text to summarize.")
        return

# Create summary
summarizer_model = Summarizer
summary = summarizer_model()(text, ratio=summary_ratio)

# Tokenize text
tokens = word_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.casefold() not in stop_words and word.isalnum()]

# Calculate word frequency
word_frequency = nltk.FreqDist(filtered_tokens)

# Calculate sentiment score
sia = SentimentIntensityAnalyzer()
sentiment_score = sia.polarity_scores(text)

# Calculate summary metrics
original_length, summary_length, precision, recall, f1 = calculate_summary_metrics(text, summary)

# Output results
st.write("Word frequency distribution:")
st.bar_chart(pd.DataFrame(word_frequency.most_common(20), columns=["Word", "Frequency"]).set_index("Word"))
st.write("Sentiment analysis:")
st.write(sentiment_score)
st.write("Summary:")
st.write(summary)
st.write("Summary length:")
st.write(summary_length)
st.write("Summary metrics:")
st.write("Original length:", original_length)
st.write("Precision:", precision)
st.write("Recall:", recall)
st.write("F1 score:", f1)


# Perform exploratory data analysis
perform_eda(text, summary_ratio)
