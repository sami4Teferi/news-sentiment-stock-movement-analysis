import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')
from nltk.corpus import stopwords

# === 1. Load Data === #
def load_data(path):
    df = pd.read_csv(path)
    return df

# === 2. Textual Descriptive Statistics === #
def analyze_text_length(df, text_column='headline'):
    df['text_length'] = df[text_column].astype(str).apply(len)
    print("Headline length statistics:\n", df['text_length'].describe())

    # Plot
    plt.figure(figsize=(8, 4))
    sns.histplot(df['text_length'], bins=50, kde=True)
    plt.title("Distribution of Headline Length")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()

# === 3. Publisher Statistics === #
def analyze_publishers(df, publisher_col='publisher'):
    print("Top publishers by article count:")
    top_publishers = df[publisher_col].value_counts()
    print(top_publishers.head(10))

    top_publishers.plot(kind='bar', figsize=(10, 4), title='Top Publishers')
    plt.ylabel("Article Count")
    plt.xlabel("Publisher")
    plt.show()

    # Email domain extraction if needed
    if df[publisher_col].str.contains("@").any():
        df['publisher_domain'] = df[publisher_col].apply(lambda x: x.split('@')[-1] if '@' in x else 'unknown')
        print("Top email domains:\n", df['publisher_domain'].value_counts().head(10))

# === 4. Time Series Analysis === #
def analyze_publication_dates(df, date_col='date'):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['pub_day'] = df[date_col].dt.date
    df['pub_hour'] = df[date_col].dt.hour

    # Daily frequency
    daily_counts = df['pub_day'].value_counts().sort_index()
    daily_counts.plot(figsize=(12, 4), title="Articles per Day")
    plt.ylabel("Count")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.show()

    # Hourly distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['pub_hour'].dropna(), bins=24, kde=False)
    plt.title("Publishing Hour Distribution")
    plt.xlabel("Hour of Day")
    plt.ylabel("Article Count")
    plt.show()

# === 5. Keyword/Topic Analysis === #
def topic_modeling(df, text_column='headline', n_topics=5):
    # Clean and preprocess text
    stop_words = set(stopwords.words('english'))
    df[text_column] = df[text_column].astype(str)

    def preprocess(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)

    df['clean_text'] = df[text_column].apply(preprocess)

    # Vectorization
    vectorizer = CountVectorizer(max_df=0.9, min_df=10)
    doc_term_matrix = vectorizer.fit_transform(df['clean_text'])

    # LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # Show top words per topic
    print("\n=== Topics Extracted ===")
    for idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        print(f"Topic {idx + 1}: {' | '.join(top_words)}")

    # Optional: word cloud
    word_freq = Counter(" ".join(df['clean_text']).split())
    wc = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Headline Keywords")
    plt.show()

# === 6. Main runner === #
def full_eda(path, text_column='headline', date_col='date', publisher_col='publisher'):
    df = load_data(path)
    print("=== Data Preview ===")
    print(df.head())

    print("\n=== Text Length Analysis ===")
    analyze_text_length(df, text_column)

    print("\n=== Publisher Analysis ===")
    analyze_publishers(df, publisher_col)

    print("\n=== Time Series Analysis ===")
    analyze_publication_dates(df, date_col)

    print("\n=== Topic Modeling ===")
    topic_modeling(df, text_column)

    return df
