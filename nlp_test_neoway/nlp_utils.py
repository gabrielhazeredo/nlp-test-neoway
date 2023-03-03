from os import path

import nlp_test_neoway

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('rslp')
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re
import spacy
nlp = spacy.load('pt_core_news_lg')
from unidecode import unidecode

def apply_regex(corpus, regex):
    corpus = [re.sub(regex, ' ', x) for x in corpus]
    return corpus

def multiple_regex(corpus, regex_list):
    # Lowcase
    corpus = corpus.apply(lambda x: x.lower())
    # Negation
    corpus = [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' não ', r) for r in corpus]
    # Basix regex
    for regex in regex_list:
        corpus = apply_regex(corpus, regex)
    return corpus

def stemmer(corpus):
    stemmer = RSLPStemmer()
    stemmetized = []
    for sentence in corpus:
        words = [stemmer.stem(word) for word in sentence]
        sentence = ' '.join(words)
        stemmetized.append(sentence)
    return corpus

def lemmatizer(corpus):
    lemmatized = []
    for sentence in corpus:
        doc = nlp(sentence)
        lemmat = ' '.join([token.lemma_ for token in doc])
        lemmatized.append(lemmat)
    return lemmatized

def data_cleaning(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    # Feature engineering
    df['review_text'] = df['review_title'] + ' ' + df['review_text']
    cols=['submission_date', 'reviewer_id', 'product_id', 'product_name', 'product_brand',
          'site_category_lv1', 'site_category_lv2', 'overall_rating', 'reviewer_birth_year',
          'reviewer_gender', 'reviewer_state', 'review_title']
    df.drop(columns=cols, inplace=True)
    df.dropna(inplace=True)
    # Noise removal
    regex_list = [r'www\S+', r'http\S+', r'@\S+', r'#\S+', r'[0-9]+', r'\W', r'\s+', r'[ \t]+$']
    df['review_text'] = multiple_regex(df['review_text'], regex_list)
    # Lemmatizer
    df['review_text'] = lemmatizer(df['review_text'])
    # Remove accents
    df['review_text'] = df['review_text'].apply(lambda x: " ".join([unidecode(word) for word in x.split()]))
    df.dropna(inplace=True)

    return df