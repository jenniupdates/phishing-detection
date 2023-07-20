from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
from joblib import Parallel, delayed
import multiprocessing

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


url = 'https://drive.google.com/file/d/1z3AN8qN3Sz1UTkDp_Bd8pGi4qqb2rXxu/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
df.dropna(inplace=True)


# Generate metrics matrix
def metrics(actual, pred):
    print('accuracy: %s%%' % round(accuracy_score(actual, pred)*100, 0))
    print('precision: %s%%' % round(precision_score(actual, pred)*100, 0))
    print('recall: %s%%' % round(recall_score(actual, pred)*100, 0))
    print('f1_score: %s%%' % round(f1_score(actual, pred)*100, 0))


def clean_text(df):
    # Converting the text to lower case
    df['Text'] = df['Text'].astype(str).apply(lambda x: x.lower())
    # Remove "=" symbol from data and replace "\n" with " "
    df['Text'] = df['Text'].apply(lambda x: x.replace("=", ''))
    df['Text'] = df['Text'].apply(lambda x: x.replace("\n", ' '))
    # Extracting url from the text
    df['Url'] = df['Text'].apply(lambda x: re.findall("http\S+", x))
    # Create new feature called Url_Count
    df['Url_Count'] = df['Url'].apply(lambda x: len(x))
    # Extracting email from the text
    df['Email'] = df['Text'].apply(lambda x: re.findall(
        r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", x))
    # Create new feature called Email_Count
    df['Email_Count'] = df['Email'].apply(lambda x: len(x))
    # Removing all symbols from the text except the "$" symbol
    df['Text'] = df["Text"].apply(lambda x: re.sub('[^a-z$\s]', '', x))
    df["Text_Length"] = df["Text"].apply(lambda x: len(x))
    # Removing stop words part 1: Tokenising the text
    df['Text_Tokens'] = df["Text"].apply(lambda x: word_tokenize(x))
    # Removing stop words part 2: Removing stopwords
    df['Text_Filtered'] = df['Text_Tokens'].apply(lambda x: remove_stop_words(x))

    return df


def remove_stop_words(word_tokens):
    nltk_stop_words = stopwords.words('english')
    custom_stop_words = ['.', ',']
    combined_stop_words = nltk_stop_words + custom_stop_words
    filtered_sentence = []
    for w in word_tokens:
        if w not in combined_stop_words:
            filtered_sentence.append(w)
    return (filtered_sentence)


def lemmatize(df):
    # Lemmatize the list of words
    wnl = WordNetLemmatizer()

    def lemmatize(s):
        s = [wnl.lemmatize(word) for word in s]
        return s
    df['Text_Filtered_Lemmatized'] = df['Text_Filtered'].apply(
        lambda x: lemmatize(x))
    # Join the word tokens into strings
    df['Text_Filtered_String'] = df['Text_Filtered_Lemmatized'].apply(
        lambda x: ' '.join(x))
    df['Url_Present'] = df["Url_Count"].apply(lambda x: 1 if x > 0 else 0)

    return df


def clean_df(df):
    df = clean_text(df)
    df = lemmatize(df)

    return df


df = clean_df(df)


######################## TEXT MESSAGE ANALYSIS MODELS ########################
# only included NB and SVM models here as they are the highest accuracy ones


# Split dataset to test and train data
X = df["Text_Filtered_String"]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# Model 1 - Multinomial Naive Bayes (With TfidfVectorizer)

vectorizer = TfidfVectorizer()
X_train_idf = vectorizer.fit_transform(X_train)
with open('./models/text_nb_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

nb = MultinomialNB()
nb.fit(X_train_idf, y_train)
with open('./models/text_nb.pkl', 'wb') as file:
    pickle.dump(nb, file)
# X_test_idf = vectorizer.transform(X_test)
# y_pred = nb.predict(X_test_idf)
# metrics(y_test, y_pred)