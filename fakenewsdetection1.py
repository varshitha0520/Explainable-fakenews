import streamlit as st

st.title("Explainable Fake News Detection")

news_text = st.text_area("Enter a news article")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        st.success("Processing...")

import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

nltk.download('stopwords')
nltk.download('punkt')

import streamlit as st

news_text = st.text_area("Enter a news article")


data = pd.read_csv("news.csv")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["clean_text"] = data["text"].apply(clean_text)

data.head()

X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

print("✅ Model training completed")

def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    credibility_score = round(max(probabilities) * 100, 2)

    # Explainable AI
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    important_words = sorted(
        zip(feature_names, coefficients),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    reasons = [word for word, weight in important_words]

    # Student-friendly summary
    sentences = sent_tokenize(news_text)
    summary = " ".join(sentences[:2])

    return {
        "Prediction": "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌",
        "Credibility Score": f"{credibility_score}%",
        "Why is it Fake/Real": f"Important words influencing decision: {', '.join(reasons)}",
        "Student-Friendly Summary": summary
    }

nltk.download('punkt_tab')

print("📰 Enter a news article:")
user_news = input()

result = predict_news(user_news)

print("\n🔍 RESULT")
print("----------------------------")
for key, value in result.items():
    print(f"{key}: {value}")
