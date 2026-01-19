import streamlit as st
import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# -------------------- NLTK SETUP --------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Explainable Fake News Detection")
st.title("📰 Explainable Fake News Detection System")
st.write(
    "Enter a news article to check whether it is **Fake or Real**, "
    "along with an explanation."
)

news_text = st.text_area("✍️ Enter a news article", height=200)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("news.csv")

data = load_data()

# -------------------- DATA VALIDATION --------------------
if len(data.columns) < 2:
    st.error("Dataset must contain at least TWO columns: text and label")
    st.stop()

text_column = data.columns[0]
label_column = data.columns[1]

# -------------------- PREPROCESSING --------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["clean_text"] = data[text_column].apply(clean_text)

X = data["clean_text"]
y = data[label_column]

# -------------------- MODEL TRAINING --------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return model, vectorizer

model, vectorizer = train_model(X, y)

# -------------------- PREDICTION FUNCTION --------------------
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    credibility_score = round(max(probabilities) * 100, 2)

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    important_words = sorted(
        zip(feature_names, coefficients),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    reasons = [word for word, _ in important_words]

    summary = " ".join(sent_tokenize(news_text)[:2])

    return prediction, credibility_score, reasons, summary

# -------------------- BUTTON ACTION --------------------
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        prediction, score, reasons, summary = predict_news(news_text)

        st.subheader("✅ Result")
        if prediction == 1:
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")

        st.metric("📊 Credibility Score", f"{score}%")

        st.subheader("🧠 Why this prediction?")
        st.write("Important words influencing the decision:")
        st.write(", ".join(reasons))

        st.subheader("📘 Student-Friendly Summary")
        st.write(summary)
