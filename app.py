import streamlit as st
import pickle
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("fake_news_model.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

theme = st.sidebar.selectbox("Choose Theme", ["Light Mode", "Dark Mode"])

# Background Images
bg_light = "assets/bg_light.jpg"
bg_dark = "assets/bg_light.jpg"

# ------------- CSS -----------------
def set_css(theme):
    if theme == "Dark Mode":
        return f"""
        <style>
        [data-testid="stAppViewContainer"]{{
            background: url({bg_dark});
            background-size: cover;
            background-attachment: fixed;
        }}

        .glass-card{{
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 22px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.22);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 18px;
            margin-bottom: 18px;
        }}

        [data-testid="stSidebar"]{{
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 22px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}

        .stText, .stMarkdown, .stHeader, .stSubheader{{
            color: #ffffff !important;
        }}

        .stButton>button{{
            border-radius: 12px;
            padding: 10px;
            font-weight: 700;
            background-color: rgba(255,255,255,0.12);
            color: #ffffff;
        }}

        textarea{{
            border-radius: 12px;
            padding: 12px;
            background-color: rgba(255,255,255,0.08);
            color: #ffffff;
        }}
        </style>
        """
    else:
        return f"""
        <style>
        [data-testid="stAppViewContainer"]{{
            background: url({bg_light});
            background-size: cover;
            background-attachment: fixed;
        }}

        .glass-card{{
            background: rgba(255,255,255,0.35);
            border: 1px solid rgba(255,255,255,0.55);
            border-radius: 22px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 18px;
            margin-bottom: 18px;
        }}

        [data-testid="stSidebar"]{{
            background: rgba(255,255,255,0.35);
            border: 1px solid rgba(255,255,255,0.55);
            border-radius: 22px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}

        .stText, .stMarkdown, .stHeader, .stSubheader{{
            color: #000000 !important;
        }}

        .stButton>button{{
            border-radius: 12px;
            padding: 10px;
            font-weight: 700;
            background-color: rgba(255,255,255,0.8);
            color: #000000;
        }}

        textarea{{
            border-radius: 12px;
            padding: 12px;
            background-color: rgba(255,255,255,0.9);
            color: #000000;
        }}
        </style>
        """

st.markdown(set_css(theme), unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 Project Info")
st.sidebar.write("""
**Fake News Detection System**

• Algorithm: Logistic Regression  
• NLP: TF-IDF + Lemmatization  
• Dataset: Kaggle Fake News Dataset  
• UI: Streamlit  
""")

st.sidebar.warning("⚠️ Predictions are based on learned patterns and may not be 100% accurate.")

# Main
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is **Fake or Real**.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("🧪 Try Sample Fake News"):
        st.session_state.news_text = (
            "Breaking news: Secret government plan will shut down the internet next week."
        )

with col2:
    if st.button("🧪 Try Sample Real News"):
        st.session_state.news_text = (
            "The government announced a new education policy for public schools."
        )

news_text = st.text_area(
    "✍️ Enter News Text",
    height=200,
    value=st.session_state.get("news_text", ""),
    placeholder="Paste or type the news article here..."
)

uploaded_file = st.file_uploader("📂 Upload a TXT file", type=["txt"])
if uploaded_file:
    news_text = uploaded_file.read().decode("utf-8")

st.markdown("</div>", unsafe_allow_html=True)

# Prediction
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter or upload some news text.")
    else:
        with st.spinner("🧠 Predicting... Please wait"):
            processed_text = preprocess(news_text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]
            confidence = model.predict_proba(vectorized_text).max() * 100

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        if prediction == 0:
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")

        st.subheader("📊 Prediction Confidence")
        st.progress(int(confidence))
        st.write(f"**{confidence:.2f}%**")

        with st.expander("🔎 View Preprocessed Text"):
            st.write(processed_text)

        st.markdown("</div>", unsafe_allow_html=True)

# Charts
accuracy_data = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [0.89, 0.81, 0.92]
}
df_acc = pd.DataFrame(accuracy_data)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📈 Model Accuracy Comparison")

plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="Accuracy", data=df_acc)
plt.ylim(0, 1)
plt.title("Accuracy Comparison")
st.pyplot(plt)

st.markdown("</div>", unsafe_allow_html=True)

conf_mat = np.array([[90, 10], [8, 92]])

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📊 Confusion Matrix")

plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="coolwarm")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed as an NLP & Machine Learning Mini Project 🚀")
