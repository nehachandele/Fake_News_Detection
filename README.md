# 📰 Fake News Detection System (NLP + ML + Streamlit)

This project detects whether a news article is **Fake or Real** using Natural Language Processing (NLP) and Machine Learning.  
The trained model is deployed using **Streamlit**, providing a user-friendly web interface for real-time predictions.

---

## 🚀 Project Overview

- **Problem Statement:** Fake news spreads quickly and misleads people. This project aims to automatically classify news articles as Fake or Real.
- **Approach:**  
  - Preprocess text using NLP (cleaning, stopword removal, lemmatization)
  - Convert text into numerical features using **TF-IDF**
  - Train a **Logistic Regression** classifier
  - Deploy the model using **Streamlit** UI

---

## 🧠 Dataset

The dataset is taken from **Kaggle** (Fake News Detection Dataset).  
It contains columns like:

| Column | Description |
|--------|-------------|
| title  | News title |
| text   | Full news content |
| label  | 0 → Real, 1 → Fake |

---

## 🛠️ Features

✔ Preprocessing using NLP  
✔ TF-IDF feature extraction  
✔ Logistic Regression model  
✔ Streamlit UI with real-time prediction  
✔ Confidence score  
✔ Sample test buttons  
✔ TXT file upload  

---

## 📌 Folder Structure

Fake_News_Detection/
│
├── app.py
├── vectorizer.pkl
├── fake_news_model.pkl
├── requirements.txt
└── README.md


---

## 📌 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py


How to Use

1.Open the app in your browser.

2.Paste news text or upload a .txt file.

3.Click Analyze News.

4.View result (Fake/Real) with confidence score.

Model Details
Component	               Description
Preprocessing	           Lowercase, remove symbols,    stopwords removal, lemmatization
Feature                    Extraction	TF-IDF Vectorizer
Model	                   Logistic Regression
Labels  	               0 → Real, 1 → Fake


Sample Output
Prediction: Fake News
Confidence: 92.35%