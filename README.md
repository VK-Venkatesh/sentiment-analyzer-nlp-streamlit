# sentiment-analyzer-nlp-streamlit
Mental Health Sentiment Analysis — NLP Project

Project: Sentiment Classification for Mental Health Discussions
Domain: Natural Language Processing (Text-Based)
Dataset Source: Kaggle — Mental Health Sentiment Dataset
 (replace with your dataset link)

🧠 Overview

This project focuses on analyzing text data related to mental health discussions to classify sentiment as Positive, Negative, or Neutral.
Using a CSV dataset from Kaggle, the project applies a Natural Language Processing (NLP) workflow to clean, preprocess, vectorize, and model the text using Deep Learning and traditional ML approaches.

The trained model is deployed through a Streamlit app for real-time sentiment prediction on custom text inputs.

🧾 Dataset

Source: Kaggle

Format: CSV

Columns Example:

text	label
“I’m feeling very anxious today.”	negative
“Therapy has really helped me stay calm.”	positive

Sentiment Classes:

Positive 🙂

Negative 🙁

Neutral 😐

(replace dataset link when publishing)

⚙️ Steps & Workflow

Data Collection

Download dataset from Kaggle

Load .csv file into pandas DataFrame

Text Preprocessing

Lowercasing, punctuation removal, stopword removal

Tokenization & Lemmatization

Handling emojis/emoticons (optional)

Converting text to clean sentences for vectorization

Feature Extraction (Vectorization)

TF-IDF Vectorizer

Word2Vec or GloVe embeddings (for DL models)

Padding & sequence encoding for RNN/LSTM networks

Model Building

Traditional ML: Logistic Regression, SVM, RandomForest

Deep Learning (Preferred):

CNN + LSTM for sentence-level features

BiLSTM / GRU with Embedding layer

Transformers (BERT / DistilBERT fine-tuning optional)

Training & Validation

Split: 70% Train / 20% Validation / 10% Test

Metrics: Accuracy, Precision, Recall, F1-score

Evaluation

Confusion matrix visualization

ROC-AUC Curve

Classification report

Deployment — Streamlit App

Real-time text input for prediction

Displays sentiment label and confidence

🧩 Example Model (LSTM-based)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

📊 Evaluation Metrics
Metric	Value (example)
Accuracy	0.91
Precision	0.89
Recall	0.90
F1-Score	0.89

(update with your actual model results)

💬 Streamlit Real-Time App
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from preprocess_text import clean_text, vectorize

model = load_model('best_model.h5')
st.title("🧠 Mental Health Sentiment Classifier")

text_input = st.text_area("Enter a thought or sentence:")
if st.button("Analyze Sentiment"):
    if text_input.strip():
        X = vectorize(clean_text(text_input))
        pred = model.predict(X)
        label = ['Negative','Neutral','Positive'][np.argmax(pred)]
        st.success(f"**Sentiment:** {label}")

🚀 How to Run

Clone the Repository

git clone https://github.com/<your_username>/Mental-Health-Sentiment-Analysis.git
cd Mental-Health-Sentiment-Analysis


Install Dependencies

pip install -r requirements.txt


Run the App

streamlit run app/streamlit_app.py

🧠 Technologies Used

Python 🐍

Pandas, NumPy

TensorFlow / Keras

Scikit-learn

NLTK / SpaCy

Streamlit

Matplotlib / Seaborn

📜 License

This project is released under the MIT License.

✉️ Contact

Maintainer: Your Name
Email: your.email@example.com

GitHub: @yourusername
