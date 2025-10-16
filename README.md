# sentiment-analyzer-nlp-streamlit
# Mental Health Sentiment Analysis — NLP Project

### * Project: Sentiment Classification for Mental Health Discussions
### * Domain: Natural Language Processing (Text-Based)
### * Dataset Source: Kaggle — Mental Health Sentiment Dataset
 (replace with your dataset link)

## 🧠  Overview

This project focuses on analyzing text data related to mental health discussions to classify sentiment as Positive, Negative, or Neutral.
Using a CSV dataset from Kaggle, the project applies a Natural Language Processing (NLP) workflow to clean, preprocess, vectorize, and model the text using Deep Learning and traditional ML approaches.

The trained model is deployed through a Streamlit app for real-time sentiment prediction on custom text inputs.

## 🧾 Dataset

* Source: Kaggle

* Format: CSV

* Columns Example:

**text	label**
“I’m feeling very anxious today.”	 Anxiety
“Therapy has really helped me stay calm.”	Stress

**Sentiment Classes:**

1.Normal
2.Depression
3.Suicidal
4.Anxiety
5.Stress
6.Bi-Polar
7.Personality Disorder

## ⚙️ Steps & Workflow

* 1.Data Collection

* 2. Download dataset from Kaggle

* 3.Load .csv file into pandas DataFrame

* 4.Text Preprocessing

* 5.Lowercasing, punctuation removal, stopword removal

* 6.Tokenization & Lemmatization

* 7.Handling emojis/emoticons (optional)

* 8.Converting text to clean sentences for vectorization

* 9.Feature Extraction (Vectorization)

* 10.TF-IDF Vectorizer

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

## 💬 Streamlit Real-Time App


## 📜 License

This project is released under the MIT License.

## ✉️ Contact

Maintainer: Venkatesh
Email: venkateshvarada56@gmail.com

GitHub: 
