
## Mental Health Sentiment Analysis - NLP Project

Project Type: Natural Language Processing (Text Classification)
Goal: Analyze and classify sentiments in mental health–related text data
Tech Stack: Python · NLP · TensorFlow/Keras · Streamlit
Dataset: Kaggle – Mental Health Sentiment Dataset
 (https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)

### 🧠  Overview

This project applies Deep Learning and NLP to classify text into Positive, Negative, or Neutral sentiments related to mental health expressions.
The data was sourced from Kaggle in CSV format and processed through a standard NLP workflow: text cleaning, tokenization, embedding, model training, evaluation, and deployment with Streamlit for real-time predictions.

### 🧾 Dataset

* Source: Kaggle (CSV file)

* Content: Text entries expressing mental/emotional states

### Columns:

* text → user’s sentence or statement

* sentiment → label (Normal/Depression/SuicidalAnxiety/Stress/Bi-Polar/Personality Disorder)

Use: Supervised classification of text sentiment for mental health awareness

### ⚙️ Steps & Workflow
* 1. Data Collection & Loading

        Dataset downloaded from Kaggle (.csv format)

        Loaded into Pandas for inspection

* 2.Preprocessing

        Lowercasing, punctuation & stopword removal

        Tokenization and Lemmatization (using NLTK or SpaCy)

        Handling URLs, mentions, and emojis

        Removing duplicates and missing entries

* 3.Feature Engineering

        Convert text → sequences

        Apply TF-IDF, Word2Vec, or Embedding layer

        Padding sequences for uniform input shape

* 4.Model Building (Deep Learning + NLP)

        LSTM / BiLSTM-based neural network

        Optional fine-tuning with pretrained BERT / DistilBERT models

        Loss: Categorical Crossentropy

        Optimizer: Adam

* 5.Training & Validation

        Split: 70% Train / 20% Validation / 10% Test

        Monitored with EarlyStopping and ModelCheckpoint

* 6.Evaluation

        Accuracy, Precision, Recall, F1-score

        Confusion Matrix visualization

* 7.Deployment

        Streamlit app for real-time text sentiment prediction

### 💬 Streamlit Real-Time App
Interactive web interface for sentiment prediction:

File: app/streamlit_app.py

### 🧰 Technologies Used

* Python (3.9+)

* TensorFlow / Keras

* NLTK / SpaCy

* Scikit-learn

* Pandas, NumPy

* Streamlit

* Matplotlib / Seaborn
## 🪄 Key Highlights

* ✅ Text preprocessing & vectorization
* ✅ LSTM-based NLP model for sequence learning
* ✅ Real-time sentiment prediction web app
* ✅ Clear visual evaluation metrics
## 📜 License

This project is released under the MIT License.
Please refer to the LICENSE
 file for details.

## ✉️ Contact

Maintainer: Venkatesh
Email: [venkateshvarada56@gmail.com]

GitHub: [https://github.com/VK-Venkatesh]
