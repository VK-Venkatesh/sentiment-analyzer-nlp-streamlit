# app.py

import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app UI
st.set_page_config(page_title="🧠 Sentiment Analysis App", page_icon="💬", layout="centered")

st.title("💬 Text Sentiment Analysis App")
st.write("Enter a product review or description below to predict its sentiment.")

# Text input
user_input = st.text_area("📝 Enter text here:", height=150, placeholder="e.g. This product is amazing and worth every penny!")

# Prediction button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        # Transform input text
        text_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(text_tfidf)[0]

        # Display result
        if prediction == "Anxiety":
            st.success(f"✅ Predicted Sentiment: **{prediction}** 😄")
        elif prediction == "Normal":
            st.error(f"🚫 Predicted Sentiment: **{prediction}** 😞")
        else:
            st.info(f"😐 Predicted Sentiment: **{prediction}**")

# Optional footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and scikit-learn.")
