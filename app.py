import streamlit as st
import joblib

# Load pipeline
clf = joblib.load("clf.pkl")

st.title("🎭 Sentiment Analysis App")
st.write("Enter a review and I will predict if it’s Positive, Negative, or Neutral.")

# Input box
text = st.text_area("Write your review here:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        pred = clf.predict([text])[0]

        if pred == 1:
            st.success("✅ Positive Review")
        elif pred == -1:
            st.error("❌ Negative Review")
        else:
            st.info("😐 Neutral Review")
