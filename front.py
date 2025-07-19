import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open("rf.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🎯 Sentiment Review Classifier")
st.write("Enter a movie review below and let the Random Forest model predict if it's positive or negative!")

# Input box
review = st.text_area("📝 Your Review")

# Predict button
if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        # Vectorize user input
        review_vector = vectorizer.transform([review]).toarray()

        # Predict using loaded model
        prediction = model.predict(review_vector)

        # Show result
        sentiment = "🌟 Positive" if prediction[0] == 1 else "💔 Negative"
        st.success(f"Prediction: {sentiment}")
