import streamlit as st
import pickle
import os
import numpy as np

# Load the trained model and vectorizer
try:
    model = pickle.load(open("mnb.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please train and save them first using your notebook.")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# App title and description
st.title("🎥 Movie Review Sentiment Classifier")
st.subheader("Analyze Your Review")
st.write("""
Enter a movie review below, and our model will predict if it's positive or negative. 
We've trained this on IMDB data for accurate sentiment detection!
""")

# Examples section
with st.expander("📖 See Example Reviews"):
    st.write("- **Positive Example**: 'This movie was absolutely fantastic! The acting was top-notch and the plot kept me on the edge of my seat.'")
    st.write("- **Negative Example**: 'What a waste of time. The story made no sense and the characters were boring.'")

# Input box with placeholder
review = st.text_area("📝 Type Your Review Here", placeholder="e.g., 'I loved the twists in this thriller!'", height=150)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review to analyze.")
    else:
        try:
            # Vectorize the input
            review_vector = vectorizer.transform([review]).toarray()
            
            # Predict sentiment
            prediction = model.predict(review_vector)[0]
            sentiment = "🌟 Positive" if prediction == 1 else "💔 Negative"
            
            # Get prediction confidence (probability)
            prob = model.predict_proba(review_vector)[0]
            confidence = np.max(prob) * 100  # Highest probability as confidence percentage
            
            # Display results
            st.success(f"**Prediction:** {sentiment}")
            st.info(f"**Confidence Level:** {confidence:.2f}%")
            
            # Fun feedback based on sentiment
            if prediction == 1:
                st.balloons()  # Streamlit's fun animation for positive
            else:
                st.snow()  # Animation for negative
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Model trained on IMDB Dataset")
