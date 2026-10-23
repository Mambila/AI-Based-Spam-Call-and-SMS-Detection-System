import streamlit as st
import joblib
import os
import pandas as pd
from src.call_scoring import score_call_metadata

# Page Configuration
st.set_page_config(
    page_title="AI-Based Spam Detection System - Zambia",
    page_icon="🛡️",
    layout="wide"
)

# Load saved model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/spam_classifier.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# Title and Overview
st.title("AI-Based Spam Call and SMS Detection System")
st.write("Protecting Zambian mobile subscribers from mobile money fraud, deceptive marketing and unwanted communications.")

# Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["📲 SMS Classification", "📞 Call Metadata Risk Check", "📝 Feedback & Reports"])

# TAB 1: SMS Detection
with tab1:
    st.header("SMS Spam Risk Detector")
    st.write("Analyze suspicious messages such as mobile money fraud or fake job offers")
    
    sms_text = st.text_area("Enter the message to scan:", placeholder="Type or paste message here...")
    
    if st.button("Check SMS Risk"):
        if sms_text.strip() == "":
            st.warning("Please enter a message first.")
        else:
            # Transform text using TF-IDF and predict
            input_vector = vectorizer.transform([sms_text])
            probability = model.predict_proba(input_vector)[0][1] # Probability of spam
            
            st.markdown("---")
            st.subheader("Analysis Results:")
            
            # Three-tier risk score mapping
            if probability > 0.7:
                st.error(f"**Risk Level: SPAM 🚨**\n\n*Spam Probability Score:* `{probability:.2f}`\n\n*Warning:* High likelihood of being a fraudulent message (e.g., mobile money or pin-stealing scam). Do not reply or send money!")
            elif probability > 0.3:
                st.warning(f"**Risk Level: SUSPICIOUS ⚠️**\n\n*Spam Probability Score:* `{probability:.2f}`\n\n*Warning:* This message contains elements commonly associated with unsolicited promotions or unverified offers. Proceed with caution.")
            else:
                st.success(f"**Risk Level: SAFE ✅**\n\n*Spam Probability Score:* `{probability:.2f}`\n\n*Status:* This message appears legitimate.")

# TAB 2: Call Metadata Scoring
with tab2:
    st.header("Call Behavior Risk Scorer")
    st.write("Evaluate incoming call risk using anonymised behavior metrics (duration, frequency, and format) without violating user privacy[cite: 1].")
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Call Duration (seconds):", min_value=0, value=15, help="Very short calls often indicate robocalls or missed-call scams.")
        frequency = st.number_input("Calls received from this number today:", min_value=1, value=1, help="High call frequency from a single unknown number.")
    with col2:
        is_intl = st.checkbox("Is international or unfamiliar prefix?", help="Check if the call originates from an unusual or unrecognized code.")
        
    if st.button("Analyze Call Risk"):
        tier, score = score_call_metadata(duration, frequency, is_intl)
        
        st.markdown("---")
        st.subheader("Call Risk Assessment:")
        if tier == "Spam":
            st.error(f"**Call Risk Tier: SPAM 🚨** (Risk Score: `{score}`)\n\n*Recommendation:* Avoid answering or block this number immediately.")
        elif tier == "Suspicious":
            st.warning(f"**Call Risk Tier: SUSPICIOUS ⚠️** (Risk Score: `{score}`)\n\n*Recommendation:* Exercise caution if engaging with this caller.")
        else:
            st.success(f"**Call Risk Tier: SAFE ✅** (Risk Score: `{score}`)\n\n*Recommendation:* Normal call patterns detected.")

# TAB 3: Feedback Loop
with tab3:
    st.header("Subscriber Feedback & Correction Hub")
    st.write("Help improve detection accuracy by reporting misclassified communications. Submissions are stored for periodic model retraining[cite: 1].")
    
    reported_content = st.text_input("Enter the misclassified message or phone number:")
    correct_label = st.selectbox("What is the correct classification?", ["Safe", "Spam"])
    
    if st.button("Submit Correction"):
        if reported_content.strip() == "":
            st.warning("Please provide the content or number before submitting.")
        else:
            feedback_file = 'feedback/feedback_store.csv'
            os.makedirs('feedback', exist_ok=True)
            new_data = pd.DataFrame([[reported_content, correct_label]], columns=['content', 'actual_label'])
            
            if os.path.exists(feedback_file):
                new_data.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                new_data.to_csv(feedback_file, index=False)
                
            st.success("Thank you! Your correction has been successfully logged to the feedback store for future retraining.")