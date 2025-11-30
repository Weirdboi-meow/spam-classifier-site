import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import csv
import os
from datetime import datetime

# ----------------- NLTK SETUP (SAFE FOR STREAMLIT) -----------------
packages = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab/english.pickle",
    "stopwords": "corpora/stopwords",
}

for pkg, path in packages.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

ps = PorterStemmer()

FEEDBACK_FILE = "feedback.csv"  # where user-labelled data will be saved


# ----------------- TEXT PREPROCESSING -----------------
def transform_text(text: str) -> str:
    text = text.lower()
    text = word_tokenize(text)  # use nltk.tokenize.word_tokenize

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ----------------- RULE-BASED SPAM TYPE -----------------
def classify_spam_type(raw_text: str) -> str:
    """Simple rule-based spam category based on keywords."""
    t = raw_text.lower()

    if any(w in t for w in ["password", "otp", "account", "bank", "verify", "login", "update your details"]):
        return "Phishing / Account Fraud"

    if any(w in t for w in ["lottery", "jackpot", "winner", "won", "prize", "lakh", "crore"]):
        return "Lottery / Prize Scam"

    if any(w in t for w in ["offer", "sale", "discount", "deal", "buy now", "limited time"]):
        return "Promotional / Marketing"

    if any(w in t for w in ["job", "work from home", "earn per day", "income", "salary"]):
        return "Fake Job / Income Scheme"

    if any(w in t for w in ["click the link", "click here", "open link", "visit link"]):
        return "Suspicious Link"

    return "General Spam"


# ----------------- LOAD TRAINED MODEL -----------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# ----------------- FEEDBACK SAVING -----------------
def save_feedback(message: str, user_label: str, model_prediction: str, msg_type: str):
    """Append feedback to CSV for later retraining."""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "message", "user_label", "model_prediction", "message_type"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            message,
            user_label,
            model_prediction,
            msg_type
        ])


# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📨 Email / SMS Spam Classifier")

# Initialize session_state BEFORE widget creation
if "message_input" not in st.session_state:
    st.session_state["message_input"] = ""
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_message" not in st.session_state:
    st.session_state["last_message"] = ""
if "last_msg_type" not in st.session_state:
    st.session_state["last_msg_type"] = ""
if "has_prediction" not in st.session_state:
    st.session_state["has_prediction"] = False

# Email / SMS select
msg_type = st.radio("I am checking a:", ["Email", "SMS"], horizontal=True)

# Input with session_state so we can reset
input_mess = st.text_area("Enter the message", key="message_input")

# Sidebar info
st.sidebar.header("Message Info")
st.sidebar.write(f"Type: **{msg_type}**")
st.sidebar.write(f"Characters: **{len(input_mess)}**")


# ----------------- RESET FUNCTION -----------------
def reset_message():
    st.session_state["message_input"] = ""
    st.session_state["has_prediction"] = False
    st.session_state["last_prediction"] = None
    st.session_state["last_message"] = ""
    st.session_state["last_msg_type"] = ""


# ----------------- BUTTONS (centered, side by side) -----------------
left_spacer, mid_col, right_spacer = st.columns([1, 2, 1])

with mid_col:
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        # Reset uses callback
        st.button("Reset", key="reset_btn", on_click=reset_message)
    with bcol2:
        # Predict returns True/False
        predict_clicked = st.button("Predict", key="predict_btn")


# ----------------- PREDICT LOGIC -----------------
if predict_clicked:
    if input_mess.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # 1. Preprocess
        transformed_mess = transform_text(input_mess)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_mess])

        # 3. Predict (0 = ham, 1 = spam)
        result = model.predict(vector_input)[0]

        # Save into session_state for feedback buttons
        st.session_state["has_prediction"] = True
        st.session_state["last_prediction"] = result  # 0 or 1
        st.session_state["last_message"] = input_mess
        st.session_state["last_msg_type"] = msg_type

        # 4. Spam score (if model supports predict_proba)
        try:
            proba = model.predict_proba(vector_input)[0][1]  # prob of class 1 (spam)
            spam_score = proba * 100
        except AttributeError:
            spam_score = None

        # 5. Display result
        if result == 1:
            st.error("🚫 This looks like **SPAM**.")
            if spam_score is not None:
                st.write(f"**Spam score:** {spam_score:.1f}%")

            spam_category = classify_spam_type(input_mess)
            st.write(f"**Spam type (estimated):** {spam_category}")
        else:
            st.success("✅ This looks **NOT spam**.")
            if spam_score is not None:
                st.write(f"**Spam score (spam probability):** {spam_score:.1f}%")


# ----------------- FEEDBACK UI (shown after a prediction exists) -----------------
if st.session_state.get("has_prediction", False) and st.session_state["last_message"].strip() != "":
    st.markdown("---")
    st.subheader("Was this classification correct?")

    col1, col2 = st.columns(2)

    # Model prediction as text label for saving
    model_pred_label = "spam" if st.session_state["last_prediction"] == 1 else "ham"

    with col1:
        if st.button("🚫 Mark as SPAM"):
            save_feedback(
                message=st.session_state["last_message"],
                user_label="spam",
                model_prediction=model_pred_label,
                msg_type=st.session_state["last_msg_type"],
            )
            st.success("Thanks! Saved as **SPAM** for future training.")

    with col2:
        if st.button("✅ Mark as NOT SPAM"):
            save_feedback(
                message=st.session_state["last_message"],
                user_label="ham",
                model_prediction=model_pred_label,
                msg_type=st.session_state["last_msg_type"],
            )
            st.success("Thanks! Saved as **NOT SPAM (HAM)** for future training.")


# ----------------- DISCLAIMER (sidebar) -----------------
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This is a demo machine learning model and may make mistakes.\n\n"
    "Do not rely on it for real security decisions.\n\n"
    "🔒 Messages are not stored on the server. Only if you click the "
    "feedback buttons, the message and label are saved locally."
)
