import streamlit as st
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# Streamlit Config
# =======================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =======================
# Custom Theme (CSS)
# =======================
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background-color: #f0f4f8; /* light blue-grey */
        }

        /* Title */
        h1 {
            color: #2c3e50;
            text-align: center;
        }

        /* Buttons */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            color: white;
        }

        /* Text Area */
        .stTextArea textarea {
            border: 2px solid #3498db;
            border-radius: 8px;
            background-color: #ffffff;
        }

        /* Metrics */
        .stMetric {
            background: #ffffff;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# NLTK Setup
# =======================
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english")), PorterStemmer()

# =======================
# Load Model
# =======================
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(base, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Missing files! Please ensure model.pkl and vectorizer.pkl are in this folder.")
        st.stop()

# =======================
# Text Preprocessing
# =======================
def clean_text(text, stop_words, stemmer):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(words)

# =======================
# Main App
# =======================
def main():
    stop_words, stemmer = setup_nltk()
    model, vectorizer = load_model()

    st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
    st.write("Paste any news text below and find out if it’s **Real or Fake** 👇")

    user_text = st.text_area("✍️ Paste your news text here:", height=150)

    if st.button("🔍 Analyze News"):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text first.")
            return

        with st.spinner("Analyzing..."):
            try:
                processed = clean_text(user_text, stop_words, stemmer)
                if not processed.strip():
                    st.warning("⚠️ Couldn’t find meaningful words after cleaning.")
                    return

                vec = vectorizer.transform([processed])
                pred = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]

                st.markdown("---")
                if pred == 1:
                    st.success("✅ This looks like **REAL NEWS**")
                    st.balloons()
                else:
                    st.error("❌ This seems like **FAKE NEWS**")

                st.metric("Confidence Level", f"{probs.max():.1%}", f"Real: {probs[1]:.2f} | Fake: {probs[0]:.2f}")

                st.write("### 📊 Probability Breakdown")
                st.progress(probs[0])
                st.write(f"Fake: {probs[0]:.1%}")
                st.progress(probs[1])
                st.write(f"Real: {probs[1]:.1%}")

            

            except Exception as e:
                st.error(f"❌ Something went wrong: {str(e)}")

    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.info("This tool predicts whether a news article might be **Real or Fake** using AI.")
        st.markdown("### 💡 Tips")
        st.write("- Paste full articles for better accuracy\n- Headlines may be less reliable\n- Results are predictions, not absolute truth")

if __name__ == "__main__":
    main()
