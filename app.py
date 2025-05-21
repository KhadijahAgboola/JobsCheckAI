import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_excel('data.xlsx')


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['cleaned_text'] = df['Job Description'].apply(clean_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
# Fit and transform the cleaned text
X = vectorizer.fit_transform(df['cleaned_text'])

# Encode labels
le = LabelEncoder()
df['Label_encoded'] = le.fit_transform(df['Label'])  # 'Fake' becomes 0, 'Real' becomes 1


# Train using encoded labels
X_train, X_test, y_train, y_test = train_test_split(
    X, df['Label_encoded'], test_size=0.2, stratify=df['Label_encoded'], random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Save all
joblib.dump(model, "model_nb.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")


# Load saved models
model = joblib.load("model_nb.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

label_map = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# Suspicious indicators
SUSPICIOUS_PHRASES = [
    "paid certificate", "training reimbursement", "urgent", "submit by", "personal documents",
    "appointment letter attached", "pre-selected", "QREDIV", "fee required"
]
SUSPICIOUS_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com"]
SUSPICIOUS_LINKS = ["bit.ly", "tinyurl", ".xyz", ".ru"]

def detect_flags(text):
    flags = []

    # Flag phrases
    for phrase in SUSPICIOUS_PHRASES:
        if phrase.lower() in text.lower():
            flags.append(f"🔸 Suspicious phrase detected: '{phrase}'")

    # Email domains
    emails = re.findall(r"\S+@\S+", text)
    for email in emails:
        domain = email.split("@")[-1].lower()
        if domain in SUSPICIOUS_DOMAINS:
            flags.append(f"🔸 Suspicious email domain: {domain}")

    # Links
    links = re.findall(r"http[s]?://\S+", text)
    for link in links:
        if any(susp in link for susp in SUSPICIOUS_LINKS):
            flags.append(f"🔸 Suspicious link: {link}")

    return flags

def classify_job(text):
    try:
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        label = label_map[prediction]
        flags = detect_flags(text)

        if label == "Real" and flags:
            verdict = "This job looks suspicious. Please thread carefully"
        elif label == "Real":
            verdict = "🟢 Likely Legit. However, do your due diligence"
        else:
            verdict = "🔴 Likely Fake. Be careful please"

        return verdict, flags
    except Exception as e:
        return f"Error: {e}", []


# Streamlit UI
st.title("JobsCheckAI - A job scam detector")
st.markdown("""
**I built JobsCheckAI after coming across a growing number of fake job emails.** Thankfully, I haven’t fallen victim but many others have, often because the messages look so real. JobsCheckAI is here to help jobseekers stay safe.

**Paste any job description or email content into the box**, and JobsCheckAI will analyze it using AI and custom rules to classify it as 🔴 Likely Fake or 🟢 Likely Legit. It also highlights suspicious phrases, emails, links, and document requests so you can make an informed decision.

Let’s stop fake jobs in their tracks before they cause real damage.
""")

job_text = st.text_area("Paste job description or email content:", height=300)

if st.button("Check Job"):
    if job_text.strip():
        verdict, flags = classify_job(job_text)
        st.subheader("Result")
        st.write(verdict)
    


        if flags:
            st.subheader("⚠️ Suspicious Elements Detected")
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("✅ No obvious red flags found.")
        st.markdown("""
        <div style='color:gray; font-size: 0.9em;'>
        This model uses learned patterns to detect fake jobs, but we also flag suspicious language manually. 
        Always review both the prediction and the flags.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter job content to analyze.")
        
