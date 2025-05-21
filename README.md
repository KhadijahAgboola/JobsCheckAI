# JobsCheckAI

JobsCheckAI is an AI-powered application designed to help jobseekers identify potentially fake job descriptions or scam emails. It uses a machine learning model trained on real and fake job postings, along with custom rule-based checks to flag suspicious language, email domains, and links.

This project was created after encountering an increasing number of fake job offers. Although I have not personally fallen victim, I realized many others could. JobsCheckAI was built to support jobseekers in making safer decisions.

Simply paste a job description or job offer email into the app, and it will classify the content as likely fake or likely legitimate. It will also highlight red flags like urgent language, requests for documents, suspicious links, and known scam keywords.

## Features

- Classifies job postings as likely real or fake using AI
- Highlights suspicious phrases and patterns commonly found in scams
- Detects risky email domains and short links often used in fraudulent messages
- Provides a combined prediction and flag-based evaluation

## How to Use

1. Paste the job description or email text into the text box
2. Click the button to analyze the job
3. View the result and any flagged elements to make an informed decision

## Installation

To run the app locally:

1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Run the app using `streamlit run your_script_name.py`

## Deployment

JobsCheckAI can be deployed on Streamlit Cloud. Make sure to include your trained model files (`model_nb.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`) in the repository.

## Requirements

- streamlit
- joblib
- scikit-learn
- pandas
- openpyxl
