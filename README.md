**📨Email/SMS Spam Classifier**

A simple Streamlit app that classifies email and SMS messages as spam or ham using a TF-IDF–based machine learning model.


**⚙️Features**

- Real-time spam/ham prediction

- Basic spam-type tagging (phishing, lottery, promo, etc.)

- User feedback buttons (mark as spam/ham)

- Feedback stored in feedback.csv for retraining

- NLTK preprocessing (tokenizing, stopwords, stemming)


**📊Dataset Information**

This project uses two publicly available datasets for training:

- SMS Spam Collection (UCI Machine Learning Repository)
Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

- Email Spam Classification Dataset by Ashfak Yeafi (Kaggle)
Source: https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification

The full datasets are not included in this repository due to licensing and redistribution restrictions.
A small sample file is provided only to show the expected format.


**📂Files**
- app.py — main app

- model.pkl — trained classifier

- vectorizer.pkl — TF-IDF vectorizer

- feedback.csv — saved user feedback

- original_training_data.csv — base dataset


**📒Retraining**
Load both the original dataset and feedback.csv, retrain the model, and save new .pkl files.
