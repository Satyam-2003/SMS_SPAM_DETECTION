import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))  # Convert stopwords to lowercase
    punctuation_set = set(string.punctuation)

    for i in text:
        if i.lower() not in stop_words and i not in punctuation_set:
            y.append(i)

    text = y[:]
    y.clear()

    ps = PorterStemmer()  # Instantiate PorterStemmer outside the loop

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

transformed_sms = transform_text(input_sms)

vector_input = tfidf.transform([transformed_sms])

result = model.predict(vector_input)[0]

if result == 1:
    st.header("Spam")
else:
    st.header("Not a Spam")
