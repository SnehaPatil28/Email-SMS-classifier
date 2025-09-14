import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transformed_text(text):
    # lowercase
    text = text.lower()
    # tokenize
    text = nltk.word_tokenize(text)

    # keep only alphanumeric
    text = [i for i in text if i.isalnum()]

    # remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Classifier")
input_sms = st.text_input("Enter the message")


if st.button("Predict"):
    ##preprocess
    transformed_sms = transformed_text(input_sms)
    ##vectorize
    vector_input = tfidf.transform([transformed_sms])
    ##predict
    result = model.predict(vector_input)[0]
    ##display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

