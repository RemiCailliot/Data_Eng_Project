import numpy as np
print("Starting predictions...")
import pickle
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
pd.options.mode.chained_assignment = None

stop_words=set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

def scrub_words(text):    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    #remove whitespace
    text=text.strip()
    return text
def sw(list_words):
    filtered_sent=[]
    for w in list_words: 
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent
def lm(list_stem):
    lm_words=[]
    for w in list_stem:
        lm_words.append(lemmatizer.lemmatize(w,pos ="a"))
    return lm_words

value="i hate myself !"
value = re.sub('[%s]' % re.escape(string.punctuation), '' , value)
value = scrub_words(value)
value = nltk.word_tokenize(value)
value = sw(value)
value = lm(value)
value = pd.DataFrame({"Value" :[" ".join(value)]})
print(value)

#predictions
model = pickle.load(open('./predict_src/clf.sav', 'rb'))
vectorizer = pickle.load(open('./predict_src/vectorizer.sav', 'rb'))
value = vectorizer.transform(value['Value'])
predicted= model.predict(value)
print("value:",predicted)

