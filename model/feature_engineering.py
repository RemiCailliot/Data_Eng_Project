
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
pd.options.mode.chained_assignment = None
#get modified dataset for feature_engineering
df=pd.read_csv("./data/prepared_application_train.csv")

#remove punctuation
print("Removing punctuation in the dataset...")
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

#clean text
print("Cleaning text of all markups and digit noise in the dataset...")
def scrub_words(text):    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    #remove whitespace
    text=text.strip()
    return text
df['clean_text'] = df.apply(lambda row: scrub_words(row['clean_text']), axis=1)

#tokenize text by word
print("Tokenizing texts in words...")
df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['clean_text']), axis=1)

#remove stopwords
print("Removing stopwords in the dataset...")
stop_words=set(stopwords.words("english"))
def sw(list_words):
    filtered_sent=[]
    for w in list_words: 
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent
df['tokenized_text_stopwords'] = df.apply(lambda row: sw(row['tokenized_text']), axis=1)

# lemmatize words
print("Lemmatizing text in the dataset...")
lemmatizer = WordNetLemmatizer()
def lm(list_stem):
    lm_words=[]
    for w in list_stem:
        lm_words.append(lemmatizer.lemmatize(w,pos ="a"))
    return lm_words
df['tokenized_text_stopwords_lemmatized'] = df.apply(lambda row: lm(row['tokenized_text_stopwords']), axis=1)

#save data for next stage
print("Saving Dataset")
df.to_csv("./data/final_application_train.csv",index=False)
print("Done!")