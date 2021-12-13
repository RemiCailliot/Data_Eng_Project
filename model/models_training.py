import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
pd.options.mode.chained_assignment = None
df=pd.read_csv("../data/final_application_train.csv")

#drop all unused columns
print("Dropping unused columns in the dataset...")
df2 = df[['tokenized_text_stopwords_lemmatized','category']]
#change data type from list to str
print("Changing values type from list to string...")
df2['tokenized_text_stopwords_lemmatized']=df2['tokenized_text_stopwords_lemmatized'].apply(lambda x: " ".join(x) )
#tokenizer to remove unwanted elements from out data like symbols and numbers
print("Removing unwanted elements like symbols and numbers in the dataset...")
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit(df2['tokenized_text_stopwords_lemmatized'])
print("Saving vectorizer...")
text_counts2 = text_counts.transform(df2['tokenized_text_stopwords_lemmatized'])
#separate train & test
print("Separating the dataset into train and test...")
X_train, X_test, y_train, y_test = train_test_split(
    text_counts2, df2['category'], test_size=0.2, random_state=1)


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)


y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
# Model Generation Using Multinomial Naive Bayes
# print("Training of the model...")
# model = MultinomialNB(alpha=2)
# clf = model.fit(X_train, y_train)
# predicted= clf.predict(X_test)
# print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Saving model...")
pickle.dump(sgd, open('../predict_src/clf.sav', 'wb'))
pickle.dump(text_counts, open('../predict_src/vectorizer.sav', 'wb'))
print("Done!")