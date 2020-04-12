import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)   #Default value to ignore quotes is 3

# Cleaning texts
# Removing non-determiners and stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
words = stopwords.words('english')
corpus = []                                                        # Initialize an empty list for all reviews

for x in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][x])
    review = review.lower()
    review = review.split()
    review = [ps.stem(i) for i in review if i not in set(words)]    # set() is optional, can be used for faster filtering
    review = ' '.join(review)
    corpus.append(review)
    
# Creating a Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)                           # 1500 most frequent words
X = cv.fit_transform(corpus).toarray()                              # toarray() is used to convert into matrix
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)