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