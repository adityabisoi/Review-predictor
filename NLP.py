import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)   #Default value to ignore quotes is 3

# Cleaning texts
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()

# Removing non-determiners and stemming
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = review.split()
words = stopwords.words('english')
ps = PorterStemmer()
review = [ps.stem(i) for i in review if i not in set(words)]    # set() is optional, can be used for faster filtering
review = ' '.join(review)