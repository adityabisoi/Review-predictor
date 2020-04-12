import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)   #Default value to ignore quotes is 3

# Cleaning texts
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])