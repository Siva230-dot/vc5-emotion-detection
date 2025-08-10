import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import os

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

max_features = params['feature_engg']['max_features']
# Load the processed training and test data
# NOTE: There is a syntax error in the following lines. 
# .dropna should be called on the DataFrame, not on the string.
# Correct usage: pd.read_csv('path').dropna(subset=["content"])
train_data = pd.read_csv('data/processed/train.csv').dropna(subset=["content"])
test_data = pd.read_csv('data/processed/test.csv').dropna(subset=["content"])

# Extract features and labels from the data
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=max_features)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

# Convert the Bag of Words matrices to DataFrames and add the labels
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train
 
test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# Save the DataFrames to CSV files for later use
# NOTE: The directory 'data/interin/' should exist before running this code
train_df.to_csv('data/interim/train_bow.csv', index=False)
test_df.to_csv('data/interim/test_bow.csv', index=False)