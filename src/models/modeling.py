import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

n_estimators = params['modeling']['n_estimators']
max_depth = params['modeling']['max_depth']
# Load the training data from CSV
train_data = pd.read_csv('data/interim/train_bow.csv')

# Separate features and target variable
x_train = train_data.drop(columns=['label']).values  # Features
y_train = train_data['label']                 # Target labels

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth, random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Save the trained model to a file using pickle
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)