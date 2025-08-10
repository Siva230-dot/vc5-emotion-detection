import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the training data from CSV
train_data = pd.read_csv('data/interim/train_bow.csv')

# Separate features and target variable
x_train = train_data.drop(columns=['label']).values  # Features
y_train = train_data['label']                 # Target labels

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Save the trained model to a file using pickle
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)