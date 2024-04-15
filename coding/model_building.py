# filename: model_building.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('uploaded_data.csv')

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build 5 different models
models = [RandomForestClassifier(), SVC(), LogisticRegression()]

model_accuracies = []

# Train and evaluate each model
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies.append((model, accuracy))

# Find the top 5 models based on accuracy
top_models = sorted(model_accuracies, key=lambda x: x[1], reverse=True)[:5]

# Choose the best model
best_model = max(model_accuracies, key=lambda x: x[1])[0]

# Model evaluation for the best model
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_test, y_pred_best)

# Save the best model
import joblib
joblib.dump(best_model, 'best_model.joblib')

# Visualization code (example using matplotlib)
import matplotlib.pyplot as plt

# Create and save visualizations
plt.figure(figsize=(10, 6))
plt.bar(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'],
        [acc for _, acc in top_models])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Top 5 Model Accuracies')
plt.savefig('top_model_accuracies.png')

# Save the model building code
code = """
# Code for model building
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('uploaded_data.csv')

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
"""
with open('ML_model.py', 'w') as file:
    file.write(code)