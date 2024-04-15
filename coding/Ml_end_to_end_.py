# filename: Ml_end_to_end_.py

# Analyzing the given dataset

# Import necessary libraries
import pandas as pd

# Load the dataset
data = pd.read_csv('coding/uploaded_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Display information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Describe the dataset
print(data.describe())

# Visualize the data using libraries like matplotlib or seaborn
# Example:
import matplotlib.pyplot as plt
import seaborn as sns

# Example visualization
plt.figure(figsize=(10, 6))
sns.histplot(data['column_name'], bins=20, kde=True)
plt.title('Histogram of a Column')
plt.xlabel('Column Values')
plt.ylabel('Frequency')
plt.show()

# Build a machine learning model (e.g., using scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample code to build a model (you need to adjust based on your data)
X = data.drop('target_column', axis=1)
y = data['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', accuracy)

# Save the model
import joblib

joblib.dump(model, 'saved_model.pkl')
