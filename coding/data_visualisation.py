# filename: data_visualisation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
data = pd.read_csv('uploaded_data.csv')

# Analyse the data
# Display the first few rows of the dataset
print(data.head())

# Get summary statistics of the dataset
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Data Visualisation

# Visualization 1: Histogram of a numerical variable
plt.figure(figsize=(10, 6))
sns.histplot(data['numerical_column'], bins=20)
plt.title('Histogram of Numerical Column')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='categorical_column', y='numerical_column', data=data)
plt.title('Box plot of Numerical Column across Categories')
plt.xlabel('Category')
plt.ylabel('Numerical Column')
plt.show()

# Visualization 3: Pairplot
sns.pairplot(data)
plt.show()

# Visualization 4: Count plot of a categorical variable
plt.figure(figsize=(10, 6))
sns.countplot(x='categorical_column', data=data)
plt.title('Count of each Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Visualization 5: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Save the code in data_visualisation.py
with open('data_visualisation.py', 'w') as file:
    file.write('''<paste the code here>''')