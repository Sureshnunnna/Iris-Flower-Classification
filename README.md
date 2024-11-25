# Iris-Flower-Classification
Project Title: Iris Flower Classification
Description:This project uses the classic Iris dataset to classify flowers into three species (Setosa, Versicolor, Virginica) based on their petal and sepal dimensions. It demonstrates basic machine learning principles and is simple enough for quick implementation.
Simple ML project classifying Iris species using Random Forest.
Built a machine learning model using Random Forest to classify Iris flower species.
Achieved an accuracy of over 90% on test data.
Demonstrated proficiency in Scikit-learn, data preprocessing, and model evaluation.

This project is:
Simple to Implement: Requires minimal setup.
Easy to Explain: Uses a well-known dataset.
Impressive for Recruiters: Demonstrates ML basics effectively.
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Map target labels to species names
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Explore the dataset
print(data.head())
print(data['species'].value_counts())

# Split data into training and testing sets
X = data.iloc[:, :-1]  # Features
y = data['species']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
![image](https://github.com/user-attachments/assets/68c85259-4e5f-4f8b-83dc-8c393c8404d1)
![image](https://github.com/user-attachments/assets/6cdad9df-b4a7-44d3-82d4-7347929e9d68)

