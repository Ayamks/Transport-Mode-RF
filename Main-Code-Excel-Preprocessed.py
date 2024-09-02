import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed training data from pickle file
with open('Preprocessed_excel_Train_Data.pickle', 'rb') as f:
    geolife_train_df = pickle.load(f)

# Load preprocessed testing data from pickle file
with open('Preprocessed_excel_Test_Data.pickle', 'rb') as f:
    test_df = pickle.load(f)

# Define features and target for training
X_train = geolife_train_df.drop(columns=['Mode'])  # Features
y_train = geolife_train_df['Mode']  # Target

# Define features and target for testing
X_test = test_df.drop(columns=['Label'])  # Features
y_test = test_df['Label']  # Target

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {accuracy}")
print("Test Classification Report:")
print(classification_report_str)



# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


