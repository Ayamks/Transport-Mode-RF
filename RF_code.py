import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data from pickle file
with open('004_Preprocessed_Data (2).pickle', 'rb') as f:
    All_Df = pickle.load(f)

# Combine all dataframes into one
if isinstance(All_Df, dict):
    combined_df = pd.concat(All_Df.values(), ignore_index=True)
else:
    combined_df = All_Df

# Check if 'Mode' column exists
if 'Mode' not in combined_df.columns:
    raise ValueError("'Mode' column not found in the DataFrame")

# Split the data into training and testing sets first to avoid any leakage
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Preprocess the training data
train_df.dropna(inplace=True)
train_df['Mode'] = train_df['Mode'].astype('category').cat.codes
train_features = train_df.drop(['Mode', 'Date'], axis=1, errors='ignore')
train_classes = train_df['Mode']

# Preprocess the testing data
test_df.dropna(inplace=True)
test_df['Mode'] = test_df['Mode'].astype('category').cat.codes
test_features = test_df.drop(['Mode', 'Date'], axis=1, errors='ignore')
test_classes = test_df['Mode']

# Create a pipeline with a scaler and the Random Forest model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # This ensures scaling is done correctly
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model on the training set
pipeline.fit(train_features, train_classes)

# Make predictions on the test set
test_prediction = pipeline.predict(test_features)

# Calculate accuracy on the test set
acc = accuracy_score(test_classes, test_prediction)
print(f"Test ACCURACY: {acc}")
print("Test Classification Report:")
print(classification_report(test_classes, test_prediction))

# Feature importance
rf_model = pipeline.named_steps['classifier']
df_feature = pd.DataFrame({'featureName': train_features.columns, 'importance': rf_model.feature_importances_})
df_feature = df_feature.sort_values(by='importance', ascending=False)

# Save feature importance to CSV
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
df_feature.to_csv(os.path.join(results_dir, 'random_forest_feature_importance.csv'), index=False)

# Calculate and plot the confusion matrix for the test set
conf_matrix = confusion_matrix(test_classes, test_prediction)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

