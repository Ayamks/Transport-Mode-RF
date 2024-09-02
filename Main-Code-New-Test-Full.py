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
from sklearn.preprocessing import LabelEncoder

# Load preprocessed Geolife training data from pickle file
with open('004_Preprocessed_Data.pickle', 'rb') as f:
    geolife_train_df = pickle.load(f)

# Load preprocessed test data from pickle file
with open('Preprocessed_excel_Test_Data.pickle', 'rb') as f:
    test_df = pickle.load(f)

# Load preprocessed additional training data from pickle file
with open('Preprocessed_excel_Train_Data.pickle', 'rb') as f:
    additional_train_df = pickle.load(f)

# Combine the test and additional training data
combined_test_df = pd.concat([test_df, additional_train_df], ignore_index=True)

# Define function to preprocess data
def preprocess_data(df, label_column, feature_columns):
    # Print columns for debugging
    print("Columns in DataFrame for preprocessing:", df.columns.tolist())

    # Check if all required columns are present
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing columns in the DataFrame: {missing_features}")

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the DataFrame")

    # Handle missing values if any
    df = df.fillna(0)  # or df.dropna() based on your requirement

    # Print the DataFrame columns to ensure 'label_column' is present
    print(f"DataFrame columns before selecting label: {df.columns.tolist()}")

    # Extract features and labels
    X = df[feature_columns]
    y = df[label_column]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Print columns for verification
print("Columns in geolife_train_df:", geolife_train_df.columns.tolist())
print("Columns in combined_test_df:", combined_test_df.columns.tolist())

# Define the columns to use
geolife_features = ['Latitude', 'Longitude', 'Altitude', 'Distance', 'Speed', 'Acc', 'Jerk', 'Bearing', 'Cum_Distance']
common_features = ['Distance', 'Speed', 'Acceleration', 'Jerk', 'Bearing', 'Bearing_rate']  # Ensure these are in both datasets

# Define the label columns
geolife_label_column = 'Mode'  # Label column in the Geolife dataset
test_label_column = 'Label'  # Label column in the test dataset
print("X")
# Ensure the common features list matches the features in the training data
common_features = [feature for feature in common_features if feature in geolife_features]

# Preprocess training data
try:
    X_train, y_train = preprocess_data(geolife_train_df, geolife_label_column, common_features)
except ValueError as e:
    print(f"Error during preprocessing: {e}")
    raise

# Preprocess combined test data
try:
    # Ensure test data has the same features as training data
    combined_test_df.columns = combined_test_df.columns.str.strip()  # Remove leading/trailing spaces
    print("Columns in combined_test_df after stripping spaces:", combined_test_df.columns.tolist())  # Debugging line

    # Print the first few rows of combined_test_df for debugging
    print("First few rows of combined_test_df:\n", combined_test_df.head())

    # Ensure columns are correctly aligned with training data
    X_test = combined_test_df[common_features]  # Select only the features that are present in the training data
    y_test = combined_test_df[test_label_column]

    # Confirm 'Label' column exists
    print(f"DataFrame columns before preprocessing for labels: {combined_test_df.columns.tolist()}")
    if test_label_column not in combined_test_df.columns:
        raise ValueError(f"Label column '{test_label_column}' not found in the DataFrame")

    X_test, y_test = preprocess_data(combined_test_df, test_label_column, common_features)
except ValueError as e:
    print(f"Error during preprocessing: {e}")
    raise

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



