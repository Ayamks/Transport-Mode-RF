import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef, mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data from pickle file
with open('004_Preprocessed_Data.pickle', 'rb') as f:
    All_Df = pickle.load(f)

# Combine all dataframes into one
if isinstance(All_Df, dict):
    combined_df = pd.concat(All_Df.values(), ignore_index=True)
else:
    combined_df = All_Df

# Check if 'Mode' column exists
if 'Mode' not in combined_df.columns:
    raise ValueError("'Mode' column not found in the DataFrame")

# Remove duplicates before splitting
combined_df = combined_df.drop_duplicates()

# Split before any preprocessing
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Function to aggregate trip data with new features
def aggregate_trip_data(df):
    def calc_percentile(x, percentile):
        return np.percentile(x, percentile)

    def calc_covariance(x, y):
        return np.cov(x, y)[0, 1]

    def calc_change_rate(x):
        return np.sum(np.diff(x) != 0) / len(x)

    def rate_category(x, thresholds):
        high = np.sum(x >= thresholds[1]) / len(x)
        medium = np.sum((x < thresholds[1]) & (x >= thresholds[0])) / len(x)
        low = np.sum(x < thresholds[0]) / len(x)
        return high, medium, low

    def heading_change_rate(bearing):
        return np.sum(np.abs(np.diff(bearing)) > 30) / len(bearing)

    def stop_rate(speed):
        return np.sum(speed == 0) / len(speed)

    # Group by 'User' and 'Trip' and calculate summary statistics
    aggregated_df = df.groupby(['User', 'Trip']).agg({
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Altitude': 'mean',
        'Speed': ['mean', 'median', 'min', 'max', lambda x: calc_percentile(x, 85), lambda x: calc_percentile(x, 50)],
        'Acc': ['mean', 'median', 'min', 'max', lambda x: calc_percentile(x, 85), lambda x: calc_percentile(x, 50)],
        'Jerk': 'mean',
        'Bearing': 'mean',
        'Cum_Distance': 'sum',
        'Mode': 'first',  # Assuming 'Mode' doesn't change within a trip
    }).reset_index()

    # Flatten multi-level columns
    aggregated_df.columns = ['_'.join(col).strip('_') for col in aggregated_df.columns.values]

    # Calculate additional features
    cov_speed_acc = df.groupby(['User', 'Trip']).apply(lambda g: calc_covariance(g['Speed'], g['Acc'])).reset_index(drop=True)
    cov_acc_speed = df.groupby(['User', 'Trip']).apply(lambda g: calc_covariance(g['Acc'], g['Speed'])).reset_index(drop=True)
    vcr = df.groupby(['User', 'Trip'])['Speed'].apply(calc_change_rate).reset_index(drop=True)
    acr = df.groupby(['User', 'Trip'])['Acc'].apply(calc_change_rate).reset_index(drop=True)
    hvr, mvr, lvr = zip(*df.groupby(['User', 'Trip'])['Speed'].apply(lambda x: rate_category(x, [10, 30])).apply(pd.Series).values)
    har, mar, lar = zip(*df.groupby(['User', 'Trip'])['Acc'].apply(lambda x: rate_category(x, [0.1, 0.5])).apply(pd.Series).values)
    hcr = df.groupby(['User', 'Trip'])['Bearing'].apply(heading_change_rate).reset_index(drop=True)
    sr = df.groupby(['User', 'Trip'])['Speed'].apply(stop_rate).reset_index(drop=True)

    # Assign new features to the aggregated dataframe
    aggregated_df['Speed_CovV'] = cov_speed_acc.values
    aggregated_df['Acc_CovA'] = cov_acc_speed.values
    aggregated_df['Speed_VCR'] = vcr.values
    aggregated_df['Acc_ACR'] = acr.values
    aggregated_df['Speed_HVR'] = hvr
    aggregated_df['Speed_MVR'] = mvr
    aggregated_df['Speed_LVR'] = lvr
    aggregated_df['Acc_HAR'] = har
    aggregated_df['Acc_MAR'] = mar
    aggregated_df['Acc_LAR'] = lar
    aggregated_df['Bearing_HCR'] = hcr.values
    aggregated_df['Speed_SR'] = sr.values

    return aggregated_df

# Aggregate the train and test dataframes separately
train_aggregated_df = aggregate_trip_data(train_df)
test_aggregated_df = aggregate_trip_data(test_df)

# Check if 'Mode' column exists
if 'Mode_first' not in train_aggregated_df.columns or 'Mode_first' not in test_aggregated_df.columns:
    raise ValueError("'Mode' column not found in the DataFrame")

# Preprocess the training data
train_aggregated_df.dropna(inplace=True)
mode_categories = train_aggregated_df['Mode_first'].astype('category').cat.categories
train_aggregated_df['Mode_first'] = train_aggregated_df['Mode_first'].astype('category').cat.codes
train_features = train_aggregated_df.drop(['Mode_first', 'User_', 'Trip_'], axis=1, errors='ignore')
train_classes = train_aggregated_df['Mode_first']

# Preprocess the testing data
test_aggregated_df.dropna(inplace=True)
test_aggregated_df['Mode_first'] = test_aggregated_df['Mode_first'].astype('category').cat.codes
test_features = test_aggregated_df.drop(['Mode_first', 'User_', 'Trip_'], axis=1, errors='ignore')
test_classes = test_aggregated_df['Mode_first']

# Create a pipeline with a scaler and the Random Forest model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # This ensures scaling is done correctly
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model on the training set
pipeline.fit(train_features, train_classes)

# Make predictions on the test set
test_prediction = pipeline.predict(test_features)
test_prediction_proba = pipeline.predict_proba(test_features)

# Convert numeric predictions back to mode names
test_prediction_modes = pd.Categorical.from_codes(test_prediction, categories=mode_categories)
test_classes_modes = pd.Categorical.from_codes(test_classes, categories=mode_categories)

# Calculate metrics
accuracy = accuracy_score(test_classes_modes, test_prediction_modes)
precision = precision_score(test_classes_modes, test_prediction_modes, average='weighted')  # Change average as needed
recall = recall_score(test_classes_modes, test_prediction_modes, average='weighted')        # Change average as needed
f1 = f1_score(test_classes_modes, test_prediction_modes, average='weighted')                # Change average as needed
conf_matrix = confusion_matrix(test_classes_modes, test_prediction_modes)
roc_auc = roc_auc_score(test_classes_modes.codes, test_prediction_proba, multi_class='ovr')
kappa = cohen_kappa_score(test_classes_modes.codes, test_prediction)
mcc = matthews_corrcoef(test_classes_modes.codes, test_prediction)
mae = mean_absolute_error(test_classes_modes.codes, test_prediction)
rmse = np.sqrt(mean_squared_error(test_classes_modes.codes, test_prediction))

# Print metrics
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1-Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Cohen's Kappa: {kappa:.2f}")
print(f"Matthews Correlation Coefficient: {mcc:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

print("Test Classification Report:")
print(classification_report(test_classes_modes, test_prediction_modes))

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
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=mode_categories, yticklabels=mode_categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation on the training set
scores = cross_val_score(pipeline, train_features, train_classes, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Mean CV Score: {np.mean(scores)}")
