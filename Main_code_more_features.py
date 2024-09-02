import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Check if 'TS' column exists
if 'TS' not in combined_df.columns:
    raise ValueError("'TS' column not found in the DataFrame")

# Split before any preprocessing
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Function to enhance trip data with new features
def enhance_trip_data(df):
    def calc_time_of_day(timestamp):
        return timestamp.hour + timestamp.minute / 60

    def calc_day_of_week(timestamp):
        return timestamp.dayofweek

    def calc_trip_duration(timestamps):
        return (timestamps.max() - timestamps.min()).total_seconds() / 60  # in minutes

    def calc_stop_duration(speed):
        return np.sum(speed == 0)

    # Assuming df has a 'TS' column in datetime format
    df['TS'] = pd.to_datetime(df['TS'])
    df['TimeOfDay'] = df['TS'].apply(calc_time_of_day)
    df['DayOfWeek'] = df['TS'].apply(calc_day_of_week)

    # Group by 'User' and 'Trip' and calculate new features
    enhanced_df = df.groupby(['User', 'Trip']).agg({
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Altitude': 'mean',
        'Speed': ['mean', 'median', 'min', 'max', 'std', lambda x: calc_stop_duration(x)],
        'Acc': ['mean', 'median', 'min', 'max', 'std'],
        'Jerk': 'mean',
        'Bearing': 'mean',
        'Cum_Distance': 'sum',
        'Mode': 'first',  # Assuming 'Mode' doesn't change within a trip
        'TimeOfDay': 'mean',
        'DayOfWeek': 'mean',
        'TS': lambda x: calc_trip_duration(x)
    }).reset_index()

    # Flatten multi-level columns
    enhanced_df.columns = ['_'.join(col).strip('_') for col in enhanced_df.columns.values]

    return enhanced_df

# Aggregate the train and test dataframes separately
train_aggregated_df = enhance_trip_data(train_df)
test_aggregated_df = enhance_trip_data(test_df)

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

# Convert numeric predictions back to mode names
test_prediction_modes = pd.Categorical.from_codes(test_prediction, categories=mode_categories)
test_classes_modes = pd.Categorical.from_codes(test_classes, categories=mode_categories)

# Calculate accuracy on the test set
acc = accuracy_score(test_classes_modes, test_prediction_modes)
print(f"Test ACCURACY: {acc}")
print("Test Classification Report:")
print(classification_report(test_classes_modes, test_prediction_modes))

# Print the test dataframe for debugging
print(test_aggregated_df)

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
conf_matrix = confusion_matrix(test_classes_modes, test_prediction_modes)

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











