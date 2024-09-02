import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import shap
import numpy as np
from sklearn.inspection import permutation_importance

# Load the pickle files
train_df = pd.read_pickle('filtered_bus_car_trips.pickle')
test_df = pd.read_pickle('trips_summary.pickle')

# Rename 'Acceleration' to 'Acc' in combined_df if needed
if 'Acceleration' in test_df.columns and 'Acc' in train_df.columns:
    test_df.rename(columns={'Acceleration': 'Acc'}, inplace=True)

# Drop non-numeric columns in train_df
train_df = train_df.drop(columns=['Date', 'User'], errors='ignore')

# Handle NaN values if needed (e.g., fill with mean or drop)
test_df = test_df.fillna(0)
train_df = train_df.fillna(0)

# Rename mode column
if 'Label' in test_df.columns and 'Mode' not in test_df.columns:
    test_df.rename(columns={'Label': 'Mode'}, inplace=True)

# Encode the labels
label_encoder = LabelEncoder()
test_df['Mode'] = label_encoder.fit_transform(test_df['Mode'])
train_df['Mode'] = label_encoder.fit_transform(train_df['Mode'])

def calculate_features(df):
    df['Velocity'] = df['Speed']
    df['Acceleration'] = df['Acc']
    features = {
        'Speed': df['Speed'].mean(),
        'Distance': df['Distance'].mean(),
        'Mode': df['Mode'].mean(),
        'Acc': df['Acc'].mean(),
        'Jerk': df['Jerk'].mean(),
        'Bearing': df['Bearing'].mean(),
        '85thV': df['Velocity'].quantile(0.85),
        '85thA': df['Acceleration'].quantile(0.85),
        'MaxV1': df['Velocity'].max(),
        'MaxA1': df['Acceleration'].max(),
        'MaxV2': df['Velocity'].nlargest(2).iloc[-1],
        'MaxA2': df['Acceleration'].nlargest(2).iloc[-1],
        'MedianV': df['Velocity'].median(),
        'MedianA': df['Acceleration'].median(),
        'MinV': df['Velocity'].min(),
        'MinA': df['Acceleration'].min(),
        'MeanV': df['Velocity'].mean(),
        'MeanA': df['Acceleration'].mean(),
        'ExpV': df['Velocity'].mean(),
        'ExpA': df['Acceleration'].mean(),
        'CovV': df['Velocity'].cov(df['Velocity']),
        'CovA': df['Acceleration'].cov(df['Acceleration']),
        'VCR': df['Velocity'].diff().mean(),
        'ACR': df['Acceleration'].diff().mean(),
        'HVR': (df['Velocity'] > df['Velocity'].quantile(0.75)).mean(),
        'MVR': ((df['Velocity'] > df['Velocity'].quantile(0.25)) & (df['Velocity'] <= df['Velocity'].quantile(0.75))).mean(),
        'LVR': (df['Velocity'] <= df['Velocity'].quantile(0.25)).mean(),
        'HAR': (df['Acceleration'] > df['Acceleration'].quantile(0.75)).mean(),
        'MAR': ((df['Acceleration'] > df['Acceleration'].quantile(0.25)) & (df['Acceleration'] <= df['Acceleration'].quantile(0.75))).mean(),
        'LAR': (df['Acceleration'] <= df['Acceleration'].quantile(0.25)).mean(),
        'BSR': (df['Speed'] == 0).mean(),
        'HCR': df['Bearing'].diff().mean(),
        'SR': (df['Speed'] == 0).mean(),
    }
    return pd.Series(features)

# Calculate features for each trip in the combined DataFrame
#test_features_df = test_df.groupby('Trip').apply(calculate_features).reset_index()
train_features_df = train_df.groupby('Trip').apply(calculate_features).reset_index()
test_features_df = test_df

# Drop trip column from dataframes
train_features_df = train_features_df.drop(['Trip'], axis=1, errors='ignore')
test_features_df = test_features_df.drop(['Trip'], axis=1, errors='ignore')
# Ensure all data is numeric
test_features_df = test_features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Ensure that the feature columns match those in the training DataFrame
common_columns = set(train_features_df.columns) & set(test_features_df.columns)
print("Common columns:", common_columns)

# Define features and target for training data
X_train = train_features_df.drop(['Mode'], axis=1, errors='ignore')
y_train = train_features_df['Mode']

# Remove NaN values from y_train
y_train = y_train.dropna()

# Remove corresponding rows from X_train
X_train = X_train.loc[y_train.index]

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prepare test data
X_test = test_features_df.drop(['Mode', 'Label'], axis=1, errors='ignore')
y_test = test_features_df['Mode']

# Remove NaN values from y_test
y_test = y_test.dropna()

# Remove corresponding rows from X_test
X_test = X_test.loc[y_test.index]

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# Ensure Mode column is an integer type
train_features_df['Mode'] = train_features_df['Mode'].astype(int)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Get unique modes for label decoding
unique_modes = np.sort(train_features_df['Mode'].unique())
tick_labels = label_encoder.inverse_transform(unique_modes)

# Plot confusion matrix with titles
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=tick_labels,
            yticklabels=tick_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate Gini Importance
importances = model.feature_importances_
feature_names = X_train.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)

# Plot Gini Importance
plt.figure(figsize=(8, 8))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Gini Importance'], color='skyblue')
plt.xlabel('Gini Importance')
plt.title('Feature Importance - Gini Importance')
plt.gca().invert_yaxis()
plt.show()

# Calculate Mean Decrease Accuracy
importances_mda = []
initial_accuracy = accuracy_score(y_test, model.predict(X_test))  # initial accuracy
for i in range(X_test.shape[1]):
    X_test_copy = X_test.copy()
    np.random.shuffle(X_test_copy.iloc[:, i])
    shuff_accuracy = accuracy_score(y_test, model.predict(X_test_copy))
    importances_mda.append(initial_accuracy - shuff_accuracy)

accuracy_df = pd.DataFrame({'Feature': feature_names, 'Decrease in Accuracy': importances_mda}).sort_values('Decrease in Accuracy', ascending=False)

# Plot Mean Decrease Accuracy
plt.figure(figsize=(8, 8))
plt.barh(accuracy_df['Feature'], accuracy_df['Decrease in Accuracy'], color='skyblue')
plt.xlabel('Mean Decrease Accuracy')
plt.title('Feature Importance - Mean Decrease Accuracy')
plt.gca().invert_yaxis()
plt.show()

# Permutation Importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
perm_imp_df = pd.DataFrame({'Feature': feature_names, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)

# Plot Permutation Importance
plt.figure(figsize=(8, 6))
plt.bar(perm_imp_df['Feature'], perm_imp_df['Permutation Importance'])
plt.xlabel('Feature')
plt.ylabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualization of SHAP summary
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

print(f"Accuracy: {accuracy}")

# Mean Absolute SHAP Value
#shap_summary = np.abs(shap_values).mean(axis=0)
#shap_summary_df = pd.DataFrame({'Feature': feature_names, 'SHAP values': shap_summary})
#shap_summary_df = shap_summary_df.sort_values('SHAP values', ascending=False)

# Plot Mean Absolute SHAP Values
#plt.figure(figsize=(10, 6))
#plt.barh(shap_summary_df['Feature'], shap_summary_df['SHAP values'], color='skyblue')
#plt.xlabel('Mean Absolute SHAP Value')
#plt.ylabel('Feature')
#plt.title('Feature Importance based on SHAP Values')
#plt.gca().invert_yaxis()
#plt.show()

