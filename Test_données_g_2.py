import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the pickle files
train_df = pd.read_pickle('filtered_bus_car_walk_trips.pickle')
test_df = pd.read_pickle('Preprocessed_excel_Test_Data.pickle')
combined_df = pd.read_pickle('Preprocessed_excel_Train_Data.pickle')

# Combine train and test DataFrames
combined_df = pd.concat([combined_df, test_df], ignore_index=True)

# Rename 'Acceleration' to 'Acc' in combined_df if needed
if 'Acceleration' in combined_df.columns and 'Acc' in train_df.columns:
    combined_df.rename(columns={'Acceleration': 'Acc'}, inplace=True)

print("Combined DataFrame columns:", combined_df.columns)

# Drop non-numeric columns in train_df
train_df = train_df.drop(columns=['Date', 'User'], errors='ignore')

# Handle NaN values if needed (e.g., fill with mean or drop)
combined_df = combined_df.fillna(0)
train_df = train_df.fillna(0)

# Rename mode column
if 'Label' in combined_df.columns and 'Mode' not in combined_df.columns:
    combined_df.rename(columns={'Label': 'Mode'}, inplace=True)

# Encode the labels
label_encoder = LabelEncoder()
combined_df['Mode'] = label_encoder.fit_transform(combined_df['Mode'])
train_df['Mode'] = label_encoder.fit_transform(train_df['Mode'])
def calculate_features(df):
    df['Velocity'] = df['Speed']
    df['Acceleration'] = df['Acc']
    #df['Elevation'] = df['Altitude']
    #df['Label'] = df['Mode']
    features = {
        #'Trip': df['Trip'].mean(),
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
combined_features_df = combined_df.groupby('Trip').apply(calculate_features).reset_index()
train_features_df = train_df.groupby('Trip').apply(calculate_features).reset_index()


# Ensure all data is numeric
combined_features_df = combined_features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
#train_features_df = train_features_df.apply(pd.to_numeric, errors='coerce').fillna(0)


# Check columns in both DataFrames
print("Columns in combined_features_df:", combined_features_df.columns)
print("Columns in train_features_df:", train_features_df.columns)

# Ensure that the feature columns match those in the training DataFrame
common_columns = set(train_features_df.columns) & set(combined_features_df.columns)
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
X_test = combined_features_df.drop(['Mode', 'Label'], axis=1, errors='ignore')
y_test = combined_features_df['Mode']

# Remove NaN values from y_test
y_test = y_test.dropna()

# Remove corresponding rows from X_test
X_test = X_test.loc[y_test.index]

# Make predictions
y_pred = model.predict(X_test)

# Decode the labels for the predictions
#y_test_decoded = label_encoder.inverse_transform(y_test)
#y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_features_df['Mode'].unique(), yticklabels=train_features_df['Mode'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
