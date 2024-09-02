# Import Libraries
import os
import numpy as np
import pandas as pd
import pickle
import math
from geopy.distance import distance
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
print("Libraries Are Imported")

# Read The Labeled Data
Data = '003_Labeled_GeoLife.pickle'
with open(Data, 'rb') as infile:
    df = pickle.load(infile)

# Combine all user data into a single DataFrame
dfs = [pd.DataFrame(value) for value in df.values()]
DF = pd.concat(dfs, axis=0, ignore_index=True)

# Select Just These Modes for the Project
Pr_Mode = ['bus', 'car', 'walk', 'bike', 'taxi', 'train', 'subway']
DF = DF[DF['Mode'].isin(Pr_Mode)]

# Replace 'taxi' with 'car' and 'subway' with 'train'
DF.replace({"taxi": "car", "subway": "train"}, inplace=True)

# Drop duplicates based on 'Trip' and 'TS' columns
DF.drop_duplicates(subset=['Trip', 'TS'], keep='first', inplace=True)

# Reset index
DF.reset_index(drop=True, inplace=True)

# Ensure each trip is assigned to either the training or test set
unique_trips = DF['Trip'].unique()
train_trips, test_trips = train_test_split(unique_trips, test_size=0.2, random_state=42)

# Convert train_trips and test_trips to sets for disjoint check
train_trip_set = set(train_trips)
test_trip_set = set(test_trips)

# Ensure trips are not split between training and test sets
assert train_trip_set.isdisjoint(test_trip_set), "Some trips are present in both training and test sets!"

# Create training and test DataFrames
train_df = DF[DF['Trip'].isin(train_trips)]
test_df = DF[DF['Trip'].isin(test_trips)]

# Function to preprocess the data
def preprocess_data(df):
    # Calculate Distance between GPS points
    Dist = [0] + [distance((df['Latitude'].iloc[i], df['Longitude'].iloc[i]),
                           (df['Latitude'].iloc[i + 1], df['Longitude'].iloc[i + 1])).meters
                  for i in range(len(df) - 1)]
    df['Distance'] = Dist

    # Reset the Distance at the start of each trip to zero
    trip_change_indices = df.index[df['Trip'].shift() != df['Trip']]
    df.loc[trip_change_indices, 'Distance'] = 0

    # Calculate Delta Time
    df['DT'] = df['TS'].diff().fillna(0)
    df.loc[trip_change_indices, 'DT'] = 0

    # Compute Speed
    df['Speed'] = df['Distance'] / df['DT'].replace(0, np.nan)
    df['Speed'] = df['Speed'].fillna(0)
    df.loc[trip_change_indices, 'Speed'] = 0

    # Compute Acceleration
    df['Acc'] = df['Speed'].diff().fillna(0) / df['DT'].replace(0, np.nan)
    df['Acc'] = df['Acc'].fillna(0)
    df.loc[trip_change_indices, 'Acc'] = 0

    # Compute Jerk
    df['Jerk'] = df['Acc'].diff().fillna(0) / df['DT'].replace(0, np.nan)
    df['Jerk'] = df['Jerk'].fillna(0)
    df.loc[trip_change_indices, 'Jerk'] = 0

    # Change Latitude, Longitude, and Altitude from String to Number
    df['Latitude'] = pd.to_numeric(df['Latitude'])
    df['Longitude'] = pd.to_numeric(df['Longitude'])
    df['Altitude'] = pd.to_numeric(df['Altitude'])

    # Calculate Bearing
    bearing = [0] + [
        (math.atan2(
            math.sin(math.radians(df['Longitude'].iloc[i + 1]) - math.radians(df['Longitude'].iloc[i])) * math.cos(math.radians(df['Latitude'].iloc[i + 1])),
            math.cos(math.radians(df['Latitude'].iloc[i])) * math.sin(math.radians(df['Latitude'].iloc[i + 1])) -
            math.sin(math.radians(df['Latitude'].iloc[i])) * math.cos(math.radians(df['Latitude'].iloc[i + 1])) *
            math.cos(math.radians(df['Longitude'].iloc[i + 1]) - math.radians(df['Longitude'].iloc[i]))
        ) * 180.0 / math.pi + 360) % 360 for i in range(len(df) - 1)
    ]
    df['Bearing'] = bearing

    # Set Bearing to 0 where Trip changes
    df.loc[trip_change_indices, 'Bearing'] = 0

    # Extract Cumulative Distance for each trip
    df['Cum_Distance'] = df.groupby('Trip')['Distance'].cumsum()

    # Apply Exponential Moving Average to Noises for Speed, Acceleration, and Jerk Features
    for feature in ['Speed', 'Acc', 'Jerk']:
        ema_feature = df[feature].ewm(alpha=0.1, adjust=False).mean()
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = np.where((df[feature] > mean + 3 * std) | (df[feature] < mean - 3 * std), ema_feature, df[feature])

    # Drop Trips with Length Smaller than 60 rows of GPS Data
    df = df.groupby('Trip').filter(lambda x: len(x) > 60)

    return df

# Preprocess the training and test data separately
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Save the preprocessed training and test data
with open("004_Preprocessed_Train_Data.pickle", 'wb') as f:
    pickle.dump(train_df, f)

with open("004_Preprocessed_Test_Data.pickle", 'wb') as f:
    pickle.dump(test_df, f)


