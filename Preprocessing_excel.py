import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
print("Libraries Are Imported")


def process_file(file_path, trip_id):
    try:
        df = pd.read_csv(file_path)
        df['Trip'] = trip_id  # Assign a trip identifier based on the filename
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def preprocess_data(df):
    # Check if required columns exist
    required_columns = ['Distance', 'Speed', 'Acceleration', 'Jerk', 'Bearing']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the data.")

    # Drop duplicates based on 'Trip' if 'Trip' column exists
    if 'Trip' in df.columns:
        df.drop_duplicates(subset=['Trip'], keep='first', inplace=True)

    # Took off the following lines because they result in empty dataframes
    # Drop trips with fewer than 60 rows of data
    #df = df.groupby('Trip').filter(lambda x: len(x) > 60)

    # Apply Exponential Moving Average to Noises for Speed, Acceleration, and Jerk Features
    for feature in ['Speed', 'Acceleration', 'Jerk']:
        if feature in df.columns:
            ema_feature = df[feature].ewm(alpha=0.1, adjust=False).mean()
            mean = df[feature].mean()
            std = df[feature].std()
            df[feature] = np.where((df[feature] > mean + 3 * std) | (df[feature] < mean - 3 * std), ema_feature,
                                   df[feature])

    return df


def main(directory):
    all_data = []
    trip_id = 0
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            data = process_file(file_path, trip_id)
            trip_id += 1
            if data is not None:
                all_data.append(data)

    if not all_data:
        raise ValueError("No data was processed. Please check the input files.")

    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure 'Label' column is in the data
    if 'Label' not in combined_df.columns:
        raise ValueError("The 'Label' column is missing from the data.")

    # Convert labels to lowercase
    combined_df['Label'] = combined_df['Label'].str.lower()

    # Print unique labels before filtering
    unique_labels = combined_df['Label'].unique()
    print(f"Unique labels before filtering: {unique_labels}")

    # Filter for specific labels
    Pr_Label = ['bus', 'car', 'walk', 'bike', 'train']
    combined_df = combined_df[combined_df['Label'].isin(Pr_Label)]

    # Replace 'taxi' with 'car' and 'subway' with 'train'
    combined_df.replace({"taxi": "car", "subway": "train"}, inplace=True)

    # Drop duplicates based on 'Trip' if 'Trip' column exists
    if 'Trip' in combined_df.columns:
        combined_df.drop_duplicates(subset=['Trip'], keep='first', inplace=True)

    # Check if there are any trips in the data
    if combined_df.empty:
        raise ValueError("The combined DataFrame is empty after filtering by 'Label'.")

    unique_trips = combined_df['Trip'].unique()
    if len(unique_trips) == 0:
        raise ValueError("No unique trips found in the combined DataFrame.")

    train_trips, test_trips = train_test_split(unique_trips, test_size=0.2, random_state=42)

    train_df = combined_df[combined_df['Trip'].isin(train_trips)]
    test_df = combined_df[combined_df['Trip'].isin(test_trips)]

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    with open("Preprocessed_excel_Train_Data.pickle", 'wb') as f:
        pickle.dump(train_df, f)

    with open("Preprocessed_excel_Test_Data.pickle", 'wb') as f:
        pickle.dump(test_df, f)

    print("Data processing complete. Preprocessed data saved.")


if __name__ == "__main__":
    input_directory = 'C:/Users/AyaMEKKASS/PycharmProjects/Transport-Mode_RF/trips'  # Change this to your directory path
    main(input_directory)

