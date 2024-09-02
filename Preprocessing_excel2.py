import os
import pandas as pd
import numpy as np


def process_trip(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Calculate the required summary statistics for the trip
    summary = {
        'Speed': df['Speed'].mean(),
        'Distance': df['Distance'].mean(),
        'Mode': df['Label'].mode()[0],  # Assuming 'Label' is the mode of transport
        'Acc': df['Acceleration'].mean(),
        'Jerk': df['Jerk'].mean(),
        'Bearing': df['Bearing'].mean(),
        '85thV': df['Speed'].quantile(0.85),
        '85thA': df['Acceleration'].quantile(0.85),
        'MaxV1': df['Speed'].max(),
        'MaxA1': df['Acceleration'].max(),
        'MaxV2': df['Speed'].nlargest(2).iloc[-1],
        'MaxA2': df['Acceleration'].nlargest(2).iloc[-1],
        'MedianV': df['Speed'].median(),
        'MedianA': df['Acceleration'].median(),
        'MinV': df['Speed'].min(),
        'MinA': df['Acceleration'].min(),
        'MeanV': df['Speed'].mean(),
        'MeanA': df['Acceleration'].mean(),
        'ExpV': df['Speed'].mean(),  # Normally, ExpV might refer to exponential smoothing; this uses mean
        'ExpA': df['Acceleration'].mean(),
        'CovV': df['Speed'].cov(df['Speed']),
        'CovA': df['Acceleration'].cov(df['Acceleration']),
        'VCR': df['Speed'].diff().mean(),
        'ACR': df['Acceleration'].diff().mean(),
        'HVR': (df['Speed'] > df['Speed'].quantile(0.75)).mean(),
        'MVR': ((df['Speed'] > df['Speed'].quantile(0.25)) & (df['Speed'] <= df['Speed'].quantile(0.75))).mean(),
        'LVR': (df['Speed'] <= df['Speed'].quantile(0.25)).mean(),
        'HAR': (df['Acceleration'] > df['Acceleration'].quantile(0.75)).mean(),
        'MAR': ((df['Acceleration'] > df['Acceleration'].quantile(0.25)) & (
                    df['Acceleration'] <= df['Acceleration'].quantile(0.75))).mean(),
        'LAR': (df['Acceleration'] <= df['Acceleration'].quantile(0.25)).mean(),
        'BSR': (df['Speed'] == 0).mean(),
        'HCR': df['Bearing'].diff().mean(),
        'SR': (df['Speed'] == 0).mean()
    }

    return summary


def process_all_trips(directory):
    summary_list = []

    # Loop through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            trip_summary = process_trip(file_path)
            trip_summary['Trip'] = filename  # Optionally add the filename as an identifier
            summary_list.append(trip_summary)

    # Convert the list of summaries into a DataFrame
    summary_df = pd.DataFrame(summary_list)

    return summary_df

def save_to_pickle(dataframe, output_path):
    dataframe.to_pickle(output_path)


if __name__ == "__main__":
    # Specify the directory containing the trip CSV files
    directory = 'trips'

    # Process all trips
    summary_df = process_all_trips(directory)

    # Save the DataFrame to a pickle file
    output_path = 'trips_summary.pickle'
    save_to_pickle(summary_df, output_path)

    print(f"Processed {len(summary_df)} trips and saved the summary to {output_path}")

