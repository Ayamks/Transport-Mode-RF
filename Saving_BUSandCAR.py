import pandas as pd

# Load the pickle file
input_pickle_path = '004_Preprocessed_Data.pickle'
df = pd.read_pickle(input_pickle_path)

# Display the first few rows and columns of the dataframe to understand its structure
print("Columns in DataFrame:", df.columns)
print("First few rows of the DataFrame:", df.head())

# Filter the DataFrame to keep only the trips with 'bus' or 'car' mode
filtered_df = df[df['Mode'].isin(['bus', 'car'])]

# Save the filtered DataFrame to a new pickle file
output_pickle_path = 'filtered_bus_car_trips.pickle'
filtered_df.to_pickle(output_pickle_path)

print(f"Filtered DataFrame saved to {output_pickle_path}")


