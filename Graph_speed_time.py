import pandas as pd
import matplotlib.pyplot as plt

# Load a single trip CSV file
# Replace 'trip_file.csv' with the actual filename
trip_df = pd.read_csv('trips/commuting_2024_05_07_to_work_single_mode_1_car.csv')

# Plotting the speed over the trip index (assuming the time series is sequential)
plt.figure(figsize=(12, 6))
plt.plot(trip_df.index, trip_df['Speed'], label='Speed')
plt.xlabel('Index (or Time if available)')
plt.ylabel('Speed (m/s)')
plt.title('Speed Over Time for a Single Trip')
plt.legend()
plt.grid(True)
plt.show()
