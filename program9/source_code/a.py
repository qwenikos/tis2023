import os
import datetime

script_filename = os.path.splitext(os.path.basename(__file__))[0]
current_datetime = datetime.datetime.now()
date_time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"{script_filename}_{date_time_str}.txt"

# Data to save to the file (replace with your data)
data_to_save = "This is the data to save in the file."

# Define the path where the file will be saved (adjust as needed)
output_path = os.path.join(os.getcwd(), output_filename)

# Save the data to the file
with open(output_path, 'w') as file:
    file.write(data_to_save)

print(f"Data saved to {output_path}")