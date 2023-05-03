import pandas as pd

# Read the CSV file
file_path = 'cow_data.csv'  # Replace with your file path
df = pd.read_csv(file_path, keep_default_na=True, na_values=['NA'])

# Move 'Estrus Status' column to the end
estrus_status_col = df.pop('Estrus Status')
df['Estrus Status'] = estrus_status_col

# Save the updated DataFrame to a new CSV file
output_file_path = 'updated_cow_data.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, index=False, na_rep='NA')
