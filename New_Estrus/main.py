import pandas as pd

# Read the dataset (replace 'your_dataset.csv' with the path to your dataset)
data = pd.read_csv('cow_data.csv')

# Move the 'Estrus Status' column to the last position
cols = data.columns.tolist()
cols.remove('Estrus Status')
cols.append('Estrus Status')
data = data[cols]

# Save the modified dataset (you can change 'modified_dataset.csv' to your desired output file name)
data.to_csv('modified_dataset.csv', index=False)

print("Dataset saved with 'Estrus Status' as the last column.")
