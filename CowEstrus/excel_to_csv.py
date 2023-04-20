import pandas as pd

def xlsx_to_csv(input_file_path, output_file_path):
    # Read the Excel file
    df = pd.read_excel(input_file_path)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file_path, index=False)

# Example usage
input_file_path = 'cow_data.xlsx'
output_file_path = 'cow_data.csv'
xlsx_to_csv(input_file_path, output_file_path)
