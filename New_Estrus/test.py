# For the training dataset
import pandas as pd
train_data = pd.read_csv('updated_cow_data.csv')  # Replace 'training_data.csv' with the name of your training data file
print("Number of columns in the training dataset:", len(train_data.columns))

# For the unseen dataset
test_data = pd.read_csv('updated_new_cow_data.csv')
print("Number of columns in the unseen dataset:", len(test_data.columns))
