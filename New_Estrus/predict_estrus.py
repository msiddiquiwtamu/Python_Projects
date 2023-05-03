import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
# Load the saved model
loaded_model = joblib.load('trained_model.pkl')

# Load the unseen data
data = pd.read_csv('updated_new_cow_data.csv')

# Preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])
data['START'] = pd.to_datetime(data['START'])
data['END'] = pd.to_datetime(data['END'])

data = data.drop(['START', 'END', 'DURATION'], axis=1)
# Replace 'data' with the name of your DataFrame
# column_names = data.columns
#
# print("Column names:")
# for name in column_names:
#     print(name)

# Handle missing values and data types
data.replace('NA ', np.nan, inplace=True)
numerical_columns = ['Eating', 'Standing', 'Rumin','Lying','L/S Transit','Mounted/Hr','Mounting/Hr','Estrus Status']  # replace with the actual column names
for column in numerical_columns:
    data[column] = data[column].astype(float)
data.fillna(data.mean(), inplace=True)

# Convert 'DATE' to Unix timestamp
data['DATE'] = data['DATE'].astype(np.int64) // 10**9

# One-hot encoding of categorical variables
categorical_vars = ['Cow ID', 'Time Block']
data = pd.get_dummies(data, columns=categorical_vars)

# Load the column names from the training data
train_columns = joblib.load('data_columns.pkl')

# Align unseen data's columns with the training data's columns, filling in missing columns with zeros
data = data.reindex(columns=train_columns, fill_value=0)


# Load the saved scaler
scaler = joblib.load('data_scaler.pkl')

# Scale the data
scaled_data = scaler.transform(data)
scaled_data= scaled_data[:, :-1]
# Reshape input to be 3D [samples, timesteps, features]
scaled_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

# Make predictions using the loaded model
predictions = loaded_model.predict(scaled_data)

# # Concatenate predictions with the other features
# concatenated_scaled_data = np.concatenate((scaled_data[:, :-1], predictions), axis=1)
#
# # Inverse transform the concatenated_scaled_data
# inverse_transformed_data = scaler.inverse_transform(concatenated_scaled_data)
# # Extract the predicted values
# predicted_values = inverse_transformed_data[:, -1]

print(predictions)