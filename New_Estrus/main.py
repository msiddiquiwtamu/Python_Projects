import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Read data
data = pd.read_csv('updated_cow_data.csv')

# Preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])
data['START'] = pd.to_datetime(data['START'])
data['END'] = pd.to_datetime(data['END'])

data = data.drop(['START', 'END', 'DURATION'], axis=1)

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

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)



# Train-test split
X = scaled_data[:, :-1]
y = scaled_data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False)

# Evaluate the model
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(np.concatenate((X_test[:, :, :].reshape(X_test.shape[0], -1), y_test.reshape(-1, 1)), axis=1))[:, -1]
y_pred_inv = scaler.inverse_transform(np.concatenate((X_test[:, :, :].reshape(X_test.shape[0], -1), y_pred.reshape(-1, 1)), axis=1))[:, -1]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print('Test RMSE: %.3f' % rmse)
