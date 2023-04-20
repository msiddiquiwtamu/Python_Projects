import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Preprocessing function to prepare the input data
def preprocess_data(data):
    # Convert START and END columns to datetime objects
    data["START"] = pd.to_datetime(data["START"])
    data["END"] = pd.to_datetime(data["END"])

    # Calculate the time difference between START and END columns in minutes
    data["duration_minutes"] = (data["END"] - data["START"]).dt.total_seconds() / 60

    # Drop the unnecessary columns
    data = data.drop(["DATE", "START", "END"], axis=1)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler



# Read the data from a CSV file
data = pd.read_csv("cow_data.csv")
# Replace 'NA ' with NaN values
data = data.replace('NA ', float('nan'))
# Select numeric columns to check for infinite values
numeric_columns = data.select_dtypes(include=[np.number]).columns
# Print the number of missing values
print("Missing values:", data.isnull().sum().sum())
# Print the number of infinite values
print("Infinite values:", data[numeric_columns].applymap(np.isinf).sum().sum())
# Drop rows with missing values
data = data.dropna()
for col in data.columns:
    if data[col].dtype == object:
        try:
            data[col] = pd.to_timedelta(data[col], errors='coerce').dt.total_seconds() / 60
        except ValueError:
            pass

# Preprocess the data
data_scaled, scaler = preprocess_data(data)

# Preprocess the data
data_scaled, scaler = preprocess_data(data)

# Function to create input data for RNN
def create_rnn_data(data, sequence_length=50, target_column=11):
    x = []
    y = []

    # Create sequences of length 'sequence_length' and corresponding targets
    for i in range(sequence_length, len(data)):
        x.append(data[i - sequence_length:i])
        y.append(data[i][target_column])

    return np.array(x), np.array(y)

# Function to prepare data for RNN training and testing
def prepare_data(data, window_size, test_split=0.2):
    # Create input data for RNN
    X, y = create_rnn_data(data_scaled.tolist(), sequence_length=sequence_length)

    # Split the data into training and test sets
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Flatten the 3D array to a 2D array before scaling
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    # Perform scaling on the 2D array
    scaler = MinMaxScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # Reshape the scaled 2D array back to the original 3D shape
    X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Set sequence_length and prepare data
sequence_length = 50
X_train, X_test, y_train, y_test = prepare_data(data, sequence_length)

# Function to create RNN model
def create_model(input_shape):
    # Create a Sequential model
    model = Sequential()
    # Add LSTM layers with 64 and 32 units, respectively
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32))
    # Add a Dense output layer with a sigmoid activation function
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with the Adam optimizer and binary_crossentropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Get the input shape for the model
input_shape = (sequence_length, X_train.shape[2])
# Create the RNN model
model = create_model(input_shape)

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with the training data, validation split, and early stopping
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# Make predictions using the test data
predictions = model.predict(X_test)

# Evaluate the model on the test set and print the test loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

