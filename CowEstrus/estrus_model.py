import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

df = pd.read_csv('cow_data.csv')

# Preprocess the data
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df = df[['Temp', 'Eating', 'Standing', 'Rumin', 'Lying', 'L/S Transit', 'Estrus Status']]

# Convert columns to appropriate data types
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the mean of the respective column
df = df.fillna(df.mean())

scaler = MinMaxScaler()
df[['Temp']] = scaler.fit_transform(df[['Temp']])
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)].drop(columns=['Estrus Status']).values
        y = data.iloc[i + seq_length]['Estrus Status']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24
X, y = create_sequences(df, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(seq_length, 6), return_sequences=True))
model.add(SimpleRNN(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Plot the training/validation loss and accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
# Save the trained model
model.save('trained_cow_model.h5')



# # Predict the Estrus status for new data
# new_data = pd.read_csv('new_cow_data.csv')  # Assuming you have a new data file named 'new_cow_data.csv'
# new_data['activity_count'] = scaler.transform(new_data[['activity_count']])
# new_X, _ = create_sequences(new_data, seq_length)
# predictions = model.predict(new_X)
#
# # You may need to apply a threshold to convert the predictions into binary labels, e.g., 0.5
# binary_predictions = (predictions > 0.5).astype(int)
