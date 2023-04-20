import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def predict_estrus_status(model_path, new_data_path):
    # Load the saved model
    model = load_model(model_path)

    # Load the new data
    new_data = pd.read_csv(new_data_path)

    # Preprocess the new data
    new_data['DATE'] = pd.to_datetime(new_data['DATE'])
    new_data = new_data.set_index('DATE')
    new_data = new_data[['Temp', 'Eating', 'Standing', 'Rumin', 'Lying', 'L/S Transit']]

    # Convert columns to appropriate data types
    new_data = new_data.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with the mean of the respective column
    new_data = new_data.fillna(new_data.mean())

    scaler = MinMaxScaler()
    new_data[['Temp']] = scaler.fit_transform(new_data[['Temp']])

    # Prepare the new data for prediction with a lookback of 24 hours
    lookback = 24
    X_new = []
    for i in range(lookback, len(new_data)):
        X_new.append(new_data.values[i - lookback:i])

    X_new = np.array(X_new)
    X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], X_new.shape[2]))

    # Make predictions
    predictions = model.predict(X_new)
    predictions = (predictions > 0.5).astype(int).flatten()

    return predictions



# create a main function
def main():
    new_data_path = "new_cow_data.csv"
    model_path = "trained_cow_model.h5"
    predictions = predict_estrus_status(model_path, new_data_path)
    print("Predicted estrus status:", predictions)


if __name__ == "__main__":
    main()


