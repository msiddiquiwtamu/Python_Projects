from keras.models import load_model

saved_model = load_model("best_model.hdf5")

# Load and preprocess the unseen data (similar to how the training data was preprocessed)
unseen_data = pd.read_csv("new_cow_data.csv")
scaled_unseen_data = scaler.transform(unseen_data)

predictions = saved_model.predict(scaled_unseen_data)
