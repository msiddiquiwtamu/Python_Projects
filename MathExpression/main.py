import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Set path to the directory containing the training data
train_dir = '/Users/mohammadfaridulhaquesiddiqui/Downloads/extracted_images'

# Define image dimensions and number of classes
img_width, img_height = 45, 45
num_classes = 82

# Set hyperparameters for the model
batch_size = 32
epochs = 20
learning_rate = 0.0001

# Create an instance of the ImageDataGenerator class
train_data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generate training data from images in the directory
train_data = train_data_generator.flow_from_directory(
    directory=train_dir,
    target_size=(img_width, img_height),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='training'
)

# Generate validation data from images in the directory
val_data = train_data_generator.flow_from_directory(
    directory=train_dir,
    target_size=(img_width, img_height),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='validation'
)

# Define a dictionary of class labels
class_labels = train_data.class_indices
print(class_labels)

# Define the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size
)

# Save the model
model.save('handwritten_math_symbols_model.h5')
