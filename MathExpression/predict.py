import numpy as np
from keras.models import load_model
import keras.utils as image


# Load the saved model
model = load_model('handwritten_math_symbols_model.h5')

# Define a dictionary of class labels
class_labels = {

0: '!', 1: '(', 2: ')', 3: '+', 4: ',', 5: '-', 6: '0', 7: '1', 8: '2', 9: '3', 10: '4', 11: '5', 12: '6', 13: '7', 14: '8', 15: '9', 16: '=', 17: 'A', 18: 'C', 19: 'Delta', 20: 'G', 21: 'H', 22: 'M', 23: 'N', 24: 'R', 25: 'S', 26: 'T', 27: 'X', 28: '[', 29: ']', 30: 'alpha', 31: 'ascii_124', 32: 'b', 33: 'beta', 34: 'cos', 35: 'd', 36: 'div', 37: 'e', 38: 'exists', 39: 'f', 40: 'forall', 41: 'forward_slash', 42: 'gamma', 43: 'geq', 44: 'gt', 45: 'i', 46: 'in', 47: 'infty', 48: 'int', 49: 'j', 50: 'k', 51: 'l', 52: 'lambda', 53: 'ldots', 54: 'leq', 55: 'lim', 56: 'log', 57: 'lt', 58: 'mu', 59: 'neq', 60: 'o', 61: 'p', 62: 'phi', 63: 'pi', 64: 'pm', 65: 'prime', 66: 'q', 67: 'rightarrow', 68: 'sigma', 69: 'sin', 70: 'sqrt', 71: 'sum', 72: 'tan', 73: 'theta', 74: 'times', 75: 'u', 76: 'v', 77: 'w', 78: 'y', 79: 'z', 80: '{', 81: '}'
    #'!': 0, '(': 1, ')': 2, '+': 3, ',': 4, '-': 5, '0': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '9': 15, '=': 16, 'A': 17, 'C': 18, 'Delta': 19, 'G': 20, 'H': 21, 'M': 22, 'N': 23, 'R': 24, 'S': 25, 'T': 26, 'X': 27, '[': 28, ']': 29, 'alpha': 30, 'ascii_124': 31, 'b': 32, 'beta': 33, 'cos': 34, 'd': 35, 'div': 36, 'e': 37, 'exists': 38, 'f': 39, 'forall': 40, 'forward_slash': 41, 'gamma': 42, 'geq': 43, 'gt': 44, 'i': 45, 'in': 46, 'infty': 47, 'int': 48, 'j': 49, 'k': 50, 'l': 51, 'lambda': 52, 'ldots': 53, 'leq': 54, 'lim': 55, 'log': 56, 'lt': 57, 'mu': 58, 'neq': 59, 'o': 60, 'p': 61, 'phi': 62, 'pi': 63, 'pm': 64, 'prime': 65, 'q': 66, 'rightarrow': 67, 'sigma': 68, 'sin': 69, 'sqrt': 70, 'sum': 71, 'tan': 72, 'theta': 73, 'times': 74, 'u': 75, 'v': 76, 'w': 77, 'y': 78, 'z': 79, '{': 80, '}': 81
    # 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    # 10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')', 16: '[', 17: ']', 18: '{', 19: '}',
    # 20: 'a', 21: 'alpha', 22: 'b', 23: 'beta', 24: 'c', 25: 'cos', 26: 'd', 27: 'delta', 28: 'div',
    # 29: 'e', 30: 'exists', 31: 'f', 32: 'forall', 33: 'forward_slash', 34: 'g', 35: 'gamma', 36: 'geq',
    # 37: 'h', 38: 'i', 39: 'in', 40: 'infty', 41: 'int', 42: 'j', 43: 'k', 44: 'l', 45: 'lambda',
    # 46: 'leq', 47: 'lim', 48: 'log', 49: 'lt', 50: 'm', 51: 'mu', 52: 'n', 53: 'neq', 54: 'o',
    # 55: 'p', 56: 'phi', 57: 'pi', 58: 'pm', 59: 'q', 60: 'r', 61: 'rightarrow', 62: 's', 63: 'sigma',
    # 64: 'sin', 65: 'sqrt', 66: 'sum', 67: 't', 68: 'tan', 69: 'theta', 70: 'times', 71: 'u', 72: 'v',
    # 73: 'w', 74: 'x', 75: 'xi', 76: 'y', 77: 'z', 78: '{', 79: '|', 80: '}', 81: '~'
}

# Load the test image and preprocess it
img = image.load_img('canvas_cropped.png', target_size=(45, 45), color_mode="grayscale")
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Make a prediction using the model
prediction = model.predict(x)

# Get the predicted label index
predicted_label_index = np.argmax(prediction)

# Get the predicted class label from the dictionary
predicted_label = class_labels[predicted_label_index]

# Print the predicted class label
print('The predicted label is:', predicted_label)
