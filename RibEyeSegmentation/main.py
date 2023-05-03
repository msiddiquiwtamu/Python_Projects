import cv2
import numpy as np

def segment_image(image_path, threshold_value):
    # Read the image
    image = cv2.imread('ribeye.jpg')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded_image)

    # Create a blank image for color segmentation
    color_segmented_image = np.zeros_like(image)

    # Assign random colors to each region
    for label in range(1, num_labels):
        color = np.random.randint(0, 255, size=3)  # Generate a random color
        color_segmented_image[labels == label] = color

    # Display the segmented image
    cv2.imshow("Segmented Image", color_segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image you want to segment
image_path = "path_to_your_image.jpg"

# Threshold value for segmentation (adjust as needed)
threshold_value = 128

# Call the segmentation function
segment_image(image_path, threshold_value)
