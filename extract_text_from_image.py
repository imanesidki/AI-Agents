import cv2
import os
import pytesseract
import numpy as np

# Function to preprocess and extract text from image (for images with '_orange' suffix)
def extract_text_from_image_proc(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    alpha = 2.5  # Fine-tuned contrast
    beta = -100  # Fine-tuned brightness
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(adjusted, None, 30, 7, 21)

    # Apply a threshold to create a binary image
    _, binary = cv2.threshold(denoised, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the image (black text on white background)
    inverted = cv2.bitwise_not(binary)

    # Apply morphological operations to reduce noise
    kernel = np.ones((1, 1), np.uint8)  # Smaller kernel size
    morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    # Invert again to get black text on white background
    final_image = cv2.bitwise_not(morph)

    # Use pytesseract to extract text from the preprocessed image
    extracted_text = pytesseract.image_to_string(final_image, lang='fra', config='--oem 3 --psm 1')

    return extracted_text

# Function to extract text from image (for images without '_orange' suffix)
# def extract_text_from_image(image_path):
#     # Load the image using OpenCV
#     image = cv2.imread(image_path)
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Extract text using pytesseract
#     extracted_text = pytesseract.image_to_string(image, config='--oem 3 --psm 1')

#     return extracted_text

# extract_text_from_image_proc(image_path)
