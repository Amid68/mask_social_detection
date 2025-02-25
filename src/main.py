# src/main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import our custom modules
from face_detection import load_face_detector, detect_faces
from social_distance import check_social_distance
from mask_detection import load_trained_model, predict_mask

# Global parameters and file paths
HAAR_CASCADE_PATH = 'data/haarcascades/haarcascade_frontalface_default.xml'
MIN_DISTANCE = 130  # Minimum allowed distance in pixels
MASK_LABELS = {0: 'MASK', 1: 'NO MASK'}
COLOR_MAPPING = {0: (0, 255, 0), 1: (255, 0, 0)}  # Green for safe, Red for violation


def main():
    # Load the Haar Cascade face detector
    face_cascade = load_face_detector(HAAR_CASCADE_PATH)

    # Load the pre-trained mask detection model (ensure this file exists in the models/ folder)
    mask_model = load_trained_model('models/masknet.h5')

    # Read the input image (change the path to your test image)
    image_path = 'data/sample_image.png'
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found. Please check the path.")
        return

    # Convert image to grayscale for face detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade
    faces = detect_faces(gray_img, face_cascade)

    # Convert original image to RGB for display (Matplotlib expects RGB)
    output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Determine social distancing violations
    violation_labels = check_social_distance(faces, min_distance=MIN_DISTANCE)

    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face from the image
        face_crop = output_img[y:y + h, x:x + w]
        # Resize the face crop to the input size expected by the mask model
        face_crop_resized = cv2.resize(face_crop, (128, 128))
        # Predict mask usage (0: MASK, 1: NO MASK)
        prediction = predict_mask(mask_model, face_crop_resized)
        label_text = MASK_LABELS[prediction]
        # Choose the bounding box color based on social distancing violation status
        color = COLOR_MAPPING[violation_labels[i]]
        # Annotate the image with the prediction label and draw the bounding box
        cv2.putText(output_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)

    # Display the final output image with annotations
    plt.figure(figsize=(10, 10))
    plt.imshow(output_img)
    plt.title("Mask and Social Distancing Detection")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
