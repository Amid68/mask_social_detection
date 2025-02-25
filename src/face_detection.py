# src/face_detection.py
import cv2
import os
import urllib.request


def load_face_detector(model_path):
    """
    Load the Haar Cascade classifier for face detection.

    :param model_path: Path to the Haar Cascade XML file.
    :return: cv2.CascadeClassifier object.
    """
    # Ensure the Haar Cascade file exists
    if not os.path.isfile(model_path):
        print(f"Error: Haar Cascade file not found at {model_path}. Attempting to download...")
        # Download the Haar Cascade file if it does not exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
        url = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml?raw=true"
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Downloaded the Haar Cascade file to {model_path}.")
        except Exception as e:
            raise IOError(f"Failed to download Haar Cascade file: {e}")

    # Load the Haar Cascade
    face_cascade = cv2.CascadeClassifier(model_path)
    if face_cascade.empty():
        raise IOError(f"Unable to load the face cascade classifier. Check if the file is corrupted: {model_path}")
    return face_cascade



def detect_faces(image, face_cascade, scaleFactor=1.1, minNeighbors=4):
    """
    Detect faces in the provided grayscale image.

    :param image: Grayscale image in which to detect faces.
    :param face_cascade: Haar Cascade classifier.
    :param scaleFactor: Image scaling factor for multi-scale detection.
    :param minNeighbors: Minimum number of neighbor rectangles for a detection to be retained.
    :return: List of bounding boxes for detected faces [(x, y, w, h), ...].
    """
    faces = face_cascade.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return faces
