# src/face_detection.py
import cv2


def load_face_detector(model_path):
    """
    Load the Haar Cascade classifier for face detection.

    :param model_path: Path to the Haar Cascade XML file.
    :return: cv2.CascadeClassifier object.
    """
    face_cascade = cv2.CascadeClassifier(model_path)
    if face_cascade.empty():
        raise IOError("Unable to load the face cascade classifier. Check the model path.")
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
