# src/social_distance.py
from scipy.spatial import distance


def check_social_distance(faces, min_distance=130):
    """
    Check for social distancing violations by computing the Euclidean distance between face positions.

    :param faces: List of bounding boxes for detected faces [(x, y, w, h), ...].
    :param min_distance: Minimum allowable distance (in pixels) between faces.
    :return: List of labels for each face (0 for safe, 1 for violation).
    """
    violation_labels = [0] * len(faces)
    if len(faces) < 2:
        return violation_labels  # No violation possible if less than 2 faces

    # Compare each pair of faces
    for i in range(len(faces) - 1):
        for j in range(i + 1, len(faces)):
            # Calculate distance between the top-left corners of the bounding boxes
            dist = distance.euclidean(faces[i][:2], faces[j][:2])
            if dist < min_distance:
                violation_labels[i] = 1
                violation_labels[j] = 1
    return violation_labels
