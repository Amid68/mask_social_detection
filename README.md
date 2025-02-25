# Mask and Social Distancing Detection

This project implements a computer vision system that detects whether people are wearing face masks and monitors social distancing violations using deep learning (VGG19) and OpenCV’s Haar Cascade for face detection.

## Features

- **Face Detection:** Uses Haar Cascade classifiers to locate faces in images.
- **Mask Detection:** Employs a transfer learning model based on VGG19 to classify whether a person is wearing a mask.
- **Social Distancing Monitoring:** Calculates distances between detected faces to identify potential social distancing violations.
- **Dataset Organization:** Includes scripts to automatically organize your dataset into Train, Validation, and Test splits.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
    ```
3. Organize your dataset:
   - Place your downloaded dataset (with annotations and images folders) in data/face-mask-dataset/.
   - Run python split_data.py to automatically organize the dataset into the required folder structure.
4. in the model (if not already trained):
    ```
    python src/train_model.py
    ```
5. Test the system by running:
    ```
    python src/main.py
    ```

### License

This project is released under the MIT License.