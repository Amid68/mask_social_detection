# src/mask_detection.py
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def build_mask_model(input_shape=(128, 128, 3), num_classes=2):
    """
    Build a mask detection model using transfer learning with VGG19.

    :param input_shape: Input image shape.
    :param num_classes: Number of classes (MASK / NO MASK).
    :return: Compiled Keras model.
    """
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_mask_model(model, train_dir, val_dir, target_size=(128, 128), batch_size=32, epochs=20):
    """
    Train the mask detection model using data augmentation.

    :param model: Keras model.
    :param train_dir: Directory with training images.
    :param val_dir: Directory with validation images.
    :param target_size: Target size for image resizing.
    :param batch_size: Batch size for training.
    :param epochs: Number of epochs to train.
    :return: Training history and the trained model.
    """
    # Data augmentation for the training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    # No augmentation for validation data – only rescaling
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )
    return history, model


def evaluate_mask_model(model, test_dir, target_size=(128, 128), batch_size=32):
    """
    Evaluate the mask detection model on the test dataset.

    :param model: Trained Keras model.
    :param test_dir: Directory with test images.
    :param target_size: Target image size.
    :param batch_size: Batch size.
    :return: Evaluation metrics.
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size
    )
    scores = model.evaluate(test_generator, steps=len(test_generator))
    return scores


def predict_mask(model, face_img):
    """
    Predict whether a given face image shows a mask or not.

    :param model: Trained Keras model.
    :param face_img: Face image (must be resized to model input dimensions).
    :return: Index of predicted class (0 for MASK, 1 for NO MASK).
    """
    # Normalize and reshape the image for prediction
    img = np.array(face_img, dtype="float32") / 255.0
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    prediction = model.predict(img)
    return prediction.argmax()


def load_trained_model(model_path):
    """
    Load a trained mask detection model from disk.

    :param model_path: Path to the saved model file (e.g., 'models/masknet.h5').
    :return: Loaded Keras model.
    """
    return load_model(model_path)
