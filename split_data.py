#!/usr/bin/env python
"""
split_data.py

This script organizes the downloaded face mask detection dataset.

The original dataset folder (data/face-mask-dataset) contains:
  - annotations/    (XML files with annotations in Pascal VOC format)
  - images/         (corresponding image files)

This script will:
  1. Parse each XML file to extract the label.
  2. Map the label to a standardized class: 'WithMask' or 'WithoutMask'.
  3. Split the data into Train, Validation, and Test sets.
  4. Copy images into the folder structure:

     data/face-mask-dataset/Face Mask Dataset/
         ├── Train/
         │     ├── WithMask/
         │     └── WithoutMask/
         ├── Validation/
         │     ├── WithMask/
         │     └── WithoutMask/
         └── Test/
               ├── WithMask/
               └── WithoutMask/

Adjust the label mapping logic if your annotation labels differ.
"""

import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


def parse_annotation(xml_path):
    """
    Parse a Pascal VOC-style XML file to extract the object's label.
    Assumes a single object per image.

    :param xml_path: Path to the XML file.
    :return: The label found in the XML, or None if not found.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find('object')
    if obj is not None:
        label = obj.find('name').text
        return label
    return None


def map_label(label):
    """
    Map the raw label from the XML to our standardized folder names.

    :param label: The label extracted from the annotation.
    :return: 'WithMask' or 'WithoutMask'
    """
    label_lower = label.lower()
    # Adjust these conditions based on your actual XML labels.
    if 'mask' in label_lower and ('without' not in label_lower and 'no' not in label_lower):
        return 'WithMask'
    elif 'mask' in label_lower and ('without' in label_lower or 'no' in label_lower):
        return 'WithoutMask'
    else:
        # Fallback: return the original label capitalized.
        return label.capitalize()


def organize_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organize images from the source dataset into Train, Validation, and Test splits.

    :param source_dir: Directory containing 'annotations' and 'images' folders.
    :param dest_dir: Destination directory for the organized dataset.
    :param train_ratio: Fraction of data for training.
    :param val_ratio: Fraction of data for validation.
    :param test_ratio: Fraction of data for testing.
    """
    annotations_dir = os.path.join(source_dir, 'annotations')
    images_dir = os.path.join(source_dir, 'images')

    # List to store tuples: (image_filename, mapped_label)
    data_list = []

    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            raw_label = parse_annotation(xml_path)
            if raw_label is None:
                continue
            mapped_label = map_label(raw_label)
            # Assume image file has same base name with .png extension
            base_name = os.path.splitext(xml_file)[0]
            image_filename = base_name + '.png'
            image_path = os.path.join(images_dir, image_filename)
            if os.path.exists(image_path):
                data_list.append((image_filename, mapped_label))
            else:
                print(f"Warning: Image '{image_filename}' not found for annotation '{xml_file}'.")

    print(f"Total images found: {len(data_list)}")

    # Create destination folder structure
    splits = ['Train', 'Validation', 'Test']
    classes = ['WithMask', 'WithoutMask']
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
            print(f"Created directory: {os.path.join(dest_dir, split, cls)}")

    # Prepare lists for stratified splitting
    filenames = [item[0] for item in data_list]
    labels = [item[1] for item in data_list]

    # Split into training and temporary set
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        filenames, labels, stratify=labels, test_size=(1 - train_ratio), random_state=42)

    # Further split temp set into validation and test sets
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, stratify=temp_labels, test_size=(1 - val_ratio_adjusted), random_state=42)

    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

    # Helper function to copy files into designated directories
    def copy_files(file_list, label_list, split):
        for file, label in zip(file_list, label_list):
            src_path = os.path.join(images_dir, file)
            dst_path = os.path.join(dest_dir, split, label, file)
            shutil.copy(src_path, dst_path)
            # Uncomment below to print each file copy if needed
            # print(f"Copied {src_path} to {dst_path}")

    copy_files(train_files, train_labels, 'Train')
    copy_files(val_files, val_labels, 'Validation')
    copy_files(test_files, test_labels, 'Test')

    print("Dataset organization complete.")


if __name__ == "__main__":
    # The source dataset directory is assumed to be: data/face-mask-dataset/
    # It should contain the 'annotations' and 'images' folders.
    source_dataset_dir = os.path.join("data", "face-mask-dataset")
    # Destination directory for the organized dataset:
    # data/face-mask-dataset/Face Mask Dataset/
    dest_dataset_dir = os.path.join(source_dataset_dir, "Face Mask Dataset")

    organize_dataset(source_dataset_dir, dest_dataset_dir)
