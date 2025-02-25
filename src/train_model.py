# train_model.py
import os
from src.mask_detection import build_mask_model, train_mask_model

# Update these paths to point to your organized dataset directories.
train_dir = '../data/face-mask-dataset/Face Mask Dataset/Train'
val_dir = '../data/face-mask-dataset/Face Mask Dataset/Validation'

# Build the model
model = build_mask_model()

# Train the model (adjust epochs as needed)
history, model = train_mask_model(model, train_dir, val_dir, epochs=10)

# Save the trained model to models/masknet.keras
os.makedirs('../models', exist_ok=True)
model.save('../models/masknet.keras')
print("Model trained and saved as models/masknet.keras")
