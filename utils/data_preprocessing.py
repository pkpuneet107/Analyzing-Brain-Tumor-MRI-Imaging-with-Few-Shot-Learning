import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

# === CONFIG ===
RAW_DATA_DIR = 'data/Brain-Tumor-MRI-Dataset-Merged'
OUTPUT_DIR = 'preprocessed'
IMG_SIZE = 256

# Classes in dataset
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Split by class: no overlaps
train_classes = ['glioma', 'meningioma']
val_classes = ['pituitary']
test_classes = ['notumor']

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / np.std(img)
    img = np.stack([img] * 3, axis=-1)  # 3-channel RGB
    return img

def save_images(image_paths, subset_name):
    for img_path in tqdm(image_paths, desc=f"Processing {subset_name}"):
        label = os.path.basename(os.path.dirname(img_path))
        img = preprocess_image(img_path)
        out_dir = os.path.join(OUTPUT_DIR, subset_name, label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, (img * 255).astype(np.uint8))  # Save as uint8

def main():
    class_to_paths = {cls: [] for cls in classes}
    
    # Collect all image paths
    for cls in classes:
        class_dir = os.path.join(RAW_DATA_DIR, cls)
        if not os.path.isdir(class_dir):
            continue
        image_paths = glob(os.path.join(class_dir, '*.jpg'))
        class_to_paths[cls].extend(image_paths)

    # Process by split
    for cls in train_classes:
        save_images(class_to_paths[cls], 'train')
    for cls in val_classes:
        save_images(class_to_paths[cls], 'val')
    for cls in test_classes:
        save_images(class_to_paths[cls], 'test')

    print("✅ Done preprocessing and saving split data.")

if __name__ == "__main__":
    main()
