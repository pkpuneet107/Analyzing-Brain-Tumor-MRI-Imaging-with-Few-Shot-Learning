import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

RAW_TRAIN_DIR = 'data/Brain-Tumor-MRI-Dataset/Training'
RAW_TEST_DIR = 'data/Brain-Tumor-MRI-Dataset/Testing'
OUTPUT_DIR = 'preprocessed'
IMG_SIZE = 256

classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / np.std(img)
    img = np.stack([img] * 3, axis=-1)
    return img

def save_images(image_paths, subset_name):
    for img_path in tqdm(image_paths, desc=f"Processing {subset_name}"):
        label = os.path.basename(os.path.dirname(img_path))
        img = preprocess_image(img_path)
        out_dir = os.path.join(OUTPUT_DIR, subset_name, label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, (img * 255).astype(np.uint8))

def main():
    for split_name, raw_dir in [('train', RAW_TRAIN_DIR), ('test', RAW_TEST_DIR)]:
        for cls in classes:
            class_dir = os.path.join(raw_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            image_paths = glob(os.path.join(class_dir, '*.jpg'))
            save_images(image_paths, split_name)
    print("Done preprocessing and saving split data.")

if __name__ == "__main__":
    main()
