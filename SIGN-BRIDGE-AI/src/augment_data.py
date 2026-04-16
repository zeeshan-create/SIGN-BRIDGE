import cv2
import os
import numpy as np
import albumentations as A
import random

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
OUTPUT_DIR = INPUT_DIR
SAMPLES_TO_GENERATE = 50 

# Augmentation Pipeline
transform = A.Compose([
    A.Rotate(limit=25, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
])

def augment_category(category_name, base_dir=INPUT_DIR):
    cat_path = os.path.join(base_dir, category_name)
    if not os.path.exists(cat_path):
        print(f"Error: Path {cat_path} does not exist.")
        return

    # Relaxed filtering: any jpg/png that isn't already an augmented version
    user_images = [f for f in os.listdir(cat_path) if ('jpg' in f or 'png' in f) and '_aug_' not in f]
    
    if not user_images:
        print(f"No original images found in {category_name}.")
        return

    print(f"Augmenting {len(user_images)} images for category '{category_name}'...")

    for img_name in user_images:
        img_path = os.path.join(cat_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        for i in range(SAMPLES_TO_GENERATE):
            augmented = transform(image=image)['image']
            
            # Save augmented image
            base_name = os.path.splitext(img_name)[0]
            aug_name = f"{base_name}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(cat_path, aug_name), augmented)

    print(f"Successfully generated {len(user_images) * SAMPLES_TO_GENERATE} augmented samples for '{category_name}'.")

import sys

def main():
    # Check for --all flag
    do_all = "--all" in sys.argv
    
    if do_all:
        print("[Augment] Mass-Augmentation mode active. Processing ALL categories...")
        # Check both data and external_data
        search_dirs = [INPUT_DIR, os.path.join(os.path.dirname(BASE_DIR), 'external_data')]
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            for category in os.listdir(s_dir):
                cat_path = os.path.join(s_dir, category)
                if os.path.isdir(cat_path):
                    augment_category(category, base_dir=s_dir)
    else:
        # Standard mode
        categories = ['hello', 'I_Love_You', 'thanks', 'no', 'HELLO', 'Y', 'LOT']
        for category in categories:
            if os.path.exists(os.path.join(INPUT_DIR, category)):
                augment_category(category)

if __name__ == "__main__":
    main()
