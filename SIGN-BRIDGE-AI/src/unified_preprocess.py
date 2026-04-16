"""
ANTIGRAVITY ASL — Unified Preprocessing Pipeline
=================================================
Scans data/ and external_data/ for images, extracts MediaPipe hand
landmarks as wrist-relative feature vectors, and saves as .npy files
in processed_data/.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# ─── Paths (always absolute, safe from any cwd) ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
SEARCH_DIRS = [
    os.path.join(PROJECT_DIR, 'data'),
    os.path.join(PROJECT_DIR, 'external_data'),
]
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'processed_data')
IGNORE_DIRS = {'signs_database', '__pycache__', '.git'}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def process_image(img_path: str, label_out_dir: str) -> bool:
    """Extract landmarks from a single image and save as .npy."""
    image = cv2.imread(img_path)
    if image is None:
        return False

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return False

    for lms in results.multi_hand_landmarks:
        bx, by = lms.landmark[0].x, lms.landmark[0].y
        feature_vec = []
        for lm in lms.landmark:
            feature_vec.extend([lm.x - bx, lm.y - by])

        # Save as .npy
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(label_out_dir, f"{base_name}.npy")
        np.save(out_path, np.array(feature_vec, dtype=np.float32))
        return True

    return False


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_ok = 0
    total_fail = 0

    for search_dir in SEARCH_DIRS:
        if not os.path.isdir(search_dir):
            print(f"[Skip] {search_dir} not found")
            continue

        print(f"\n[Scan] {search_dir}")

        for label in sorted(os.listdir(search_dir)):
            label_in = os.path.join(search_dir, label)
            if not os.path.isdir(label_in) or label in IGNORE_DIRS:
                continue

            label_out = os.path.join(OUTPUT_DIR, label)
            os.makedirs(label_out, exist_ok=True)

            images = [
                f for f in os.listdir(label_in)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ]

            ok, fail = 0, 0
            for img_name in images:
                if process_image(os.path.join(label_in, img_name), label_out):
                    ok += 1
                else:
                    fail += 1

            total_ok += ok
            total_fail += fail
            status = "✓" if ok > 0 else "⚠"
            print(f"  {status} {label}: {ok} extracted, {fail} failed")

    print(f"\n[Done] Total: {total_ok} extracted, {total_fail} failed")


if __name__ == "__main__":
    run()
    hands.close()
