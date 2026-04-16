import cv2
import mediapipe as mp
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'external_data')
OUTPUT_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'processed_data')

ALPHABET = list("ABCDEFGHIKLMNOPQRSUVWYZ")
EXPRESSIONS = [
    'hello', 'goodbye', 'thanks', 'yes', 'no', 'please', 'Sorry', 'Maybe', 
    'I_Love_You', 'How', 'You', 'Name', 'Deaf', 'Hearing'
]
LABELS = ALPHABET + EXPRESSIONS

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)

def process_external_images():
    if not os.path.exists(INPUT_FOLDER): return
    for label in os.listdir(INPUT_FOLDER):
        label_in, label_out = os.path.join(INPUT_FOLDER, label), os.path.join(OUTPUT_FOLDER, label)
        os.makedirs(label_out, exist_ok=True)
        print(f"[Ingestion] Processing Folder: {label}")
        for img_name in os.listdir(label_in):
            img_path = os.path.join(label_in, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    bx, by = lms.landmark[0].x, lms.landmark[0].y
                    v = []
                    for lm in lms.landmark: v.extend([lm.x - bx, lm.y - by])
                    np.save(os.path.join(label_out, img_name + ".npy"), v)
                    print(f"    [OK] Captured: {img_name}")

if __name__ == "__main__":
    process_external_images()
