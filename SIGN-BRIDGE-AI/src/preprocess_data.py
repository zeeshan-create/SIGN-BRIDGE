import cv2
import os
import mediapipe as mp


input_dir = './data'
output_dir = './processed_data'
os.makedirs(output_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


image_size = (224, 224)


for label_folder in os.listdir(input_dir):
    label_path = os.path.join(input_dir, label_folder)
    output_path = os.path.join(output_dir, label_folder)
    os.makedirs(output_path, exist_ok=True)

    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        image = cv2.imread(image_path)

        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            # Code to extract hand region
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, c = image.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                margin = 20
                x_min = max(0, int(x_min - margin))
                y_min = max(0, int(y_min - margin))
                x_max = min(w, int(x_max + margin))
                y_max = min(h, int(y_max + margin))

                hand_image = image[y_min:y_max, x_min:x_max]
                hand_image_resized = cv2.resize(hand_image, image_size)

                output_file = os.path.join(output_path, image_file)
                cv2.imwrite(output_file, hand_image_resized)
                print(f"Processed and saved (Hand Detected): {output_file}")
        else:
            # Fallback: Save resized image even if no hand detected (for testing pipeline)
            hand_image_resized = cv2.resize(image, image_size)
            output_file = os.path.join(output_path, image_file)
            cv2.imwrite(output_file, hand_image_resized)
            print(f"No hand detected in {image_file}, saving resized image as fallback.")

# Release Mediapipe resources
hands.close()
