import cv2
import os

output_dir = './data'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

current_label = input('Enter label for the images (e.g., A, B, C, etc.): ')
output_path = os.path.join(output_dir, current_label)
os.makedirs(output_path, exist_ok=True)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('ASL Data Capture', frame)
    key = cv2.waitKey(1)
    
    if key == ord('s'):  # Press 's' to save frame
        image_path = os.path.join(output_path, f'{current_label}_{count}.jpg')
        cv2.imwrite(image_path, frame)
        count += 1
        print(f'Saved {image_path}')
    
    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
