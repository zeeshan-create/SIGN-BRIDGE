import cv2
import numpy as np
import os

img_path = r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776266045567.png"
out_dir = r"d:\Antigravity\ASL-Detection\web\public\asl"

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# The individual photos are black. The canvas is white.
# Threshold to isolate the photos: anything NOT white.
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bboxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 50: # filter out text characters or noise
        bboxes.append((x, y, w, h))

# Sort by Y first (rows), then X (cols)
# Given 2 rows, we can sort by Y. Differences in Y > 100 mean a new row.
bboxes.sort(key=lambda b: (int(b[1] / 100), b[0]))

# Based on the user image:
# Row 1: 0, 1, 2, 3, 4, 5 (Wait, the labels were weird, but numerically we need 0-9)
# Let's map exactly to 0-10:
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

for i, (x, y, w, h) in enumerate(bboxes):
    if i >= len(digits):
        break
    
    char = digits[i]
    # Crop the exact photo
    roi = img[y:y+h, x:x+w]
    
    # Resize to 200x200
    roi_resized = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(os.path.join(out_dir, f"{char}.jpg"), roi_resized)

print("Digits successfully processed and added to the converter.", len(bboxes), "photos found.")
