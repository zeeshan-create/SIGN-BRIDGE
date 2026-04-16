import cv2
import os

img_path = r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776266003162.jpg"
out_dir = r"d:\Antigravity\ASL-Detection\web\public\asl"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img = cv2.imread(img_path)
h, w = img.shape[:2]

# The image is a 5 column grid with roughly 5.6 rows (because the last row is shorter, or it's exactly 6 rows).
# Looking at the original: 6 rows.
cols = 5
rows = 6

col_w = w / cols
row_h = h / rows

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
idx = 0

for r in range(rows):
    for c in range(cols):
        if idx >= len(letters):
            break
        char = letters[idx]
        
        left = int(c * col_w)
        top = int(r * row_h)
        right = int((c + 1) * col_w)
        bottom = int((r + 1) * row_h)
        
        # Crop exactly to the grid cell
        roi = img[top:bottom, left:right]
        
        # Resize to square 200x200 standard size for the UI
        roi_resized = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_CUBIC)
        
        # Overwrite the existing files with the new realistic photo dataset
        cv2.imwrite(os.path.join(out_dir, f"{char}.jpg"), roi_resized)
        
        idx += 1

print("New realistic photo dataset successfully sliced and integrated.")
