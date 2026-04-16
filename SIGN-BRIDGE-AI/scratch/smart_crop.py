import cv2
import numpy as np
import os

img_path = r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776265562189.jpg"
out_dir = r"d:\Antigravity\ASL-Detection\web\public\asl"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img = cv2.imread(img_path)
height, width = img.shape[:2]
cols = 9
rows = 4

w = width / cols
h = height / rows

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
idx = 0

for r in range(3):
    for c in range(9):
        if idx >= len(letters):
            break
        char = letters[idx]
        
        # Initial loose grid crop
        left = int(c * w)
        top = int(r * h)
        right = int((c + 1) * w)
        bottom = int((r + 1) * h)
        
        roi = img[top:bottom, left:right]
        
        # Check background color - usually the top left pixel is a safe bet for background in this image
        bg_color = roi[5, 5]
        
        # We need to find the bounding box of NOT orange pixels
        # An easy way holds: difference from bg_color
        diff = cv2.absdiff(roi, np.full(roi.shape, bg_color, dtype=np.uint8))
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold: if difference is > 20, it's part of the hand/letter
        _, mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x_arr, y_arr, w_arr, h_arr = [], [], [], []
            for cnt in contours:
                x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                # Ignore tiny specks
                if w_b > 2 and h_b > 2:
                    x_arr.append(x_b)
                    y_arr.append(y_b)
                    w_arr.append(w_b)
                    h_arr.append(h_b)
            
            if x_arr:
                min_x = min(x_arr)
                min_y = min(y_arr)
                max_x = max([x_arr[i] + w_arr[i] for i in range(len(x_arr))])
                max_y = max([y_arr[i] + h_arr[i] for i in range(len(y_arr))])
                
                # Add a little padding tight to the actual artwork
                pad = 4
                crop_y1 = max(0, min_y - pad)
                crop_y2 = min(roi.shape[0], max_y + pad)
                crop_x1 = max(0, min_x - pad)
                crop_x2 = min(roi.shape[1], max_x + pad)
                
                hand_crop = roi[crop_y1:crop_y2, crop_x1:crop_x2]
                
                target_size = 200
                # Fill new square canvas with the background color
                final_img = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
                
                hh, hw = hand_crop.shape[:2]
                
                # Scale up or down to fit 80% of the canvas
                canvas_avail = int(target_size * 0.8)
                scale = min(canvas_avail / hh, canvas_avail / hw)
                new_w, new_h = int(hw * scale), int(hh * scale)
                
                hand_crop_resized = cv2.resize(hand_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Center exactly
                y_off = (target_size - new_h) // 2
                x_off = (target_size - new_w) // 2
                
                final_img[y_off:y_off+new_h, x_off:x_off+new_w] = hand_crop_resized
                
                cv2.imwrite(os.path.join(out_dir, f"{char}.jpg"), final_img)
        idx += 1

print("Smart crop completed.")
