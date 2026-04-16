from PIL import Image
import os

img_path = r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776265562189.jpg"
out_dir = r"d:\Antigravity\ASL-Detection\web\public\asl"

img = Image.open(img_path)
width, height = img.size

cols = 9
rows = 4

w = width / cols
h = height / rows

# Layout:
# Row 0: A B C D E F G H I
# Row 1: J K L M N O P Q R
# Row 2: S T U V W X Y Z

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

idx = 0
for r in range(3):
    for c in range(9):
        if idx >= len(letters):
            break
        char = letters[idx]
        left = c * w
        top = r * h
        right = (c + 1) * w
        bottom = (r + 1) * h
        
        box = (int(left), int(top), int(right), int(bottom))
        char_img = img.crop(box)
        char_img.save(os.path.join(out_dir, f"{char}.jpg"))
        idx += 1

print("Cropped A-Z successfully.")
