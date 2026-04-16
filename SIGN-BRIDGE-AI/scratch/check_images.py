from PIL import Image
import os

images = [
    r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776265562162.jpg",
    r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776265562166.jpg",
    r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776265562189.jpg"
]

for img_path in images:
    if os.path.exists(img_path):
        img = Image.open(img_path)
        print(f"File: {os.path.basename(img_path)} Size: {img.size} Mode: {img.mode}")
