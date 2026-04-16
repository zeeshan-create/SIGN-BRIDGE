from PIL import Image
import os

img_path = r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776266003162.jpg"

if os.path.exists(img_path):
    img = Image.open(img_path)
    print(f"File: {os.path.basename(img_path)} Size: {img.size} Mode: {img.mode}")
else:
    print("Not found")
