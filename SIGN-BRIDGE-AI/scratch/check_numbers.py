from PIL import Image
import os
import glob

files = glob.glob(r"C:\Users\ASUS\.gemini\antigravity\brain\d1234857-5eac-4a68-93a0-35d47edda6a4\media__1776266*.png")
for f in files:
    try:
        img = Image.open(f)
        print(f"File: {os.path.basename(f)} Size: {img.size}")
    except Exception as e:
        print(f"Error on {f}: {e}")
