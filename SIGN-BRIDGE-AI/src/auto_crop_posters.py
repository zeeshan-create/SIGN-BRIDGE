import cv2
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR) 
EXTERNAL_DATA_DIR = os.path.join(PROJECT_ROOT, 'external_data')

PHOTO_CONFIGS = {
    'photo_essential.jpg': {
        'rows': 2, 'cols': 3,
        'labels': [['hello', 'yes', 'no'], ['thanks', 'please', 'Sorry']]
    },
    'photo_alphabet.jpg': {
        'rows': 6, 'cols': 4,
        'labels': [
            ['A', 'B', 'C', 'D'], ['F', 'G', 'H', 'I'],
            ['J', 'L', 'M', 'O'], ['K', 'L', 'M', 'N'],
            ['P', 'Q', 'R', 'S'], ['U', 'V', 'W', 'Y']
        ]
    },
    'photo_phrases.jpg': {
        'rows': 7, 'cols': 4, # Estimated grid for the large poster
        'labels': [
            ['hello', 'hello', 'hello', 'hello'],
            ['How', 'How', 'How', 'How'],
            ['You', 'You', 'You', 'You'],
            ['How', 'How', 'How', 'How'], # Placeholder for rows
            ['Name', 'Name', 'Name', 'Name'],
            ['Name', 'Name', 'Name', 'Name'],
            ['thanks', 'Deaf', 'Hearing', 'Hearing']
        ]
    }
}

def crop_photo_poster(image_name, config):
    search_paths = [
        os.path.join(os.path.dirname(PROJECT_ROOT), image_name),
        os.path.join(PROJECT_ROOT, image_name),
        os.path.join('D:\\Antigravity', image_name)
    ]
    img_path = next((p for p in search_paths if os.path.exists(p)), None)
    if not img_path: return

    img = cv2.imread(img_path)
    if img is None: return
    h, w, _ = img.shape
    rows, cols = config['rows'], config['cols']
    grid = config['labels']
    cell_h, cell_w = h // rows, w // cols
    
    print(f"[Cropper] Processing {image_name}...")
    for r in range(rows):
        for c in range(cols):
            if r >= len(grid) or c >= len(grid[0]): continue
            label = grid[r][c]
            label_dir = os.path.join(EXTERNAL_DATA_DIR, label)
            os.makedirs(label_dir, exist_ok=True)
            cell = img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            cv2.imwrite(os.path.join(label_dir, f"{image_name}_{r}_{c}.jpg"), cell[0:int(cell_h*0.8), :])

def main():
    for poster, config in PHOTO_CONFIGS.items():
        crop_photo_poster(poster, config)

if __name__ == "__main__":
    main()
