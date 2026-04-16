import cv2
import numpy as np
import os

def create_mock_images(label, num_images=50):
    os.makedirs(f'data/{label}', exist_ok=True)
    for i in range(num_images):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some color to distinguish labels if needed
        if label == 'hello':
            img[:, :, 0] = 255 # Blue
        elif label == 'thanks':
            img[:, :, 1] = 255 # Green
        
        cv2.imwrite(f'data/{label}/{label}_{i}.jpg', img)
        print(f"Created data/{label}/{label}_{i}.jpg")

create_mock_images('hello')
create_mock_images('thanks')
create_mock_images('goodbye')
create_mock_images('I_Love_You')
create_mock_images('where')
create_mock_images('yes')
