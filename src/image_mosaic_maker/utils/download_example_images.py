import requests
import os
import random
from tqdm import tqdm

def download_random_images(count, min_size=50, max_size=300, save_path=None):
    # Create the directory where the images will be saved
    os.makedirs(save_path, exist_ok=True)

    for i in tqdm(range(count), desc="Downloading images"):
        width = random.randint(min_size, max_size)
        height = random.randint(min_size, max_size)
        url = f"https://picsum.photos/{width}/{height}"
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(save_path, f"random_image_{i+1}_{width}x{height}.jpg")
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download image {i+1}")

