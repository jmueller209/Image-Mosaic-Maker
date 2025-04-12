import numpy as np
import os
import time
from PIL import Image
from pathlib import Path
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from image_mosaic_maker.utils.paths import MODELS_PATH, PROCESSED_IMGS_PATH
from image_mosaic_maker.utils.debugging import debug_print

def assemble_mosaic(mosaic_img_path, save_path, num_columns, enable_debugging, color_shift_factor):
    start_time = time.time()
    if not save_path.lower().endswith(('.jpg', '.jpeg')):
        print("Consider saving as .JPEG for better performance")

    if not os.path.isdir(PROCESSED_IMGS_PATH) or len(os.listdir(PROCESSED_IMGS_PATH)) == 0:
        error = f"There are no processed images available."
        return False, error
    
    files = os.listdir(PROCESSED_IMGS_PATH)
    mosaic_tiles = [f for f in files if f.endswith(".png")]
    
    if len(mosaic_tiles) == 0:
        error = f"Processed image directory contains files of wrong type. Process images again."
        return False, error
    
    first_mosaic_tile_path = os.path.join(PROCESSED_IMGS_PATH, mosaic_tiles[0])
    
    # Open the image using PIL to get width and height
    tile_width, tile_height = Image.open(first_mosaic_tile_path).size

    mosaic_img = Image.open(mosaic_img_path)
    mosaic_img_width, mosaic_img_height = mosaic_img.size
    mosaic_aspect_ratio = mosaic_img_width/mosaic_img_height

    mosaic_width = num_columns*tile_width
    mosaic_height = mosaic_width/mosaic_aspect_ratio

    num_rows = round(mosaic_height/tile_height)
    mosaic_height = num_rows*tile_height

    resized_mosaic_img = mosaic_img.resize((num_columns, num_rows))
    
    # Saving color data in dictionary
    colors = []
    for tile in mosaic_tiles:
        # Split into parts between parentheses
        parts = tile.split("(", 1)  # Split into [prefix, color_part...]
        
        color_part = parts[1]
        color_str = color_part.split(")", 1)[0]  # Split at first ')'
        color_tuple = tuple(map(int, color_str.split(", ")))
        colors.append(color_tuple)

    # create color tree
    color_tree = cKDTree(colors)

    # Create list of all tile positions and colors
    tile_args = [
        (x, y, resized_mosaic_img.getpixel((x, y)))
        for y in range(num_rows)
        for x in range(num_columns)
    ]

    # Create partial function with fixed parameters
    worker = partial(process_tile,
        color_tree=color_tree,
        mosaic_tiles=mosaic_tiles,
        tile_width=tile_width,
        tile_height=tile_height,
        color_shift_factor=color_shift_factor
    )

    # Create empty mosaic canvas
    mosaic = Image.new('RGB', 
        (num_columns * tile_width, num_rows * tile_height))

    # Process tiles in parallel with progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker, tile_args, chunksize=100),
            total=len(tile_args),
            desc="Building mosaic",
            unit="tile"
        ))

    # Paste all processed tiles into final mosaic
    for position, tile in results:
        mosaic.paste(tile, position)

    debug_print(f"Saving mosaic...", enable_debugging)
    mosaic.save(save_path, 
            quality=100, 
            subsampling=1,  # 4:2:0 chroma
            optimize=False,
            progressive=False)
    
    debug_print(f"Saved mosaic at {save_path} with size ({mosaic_width}, {mosaic_height})", enable_debugging)
    debug_print(f"Building the mosaic took {time.time()-start_time} seconds.", enable_debugging)
    return True, None

def process_tile(args, color_tree, mosaic_tiles, tile_width, tile_height, color_shift_factor):

    x, y, pixel_color = args
    # Find closest matching image
    closest_image, _ = find_closest_color_with_kdtree(pixel_color, mosaic_tiles, color_tree)
    
    # Load and process tile
    tile_path = os.path.join(PROCESSED_IMGS_PATH, closest_image)
    img = Image.open(tile_path)
    processed = apply_color_filter(img, pixel_color, color_shift_factor)

    # Calculate position and return both image and position
    position = (x * tile_width, y * tile_height)
    return position, processed

def find_closest_color_with_kdtree(input_color, mosaic_tiles, tree):
    # Query the k-d tree to find the closest color
    distance, index = tree.query(input_color)
    # Get the corresponding image for the closest color
    closest_image = mosaic_tiles[index]
    
    return closest_image, distance

def apply_color_filter(img, target_color, color_shift_factor):
    if color_shift_factor == 0:
        return img
    target_img = Image.new('RGB', img.size, target_color)
    return Image.blend(img, target_img, color_shift_factor)
