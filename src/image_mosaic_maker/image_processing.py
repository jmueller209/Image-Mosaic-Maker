import time
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO
from image_mosaic_maker.utils.paths import MODELS_PATH, PROCESSED_IMGS_PATH
from image_mosaic_maker.utils.debugging import debug_print

def process_images(images_path, crop_type="aspect_ratio_crop", num_threads=16, enable_debugging=False, **kwargs):
    debug_print("Start processing files.", enable_debugging)
    # create directory if it does not exist yet
    PROCESSED_IMGS_PATH.mkdir(parents=True, exist_ok=True)

    # delete all remaining files in that directory
    for filename in os.listdir(PROCESSED_IMGS_PATH):
        file_path = os.path.join(PROCESSED_IMGS_PATH, filename)
        # Check if it's a file and delete it
        if os.path.isfile(file_path):
            os.remove(file_path)

    # get the right crop function
    match crop_type:
        case "center_crop":
            crop_function = center_crop
        case "smart_crop":
            crop_function = smart_crop
        case "stretch":
            crop_function = stretch
        case "aspect_ratio_crop":
            crop_function = aspect_ratio_crop
        case "aspect_ratio_resize":
            crop_function = aspect_ratio_resize
        case _:
            error = f"{crop_type} is not a valid crop type!"
            return False, error
        
    num_cores = os.cpu_count()
    debug_print(f"Number of available cores: {num_cores}", enable_debugging)
    if num_threads == "all":
        num_threads = num_cores
    elif isinstance(num_threads, int) and 0 <= num_threads <= num_cores:
        pass
    else:
        error = f"'{num_threads}' is not a valid option to specify the number of threads. Make sure you either provide an integer between 0 and the number of available cores ({num_cores}) or 'all' to use all cores!"
        return False, error

    files = os.listdir(images_path)
    # create thread pool

    debug_print(f"Using {num_threads} cores to process {len(files)} files.", enable_debugging)
    start_time = time.time()

    # Create a ProcessPoolExecutor with a fixed number of processes
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = [executor.submit(process_file, filename, images_path, PROCESSED_IMGS_PATH, crop_function, kwargs) for filename in files]
        
        # Use tqdm to display the progress bar and track task completion
        for future in tqdm(futures, desc="Processing files", total=len(files)):
            future.result()  # Block until each future completes
    
    processing_time = time.time()-start_time
    debug_print(f"Finished in {processing_time} seconds.", enable_debugging)
    return True, None

def process_file(filename, input_path, output_path, crop_function, kwargs):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            img_path = os.path.join(input_path, filename)
            unprocessed_img = Image.open(img_path)
            if unprocessed_img.mode != 'RGBA':
                unprocessed_img = unprocessed_img.convert('RGBA')
            # Create a black RGB background
            img = Image.new('RGB', unprocessed_img.size, (0, 0, 0))
            # Paste the image onto the black background using the alpha channel as a mask
            img.paste(unprocessed_img, mask=unprocessed_img.split()[-1])  # Use the alpha channel as the mask

            # process image
            cropped_img = crop_function(img, kwargs)
            if cropped_img == None:
                return
            
            avg_color = calculate_average_color(cropped_img)
            new_filename = f"{avg_color}{filename}"[:-4] + ".png"
            # save image
            output_path = os.path.join(output_path, new_filename)
            cropped_img.save(output_path, 'PNG')
        
        except Exception as e:
            print(f"Error processing {filename}: {e} ")
        
    else:
        warning = f"{filename} does not seem to be a valid image. Skipping File."
        print(warning)

def center_crop(img, kwargs):
    # get target size
    target_width, target_height = kwargs["target_size"]

    # get current img size
    original_width, original_height = img.size

    # If image is larger => crop
    if original_width >= target_width and original_height >= target_height:
        left = (original_width - target_width) // 2
        top = (original_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return img.crop((left, top, right, bottom))

    # If image is smaller => paste it on a blurred background
    # Step 1: Resize to fit inside target, preserving aspect ratio
    img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

    # Step 2: Create blurred background from original
    background = img.copy().resize((target_width, target_height), Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(radius=20))

    # Step 3: Paste the resized image in the center
    bg = background.copy()
    offset_x = (target_width - img.size[0]) // 2
    offset_y = (target_height - img.size[1]) // 2
    bg.paste(img, (offset_x, offset_y))

    return bg

def stretch(img, kwargs):
    # Get target size
    target_width, target_height = kwargs["target_size"]

    # Stretch image to fit target size exactly
    stretched_img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

    return stretched_img

def aspect_ratio_crop(img, kwargs):
    # Retrieve the target size from kwargs
    target_size = kwargs.get("target_size")  # Expected format (width, height)
    target_width, target_height = target_size
    # Get the original image size (width, height)
    original_width, original_height = img.size
    
    # Calculate aspect ratios
    target_aspect_ratio = target_width / target_height
    img_aspect_ratio = original_width / original_height
    
    if img_aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio, crop the sides (left and right equally)
        new_width = int(original_height * target_aspect_ratio)
        offset = (original_width - new_width) // 2  # Calculate how much to crop on each side
        img_cropped = img.crop((offset, 0, original_width - offset, original_height))

    elif img_aspect_ratio < target_aspect_ratio:
        # Image is taller than target aspect ratio, crop the top and bottom equally
        new_height = int(original_width / target_aspect_ratio)
        offset = (original_height - new_height) // 2  # Calculate how much to crop on each side
        img_cropped = img.crop((0, offset, original_width, original_height - offset))
    else:
        # Image is already in the same aspect ratio as the target
        img_cropped = img

    # Resize the image to the target size
    img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
    return img_resized

def aspect_ratio_resize(img, kwargs):
    target_width, target_height = kwargs["target_size"]
    target_aspect = target_width / target_height

    original_width, original_height = img.size
    original_aspect = original_width / original_height

    # Decide resizing based on which dimension should be fixed
    if original_aspect > target_aspect:
        # Image is too wide → fit width, pad top/bottom
        new_width = target_width
        new_height = round(target_width / original_aspect)
    else:
        # Image is too tall → fit height, pad left/right
        new_height = target_height
        new_width = round(target_height * original_aspect)

    # Resize with maintained aspect ratio
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create blurred background
    blurred_bg = img.copy().resize((target_width, target_height), Image.Resampling.LANCZOS)
    blurred_bg = blurred_bg.filter(ImageFilter.GaussianBlur(radius=20))

    # Center resized image on background
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    blurred_bg.paste(img_resized, (offset_x, offset_y))

    return blurred_bg

def smart_crop(img, kwargs):
    target_size = kwargs["target_size"]
    
    try:
        fall_back_crop_type = kwargs['fall_back_crop_type']
    except:
        fall_back_crop_type = aspect_ratio_resize
        print("No fall_back_crop_type specified. Will default to 'aspect_ratio_resize'.")

    try: 
        device_name = kwargs["device"]
    except:
        device_name = "cpu"
        print(f"No device type provided for running face recognition. Will default to using {device_name}.")

    try: 
        face_padding_ratio = kwargs["face_padding_ratio"]
    except:
        face_padding_ratio = 1
        print(f"No face_padding_ratio provided. Will default to using '1', which might result in parts of faces being cut off.")

    
    img_array = np.array(img)

    # Setting up the face detection model
    boxes = None
    model_path = MODELS_PATH / "yolov11n-face.pt"
    yolo_face_model = YOLO(model_path) 
    yolo_face_model.to(device_name)
    # Run detection
    results = yolo_face_model(img_array, verbose=False)[0]
    if results.boxes is not None and results.boxes.shape[0] > 0:
        # Extract boxes (xyxy format) and convert to NumPy
        boxes = results.boxes.xyxy.cpu().numpy()

    # If no faces detected, fallback to aspect_ratio_crop
    if boxes is None or len(boxes) == 0:
        match fall_back_crop_type:
            case "center_crop":
                crop_function = center_crop
            case "stretch":
                crop_function = stretch
            case "aspect_ratio_crop":
                crop_function = aspect_ratio_crop
            case "aspect_ratio_resize":
                crop_function = aspect_ratio_resize
            case _:
                crop_function = aspect_ratio_resize

        return crop_function(img, kwargs)

    else:
        # Retrieve the target size from kwargs
        target_size = kwargs.get("target_size")  # Expected format (width, height)
        target_width, target_height = target_size
        # Get the original image size (width, height)
        original_width, original_height = img.size
        
        # Calculate aspect ratios
        target_aspect_ratio = target_width / target_height
        img_aspect_ratio = original_width / original_height

        # Scale the boxes so that no parts of the faces are cut off
        img = img.convert('RGB')

        # Initialize ImageDraw to draw on the image
        draw = ImageDraw.Draw(img)
        scaled_boxes = []
        for box in boxes:
            width_offset = (box[2] - box[0]) * face_padding_ratio // 2
            height_offset = (box[3] - box[1]) * face_padding_ratio // 2
            x1 = max(box[0] - width_offset, 0)
            y1 = max(box[1] - height_offset, 0)
            x2 = min(box[2] + width_offset, original_width)
            y2 = min(box[3] + height_offset, original_height)

            scaled_boxes.append([x1, y1, x2, y2])
            #draw.rectangle([x1, y1, x2, y2], outline="green", width=10) # draw rectangles on faces for debugging purposes

        # Get bounding box that includes all faces
        x1 = min([box[0] for box in scaled_boxes])
        y1 = min([box[1] for box in scaled_boxes])
        x2 = max([box[2] for box in scaled_boxes])
        y2 = max([box[3] for box in scaled_boxes])
        face_box_width = x2-x1
        face_box_height = y2-y1


        if img_aspect_ratio > target_aspect_ratio:
            # Image is wider than target aspect ratio, crop the sides without cutting off faces
            new_width = max(int(original_height * target_aspect_ratio), face_box_width)
            delta = (new_width - face_box_width) // 2
            crop_x1 = max(0, x1-delta)
            crop_x2 = int(crop_x1 + new_width)
            if crop_x2 > original_width:
                crop_x2 = int(original_width)
                crop_x1 = int(crop_x2-new_width)

            img_cropped = img.crop((crop_x1, 0, crop_x2, original_height))

        elif img_aspect_ratio < target_aspect_ratio:
            # Image is taller than target aspect ratio, crop the top and bottom without cutting of faces
            new_height = max(int(original_width / target_aspect_ratio), face_box_height)
            delta = (new_height - face_box_height) // 2
            crop_y1 = max(0, y1-delta)
            crop_y2 = int(crop_y1 + new_height)
            if crop_y2 > original_height:
                crop_y2 = int(original_height)
                crop_y1 = int(crop_y2-new_height)

            img_cropped = img.crop((0, crop_y1, original_width, crop_y2))
        else:
            # Image is already in the same aspect ratio as the target
            img_cropped = img

        img_resized = aspect_ratio_resize(img_cropped, kwargs)
        #img_resized = img_cropped
    return img_resized

def calculate_average_color(img):
    # Resize image to 1x1 pixel
    resized_image = img.resize((1, 1))
    # Get pixel color
    color = resized_image.getpixel((0, 0))
    return color
