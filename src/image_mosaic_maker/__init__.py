import argparse
import os
import sys
from image_mosaic_maker.assemble_mosaic import assemble_mosaic
from image_mosaic_maker.image_processing import process_images
from image_mosaic_maker.utils.download_example_images import download_random_images
from image_mosaic_maker.utils.paths import EXAMPLE_IMGS_PATH, RANDOM_IMGS_PATH


def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="A tool for creating photomosaics from collections of images."
    )
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Process Parser
    process_parser = subparsers.add_parser(
        "process_images", 
        help="Process a directory of images, which can then be used to create a photomosaic."
    )
    process_parser.add_argument(
        "path", 
        type=str, 
        help="Path to the directory containing the images to be processed."
    )
    process_parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[36, 36], 
        help="Target size for processed images (default: 36 36)"
    )
    process_parser.add_argument(
        "--crop_type",
        type=str,
        choices=["smart_crop", "center_crop", "aspect_ratio_resize", "aspect_ratio_crop", "stretch"],
        default="smart_crop",
        help="Primary cropping method to use (default: center_crop)"
    )
    process_parser.add_argument(
        "--fallback_crop_type",
        type=str,
        choices=["center_crop", "aspect_ratio_resize", "aspect_ratio_crop", "stretch"],
        default="aspect_ratio_resize",
        help="Fallback cropping method. This is only used when the primary cropping method is smart crop and no faces are detected (default: aspect_ratio_resize)"
    )
    process_parser.add_argument(
        "--enable_debugging",
        action="store_true",
        help="Enable debug output and logging"
    )
    process_parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Number of threads to use for processing (default: 0 (all available cores))"
    )
    process_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device to use for face recognition during smart cropping (default: cpu)"
    )
    process_parser.add_argument(
        "--face_padding_ratio",
        type=float,
        default=1.1,
        help="Padding ratio around detected faces (default: 1.1)"
    )

    # Build Parser
    build_parser = subparsers.add_parser(
        "build", 
        help="Build the mosaic using the preprocessed images."
    )

    build_parser.add_argument(
        "mosaic_img_path",
        type=str,
        help="Path to the base image that will be converted into a mosaic"
    )
    build_parser.add_argument(
        "num_columns",
        type=int,
        help="Number of columns in the output mosaic (controls density of tiles)"
    )

    # Optional arguments
    build_parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(os.getcwd(), "mosaic.jpeg"),
        help="Path where the final mosaic image will be saved"
    )
    build_parser.add_argument(
        "--enable_debugging",
        action="store_true",
        help="Enable debug output and logging"
    )
    build_parser.add_argument(
        "--color_shift_factor",
        type=float,
        default=0,
        help="Color adjustment factor (0.0-1.0) for tile matching (default: 0)"
    )

    # Example parser
    example_parser = subparsers.add_parser(
        "create_example", 
        help="Create an example mosaic."
    )
    example_parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Combined with the --download argument, this specifies how many random images will be used for the mosaic creation")
    
    example_parser.add_argument(
        "--download_type",
        type=str,
        choices=["fill","new", "add"],
        default="fill",
        help="'fill' checks how many images have been downloaded already and will download as many new images as necessary to meet the demand. 'New' will delete the current images and start a new download. 'Add' will keep all the images that have been downloaded already and add the specified amount of new images."
    )

    example_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Specifies the device that should be used for image processing."
        )


    # Parse the arguments
    args = parser.parse_args()

    if args.command == "process_images":
        if args.num_threads == 0:
            num_threads = 'all'
        else:
            num_threads = args.num_threads
        successful, error = process_images(images_path=args.path, crop_type=args.crop_type, fall_back_crop_type=args.fallback_crop_type, enable_debugging=args.enable_debugging, num_threads=num_threads, target_size=args.target_size, device=args.device, face_padding_ratio=args.face_padding_ratio)
        if not successful:
            print(error)
            sys.exit()
    
    elif args.command == "build":
        successful, error = assemble_mosaic(mosaic_img_path = args.mosaic_img_path, save_path=args.save_path, num_columns=args.num_columns, enable_debugging=args.enable_debugging, color_shift_factor=args.color_shift_factor)
        if not successful:
            print(error)
            sys.exit()

    elif args.command == "create_example":
        # Download images
        num_images = args.num_images
        download_type = args.download_type
        match download_type:
            case "fill":
                try: 
                    currently_available_images = len(os.listdir(RANDOM_IMGS_PATH))
                except:
                    currently_available_images = 0
                num_images = max(0, num_images-currently_available_images)
            case "new":
                os.remove(RANDOM_IMGS_PATH)
            case _:
                pass

        download_random_images(num_images, save_path=RANDOM_IMGS_PATH)

        # Process images
        num_cores = 'all'
        device = args.device
        if device == "cuda":
            num_cores = max(os.cpu_count()//2, 1)
        successful, error = process_images(images_path=RANDOM_IMGS_PATH, crop_type="smart_crop", fall_back_crop_type="aspect_ratio_resize", enable_debugging=True, num_threads=num_cores, target_size=(200, 200), device=device, face_padding_ratio=1.2)
        if not successful:
            print(error)
            sys.exit()

        # Asssemble Mosaics
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        for filename in os.listdir(EXAMPLE_IMGS_PATH):
            if not os.path.splitext(filename)[1] in image_extensions:
                continue
            img_path = os.path.join(EXAMPLE_IMGS_PATH, filename)
            save_name = f"mosaic_{os.path.splitext(filename)[0]}.jpg"
            save_path = os.path.join(os.getcwd(), save_name)
            successful, error = assemble_mosaic(mosaic_img_path = img_path, save_path=save_path, num_columns=150, enable_debugging=True, color_shift_factor=0.3)
            if not successful:
                print(error)
                

if __name__ == "__main__":
    main()