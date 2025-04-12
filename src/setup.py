from setuptools import setup, find_packages

setup(
    name="image_mosaic_maker",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "image_mosaic_maker.models": ["*"],
        "image_mosaic_maker.example_images": ["*"],
    },
    entry_points={
        "console_scripts": [
            "mosaic=image_mosaic_maker:main",  # Base command
        ],
    },
    install_requires=[
        "Pillow>=9.0.0",           # For PIL (Image, ImageDraw, ImageFilter)
        "numpy>=1.20.0",           # For numpy
        "scipy>=1.0.0",            # For scipy.spatial.cKDTree
        "tqdm>=4.0.0",             # For progress bars
        "ultralytics>=8.0.0",      # For YOLO
    ],
    extras_require={
            'cuda': [
                'torch>=2.0.0',
                'torchvision>=0.15.0',
                'torchaudio>=2.0.0',
            ],
            'cpu': [
                'torch>=2.0.0', 
                'torchvision>=0.15.0',
                'torchaudio>=2.0.0',
            ],
        },
    },

    description="A tool for creating photo mosaics from collections of images.",
    author="Jonas Mueller",
    author_email="jonas.mueller.wpk@gmail.com",
    url="https://github.com/jmueller209/Image-Mosaic-Maker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Intended Audience :: End Users/Desktop",
    ],
    python_requires=">=3.10",

)
