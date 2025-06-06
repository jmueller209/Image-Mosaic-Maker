# 🖼️ Image Mosaic Maker

Welcome to the **Image Mosaic Toolkit**—a command-line tool that simplifies the creation of image mosaics! It supports several cropping techniques including smart crop, which uses face detection to preserve important details when processing images.

---

## 🚀 Features

- **Multiple Cropping Options:** Choose from smart crop, center crop, aspect ratio adjustments, and stretching.
- **Face Recognition:** Smart cropping using face detection with YOLO model.
- **Customizable Mosaic Generation:** Adjust density, color matching, and dimensions.

**_Important Note:_**
I have not optimized for memory efficiency yet. in order to create decently looking mosaics, 16GB of available RAM is the bare minimum. For larger mosaics 32GB is recommended.

---

## 🛠️ Installation
Make sure `python` and `pipx` are installed on your system. You can install `pipx` by running
```bash
pip install pipx
```
Now you can install the CLI-tool:

Install CPU-only version:
```bash
pipx install --verbose "image_mosaic_maker[cpu] @ git+https://github.com/jmueller209/Image-Mosaic-Maker.git#subdirectory=src" --pip-args="--extra-index-url https://download.pytorch.org/whl/cpu"
```
Install version with CUDA support:
```bash
pipx install --verbose "image_mosaic_maker[cuda] @ git+https://github.com/jmueller209/Image-Mosaic-Maker.git#subdirectory=src" --pip-args="--extra-index-url https://download.pytorch.org/whl/cu118"
```

You might need to run the following command to make the tool globally available:
```bash
pipx ensurepath
```
---

**_Important Note:_**
Unfortunately, `pipx` (the module used for installing this CLI tool) will not show any progress bars during installation. Since some of the dependencies are quite large (1-2 GB depending on wether you install the CPU or GPU version), the installation process might take a while. Therefore, be patient and do not panic when you do not see command line output for some time during the installation.

## 🎯 Command Usage

### 1️⃣ **Process Images**

Prepares images for mosaic creation:

```bash
mosaic process_images <path> [options]
```

**Options:**

- `--target_size WIDTH HEIGHT`: Set output image size (default: `36 36`).
- `--crop_type`: Choose cropping method (default: `smart_crop`):

  - `smart_crop` (face-aware)
  - `center_crop`
  - `aspect_ratio_resize`
  - `aspect_ratio_crop`
  - `stretch`

- `--num_threads`: Number of threads for parallel processing (default: `0` (Uses all cores)).
- `--enable_debugging`: Enables debug output.

The following options are only relevant when using `smart_crop`:

- `--fallback_crop_type`: Cropping method if `smart_crop` detects no faces (default: `aspect_ratio_resize`).
- `--device`: Compute device (`cuda` or `cpu`; default: `cpu`).
- `--face_padding_ratio`: Padding around detected faces (default: `1.1`).

**_Important Note:_**
When using `smart_crop` with CUDA, you should set `num_threads` to approximately half of your CPU cores for better performance.

**Example:**

```bash
mosaic process_images ./my_images --target_size 192 108 --crop_type smart_crop --num_threads 6 --device cuda --fallback_crop_type stretch --face_padding_ratio 0 --enable_debugging
```

---

### 2️⃣ **Build Mosaic**

Generates the final mosaic image. This function uses processed images that were generated by the `process_images` command:

```bash
mosaic build <mosaic_img_path> <num_columns> [options]
```

**Options:**

- `--save_path`: Output location for the mosaic image (default: current working directory).
- `--color_shift_factor`: Adjusts color matching accuracy (`0.0`–`1.0`; default: `0`).
- `--enable_debugging`: Enables detailed debug output.

**Example:**

```bash
mosaic build base.jpg 50 --save_path ./output/mosaic.jpg --color_shift_factor 0.3 --enable_debugging
```

---

### 3️⃣ **Create Example Mosaic**

Quickly creates some example mosaics using randomly downloaded images:

```bash
mosaic create_example [options]
```

**Options:**

- `--num_images`: Number of random images to download and use (default: `70`).
- `--download_type`: Specifies download behavior (default: `fill`):
  - `fill`: Adds images to existing ones to reach target number.
  - `new`: Clears existing images and downloads new ones.
  - `add`: Adds specified number of new images.
- `--device`: Specifies device used for image processing (`cuda` or `cpu`; default: `cpu`):

**Example:**

```bash
mosaic create_example --num_images 200 --download_type fill --device cpu
```

---

## 🧩 Cropping Methods Explained

| Method                  | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| **Smart Crop**          | Crops images without cutting of faces.                          |
| **Center Crop**         | Crops the center area; smaller images get a blurred background. |
| **Aspect Ratio Resize** | Resizes preserving aspect ratio, pads with blurred background.  |
| **Aspect Ratio Crop**   | Crops excess to fit the desired aspect ratio, then resizes.     |
| **Stretch**             | Stretches the image to exactly fit target dimensions.           |

---

## 📝 License

This toolkit is licensed under the [MIT License](/LICENSE.md).
