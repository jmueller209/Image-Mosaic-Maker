# Installation Guide

## Prerequisites
- Python 3.10 or higher
- Virtual environment highly recommended to avoid conflicts

## Installation

Clone this directory:
```bash
git clone https://github.com/jmueller209/Image-Mosaic-Maker.git
cd Image-Mosaic-Maker/src
```

To install the toolkit, the following command must be run from the src directory (The directory which contains this file):

Without CUDA support:
```bash
python install.py cpu
```

With CUDA support:

```bash
python install.py gpu
```
