import os
import platform
import subprocess
import sys

def install_pytorch(cpu_only=False, use_gpu=False):
    if cpu_only:
        print("Installing CPU-only version of PyTorch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    elif use_gpu:
        print("Installing PyTorch with CUDA support...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    else:
        print("Installing default version of PyTorch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])

def install_package():
    print("Cleaning previous builds...")
    subprocess.check_call([sys.executable, "setup.py", "clean"])

    print("Building the package...")
    subprocess.check_call([sys.executable, "setup.py", "sdist", "bdist_wheel"])

    print("Finding the built package...")
    dist_files = [f for f in os.listdir('dist') if f.endswith('.tar.gz')]
    if not dist_files:
        print("No .tar.gz package found in dist folder!")
        return
    
    package_file = os.path.join('dist', dist_files[0])
    print(f"Installing package: {package_file}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_file])

def main():
    os_type = platform.system().lower()
    print(f"Detected operating system: {os_type.capitalize()}")

    # Determine if we want GPU or CPU support
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"

    if mode == "gpu":
        use_gpu = True
        print("GPU support requested.")
    else:
        use_gpu = False
        print("CPU-only version requested.")

    if os_type == "windows":
        print("Detected Windows OS.")
        install_pytorch(cpu_only=(not use_gpu), use_gpu=use_gpu)
        install_package()

    elif os_type == "linux":
        print("Detected Linux OS.")
        install_pytorch(cpu_only=(not use_gpu), use_gpu=use_gpu)
        install_package()

    else:
        print(f"Unsupported OS: {os_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()
