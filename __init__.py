bl_info = {
    "name": "AI Facial Texture Generator",
    "blender": (3, 0, 0),
    "category": "Material",
    "version": (1, 0, 0),
    "author": "ranfaranfis",
    "description": "Generate UV textures from photos using AI inpainting and facial landmarks",
    "location": "3D Viewport > Sidebar > Face Texture",
    "doc_url": "",
    "tracker_url": "",
}

import bpy
import sys
import os
import subprocess
import importlib

# Add addon directory to Python path
addon_dir = os.path.dirname(__file__)
if addon_dir not in sys.path:
    sys.path.insert(0, addon_dir)

# Required packages
REQUIRED_PACKAGES = [
    "opencv-python",
    "numpy",
    "scipy", 
    "mediapipe",
    "transformers",
    "diffusers",
    "torch",
    "torchvision",
    "Pillow",
    "onnxruntime"
]

def install_package(package):
    """Install required Python packages"""
    try:
        python_exe = bpy.app.binary_path_python
        subprocess.check_call([python_exe, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    return missing

# Import addon modules
from . import operators
from . import panels
from . import properties

def register():
    """Register addon"""
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {missing}")
        print("Installing required packages...")
        for package in missing:
            if install_package(package):
                print(f"✓ Installed {package}")
            else:
                print(f"✗ Failed to install {package}")
    
    # Register classes
    operators.register()
    panels.register()
    properties.register()
    
    print("AI Facial Texture Generator registered successfully!")

def unregister():
    """Unregister addon"""
    operators.unregister()
    panels.unregister() 
    properties.unregister()

if __name__ == "__main__":
    register()