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
    # Map package names to their import names
    package_import_map = {
        "opencv-python": "cv2",
        "numpy": "numpy", 
        "scipy": "scipy",
        "mediapipe": "mediapipe",
        "transformers": "transformers",
        "diffusers": "diffusers", 
        "torch": "torch",
        "torchvision": "torchvision",
        "Pillow": "PIL",
        "onnxruntime": "onnxruntime"
    }
    
    missing = []
    for package in REQUIRED_PACKAGES:
        import_name = package_import_map.get(package, package.replace('-', '_'))
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    return missing

# Import addon modules
try:
    from . import operators
    from . import panels
    from . import properties
except ImportError as e:
    print(f"Error importing addon modules: {e}")
    print("Some functionality may be limited due to missing dependencies")
    # Create dummy modules to prevent complete failure
    class DummyModule:
        def register(self): pass
        def unregister(self): pass
    
    if 'operators' not in locals():
        operators = DummyModule()
    if 'panels' not in locals():
        panels = DummyModule()
    if 'properties' not in locals():
        properties = DummyModule()

def register():
    """Register addon"""
    try:
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
        
    except Exception as e:
        print(f"Error during addon registration: {e}")
        print("The addon may not function correctly due to missing dependencies or other issues")

def unregister():
    """Unregister addon"""
    try:
        operators.unregister()
        panels.unregister() 
        properties.unregister()
    except Exception as e:
        print(f"Error during addon unregistration: {e}")

if __name__ == "__main__":
    register()