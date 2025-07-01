import bpy
from bpy.props import *
from bpy.types import PropertyGroup

class FaceTextureProperties(PropertyGroup):
    """Properties for the face texture generator"""
    
    # Image properties
    reference_image: StringProperty(
        name="Reference Image",
        description="Path to the reference photo",
        subtype='FILE_PATH'
    )
    
    texture_resolution: IntProperty(
        name="Texture Resolution", 
        description="Output texture resolution",
        default=1024,
        min=512,
        max=4096
    )
    
    # Landmark properties
    auto_detect_landmarks: BoolProperty(
        name="Auto Detect Landmarks",
        description="Automatically detect facial landmarks",
        default=True
    )
    
    landmark_confidence: FloatProperty(
        name="Detection Confidence",
        description="Minimum confidence for landmark detection",
        default=0.5,
        min=0.1,
        max=1.0
    )
    
    # Warping properties
    warping_method: EnumProperty(
        name="Warping Method",
        description="Choose warping algorithm",
        items=[
            ('TPS', 'Thin Plate Spline', 'Smooth global deformation'),
            ('MLS', 'Moving Least Squares', 'Local control warping'),
            ('RBF', 'Radial Basis Function', 'Flexible interpolation')
        ],
        default='TPS'
    )
    
    warping_strength: FloatProperty(
        name="Warping Strength",
        description="Strength of the warping effect",
        default=1.0,
        min=0.1,
        max=2.0
    )
    
    # AI Inpainting properties
    enable_ai_inpainting: BoolProperty(
        name="Enable AI Inpainting",
        description="Use AI to fill missing parts",
        default=True
    )
    
    ai_model: EnumProperty(
        name="AI Model",
        description="Choose AI inpainting model",
        items=[
            ('LAMA', 'LAMA', 'Large Mask Inpainting'),
            ('MAT', 'MAT', 'Mask-Aware Transformer'),
            ('LDM', 'LDM', 'Latent Diffusion Model')
        ],
        default='LAMA'
    )
    
    inpainting_strength: FloatProperty(
        name="Inpainting Strength",
        description="Strength of AI inpainting",
        default=0.8,
        min=0.1,
        max=1.0
    )
    
    # UV properties
    target_object: PointerProperty(
        name="Target Object",
        description="Mesh object to generate texture for",
        type=bpy.types.Object
    )
    
    uv_layer: StringProperty(
        name="UV Layer",
        description="UV layer to use for texture mapping",
        default=""
    )
    
    # Output properties
    output_path: StringProperty(
        name="Output Path",
        description="Path to save generated texture",
        subtype='DIR_PATH',
        default="//"
    )
    
    export_format: EnumProperty(
        name="Export Format",
        description="Texture export format",
        items=[
            ('PNG', 'PNG', 'PNG format'),
            ('TIFF', 'TIFF', 'TIFF format'),
            ('EXR', 'EXR', 'OpenEXR format')
        ],
        default='PNG'
    )
    
    generate_normal_map: BoolProperty(
        name="Generate Normal Map",
        description="Generate normal map from texture details",
        default=False
    )
    
    generate_roughness_map: BoolProperty(
        name="Generate Roughness Map", 
        description="Generate roughness map",
        default=False
    )

def register():
    bpy.utils.register_class(FaceTextureProperties)
    bpy.types.Scene.face_texture = PointerProperty(type=FaceTextureProperties)

def unregister():
    bpy.utils.unregister_class(FaceTextureProperties)
    del bpy.types.Scene.face_texture