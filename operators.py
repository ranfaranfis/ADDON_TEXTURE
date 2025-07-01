import bpy
import bmesh
import mathutils
import numpy as np
from bpy.types import Operator
from bpy.props import StringProperty
import os

from .utils.landmarks import FacialLandmarkDetector
from .utils.warping import ImageWarper
from .utils.ai_inpainting import AIInpainter
from .utils.uv_analysis import UVAnalyzer
from .utils.image_utils import ImageProcessor

class FACETEX_OT_load_image(Operator):
    """Load reference image"""
    bl_idname = "facetex.load_image"
    bl_label = "Load Reference Image"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(subtype='FILE_PATH')
    
    def execute(self, context):
        scene = context.scene
        scene.face_texture.reference_image = self.filepath
        
        # Initialize image processor
        self.image_processor = ImageProcessor()
        success = self.image_processor.load_image(self.filepath)
        
        if success:
            self.report({'INFO'}, f"Loaded image: {os.path.basename(self.filepath)}")
        else:
            self.report({'ERROR'}, "Failed to load image")
            
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class FACETEX_OT_detect_landmarks(Operator):
    """Detect facial landmarks automatically"""
    bl_idname = "facetex.detect_landmarks"
    bl_label = "Detect Landmarks"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        props = scene.face_texture
        
        if not props.reference_image:
            self.report({'ERROR'}, "No reference image loaded")
            return {'CANCELLED'}
        
        # Initialize landmark detector
        detector = FacialLandmarkDetector()
        landmarks = detector.detect(props.reference_image, props.landmark_confidence)
        
        if landmarks is None:
            self.report({'ERROR'}, "No face detected in image")
            return {'CANCELLED'}
        
        # Create empty objects for landmarks
        self.create_landmark_empties(landmarks)
        
        self.report({'INFO'}, f"Detected {len(landmarks)} facial landmarks")
        return {'FINISHED'}
    
    def create_landmark_empties(self, landmarks):
        """Create empty objects for each landmark"""
        # Clear existing landmarks
        for obj in bpy.data.objects:
            if obj.name.startswith("Landmark_"):
                bpy.data.objects.remove(obj)
        
        # Create new landmarks
        for i, (x, y) in enumerate(landmarks):
            bpy.ops.object.empty_add(type='SPHERE', location=(x/100, y/100, 0))
            empty = bpy.context.active_object
            empty.name = f"Landmark_{i:03d}"
            empty.show_name = True
            empty.empty_display_size = 0.02

class FACETEX_OT_generate_texture(Operator):
    """Generate texture using AI and warping"""
    bl_idname = "facetex.generate_texture"
    bl_label = "Generate Texture" 
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        props = scene.face_texture
        
        # Validation
        if not props.reference_image:
            self.report({'ERROR'}, "No reference image loaded")
            return {'CANCELLED'}
            
        if not props.target_object:
            self.report({'ERROR'}, "No target object selected")
            return {'CANCELLED'}
        
        # Get landmarks from empty objects
        landmarks = self.get_landmarks_from_empties()
        if not landmarks:
            self.report({'ERROR'}, "No landmarks found. Run landmark detection first.")
            return {'CANCELLED'}
        
        try:
            # Step 1: Analyze UV mapping
            self.report({'INFO'}, "Analyzing UV mapping...")
            uv_analyzer = UVAnalyzer()
            uv_data = uv_analyzer.analyze_mesh(props.target_object)
            
            # Step 2: Warp image based on landmarks
            self.report({'INFO'}, "Warping image...")
            warper = ImageWarper(method=props.warping_method)
            warped_image, mask = warper.warp_image(
                props.reference_image, 
                landmarks, 
                uv_data,
                strength=props.warping_strength
            )
            
            # Step 3: AI inpainting for missing parts
            if props.enable_ai_inpainting:
                self.report({'INFO'}, "Running AI inpainting...")
                inpainter = AIInpainter(model=props.ai_model)
                final_image = inpainter.inpaint(
                    warped_image, 
                    mask, 
                    strength=props.inpainting_strength
                )
            else:
                final_image = warped_image
            
            # Step 4: Generate additional maps
            image_processor = ImageProcessor()
            
            if props.generate_normal_map:
                self.report({'INFO'}, "Generating normal map...")
                normal_map = image_processor.generate_normal_map(final_image)
            
            if props.generate_roughness_map:
                self.report({'INFO'}, "Generating roughness map...")
                roughness_map = image_processor.generate_roughness_map(final_image)
            
            # Step 5: Save textures
            self.save_textures(final_image, props)
            
            # Step 6: Apply to material
            self.apply_to_material(props.target_object, final_image)
            
            self.report({'INFO'}, "Texture generation completed!")
            
        except Exception as e:
            self.report({'ERROR'}, f"Error generating texture: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def get_landmarks_from_empties(self):
        """Extract landmark positions from empty objects"""
        landmarks = []
        landmark_objects = [obj for obj in bpy.data.objects if obj.name.startswith("Landmark_")]
        landmark_objects.sort(key=lambda x: x.name)
        
        for obj in landmark_objects:
            loc = obj.location
            landmarks.append((loc.x * 100, loc.y * 100))  # Scale back
        
        return landmarks
    
    def save_textures(self, image, props):
        """Save generated textures to disk"""
        if not props.output_path:
            return
        
        output_path = bpy.path.abspath(props.output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Save main texture
        filename = f"face_texture.{props.export_format.lower()}"
        filepath = os.path.join(output_path, filename)
        
        # Convert numpy array to Blender image
        height, width = image.shape[:2]
        blender_image = bpy.data.images.new("FaceTexture", width, height)
        
        # Flatten and normalize image data
        pixels = image.flatten() / 255.0
        if len(pixels) == width * height * 3:  # RGB
            # Add alpha channel
            rgba_pixels = []
            for i in range(0, len(pixels), 3):
                rgba_pixels.extend([pixels[i], pixels[i+1], pixels[i+2], 1.0])
            pixels = rgba_pixels
        
        blender_image.pixels = pixels
        blender_image.filepath_raw = filepath
        blender_image.file_format = props.export_format
        blender_image.save()
    
    def apply_to_material(self, target_obj, texture_image):
        """Apply generated texture to object material"""
        # Ensure object has material
        if not target_obj.data.materials:
            mat = bpy.data.materials.new("FaceTextureMat")
            target_obj.data.materials.append(mat)
        else:
            mat = target_obj.data.materials[0]
        
        # Setup material nodes
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Create nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        principled = nodes.new('ShaderNodeBsdfPrincipled') 
        tex_image = nodes.new('ShaderNodeTexImage')
        
        # Load image into texture node
        height, width = texture_image.shape[:2]
        if "FaceTexture" in bpy.data.images:
            tex_image.image = bpy.data.images["FaceTexture"]
        
        # Connect nodes
        links = mat.node_tree.links
        links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

class FACETEX_OT_clear_landmarks(Operator):
    """Clear all landmark empties"""
    bl_idname = "facetex.clear_landmarks" 
    bl_label = "Clear Landmarks"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Remove all landmark objects
        landmark_objects = [obj for obj in bpy.data.objects if obj.name.startswith("Landmark_")]
        for obj in landmark_objects:
            bpy.data.objects.remove(obj)
        
        self.report({'INFO'}, f"Cleared {len(landmark_objects)} landmarks")
        return {'FINISHED'}

# Registration
classes = [
    FACETEX_OT_load_image,
    FACETEX_OT_detect_landmarks, 
    FACETEX_OT_generate_texture,
    FACETEX_OT_clear_landmarks,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)