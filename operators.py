import bpy
import bmesh
import mathutils
import numpy as np
import cv2
from bpy.types import Operator
from bpy.props import StringProperty
import os

from .landmarks import FacialLandmarkDetector
from .warping import ImageWarper
from .inpainting import AIInpainter
from .uv_analysis import UVAnalyzer
from .image_utils import ImageProcessor

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
        
        # Get image dimensions
        try:
            import cv2
            image = cv2.imread(bpy.path.abspath(props.reference_image))
            if image is not None:
                image_height, image_width = image.shape[:2]
            else:
                # Fallback dimensions
                image_width, image_height = 1024, 1024
        except:
            # Fallback dimensions
            image_width, image_height = 1024, 1024
        
        # Create empty objects for landmarks
        self.create_landmark_empties(landmarks, image_width, image_height)
        
        self.report({'INFO'}, f"Detected {len(landmarks)} facial landmarks")
        return {'FINISHED'}
    
    def create_landmark_empties(self, landmarks, image_width=1024, image_height=1024):
        """Create empty objects for each landmark"""
        # Clear existing landmarks
        for obj in bpy.data.objects:
            if obj.name.startswith("Landmark_"):
                bpy.data.objects.remove(obj)
        
        # Store image dimensions as custom properties for later reference
        scene = bpy.context.scene
        scene["landmark_image_width"] = image_width
        scene["landmark_image_height"] = image_height
        
        # Create new landmarks with proper scaling
        # Scale down pixel coordinates to reasonable Blender units (0.01 units per pixel)
        scale_factor = 0.01
        for i, (x, y) in enumerate(landmarks):
            # Convert from image coordinates to Blender world coordinates
            world_x = (x - image_width/2) * scale_factor
            world_y = (image_height/2 - y) * scale_factor  # Flip Y axis for Blender
            
            bpy.ops.object.empty_add(type='SPHERE', location=(world_x, world_y, 0))
            empty = bpy.context.active_object
            empty.name = f"Landmark_{i:03d}"
            empty.show_name = True
            empty.empty_display_size = 0.02
            
            # Store original pixel coordinates as custom properties
            empty["pixel_x"] = x
            empty["pixel_y"] = y

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
            # Use stored pixel coordinates if available
            if "pixel_x" in obj and "pixel_y" in obj:
                landmarks.append((obj["pixel_x"], obj["pixel_y"]))
            else:
                # Fallback: convert world coordinates back to pixel coordinates
                scene = bpy.context.scene
                image_width = scene.get("landmark_image_width", 1024)
                image_height = scene.get("landmark_image_height", 1024)
                scale_factor = 0.01
                
                # Convert from world coordinates back to pixel coordinates
                loc = obj.location
                pixel_x = loc.x / scale_factor + image_width/2
                pixel_y = image_height/2 - loc.y / scale_factor
                landmarks.append((pixel_x, pixel_y))
        
        return landmarks
    
    def save_textures(self, image, props):
        """Save generated textures to disk"""
        if not props.output_path:
            return
            
        # Validate input image
        if not isinstance(image, np.ndarray):
            self.report({'ERROR'}, "Invalid image data: expected numpy array")
            return
            
        if len(image.shape) < 2:
            self.report({'ERROR'}, "Invalid image dimensions")
            return
        
        output_path = bpy.path.abspath(props.output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Validate and normalize image format
        if len(image.shape) == 2:
            # Grayscale image - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA image - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            self.report({'ERROR'}, f"Unsupported image format: {image.shape[2]} channels")
            return
        
        # Ensure image values are in valid range
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Save main texture
        filename = f"face_texture.{props.export_format.lower()}"
        filepath = os.path.join(output_path, filename)
        
        # Convert numpy array to Blender image
        height, width = image.shape[:2]
        
        try:
            blender_image = bpy.data.images.new("FaceTexture", width, height)
            
            # Flatten and normalize image data for Blender
            # Blender expects RGBA values in 0.0-1.0 range
            pixels = image.astype(np.float32) / 255.0
            
            # Ensure we have RGB data and add alpha channel
            if len(pixels.shape) == 3 and pixels.shape[2] == 3:
                # Add alpha channel (fully opaque)
                rgba_pixels = np.ones((height, width, 4), dtype=np.float32)
                rgba_pixels[:, :, :3] = pixels
                pixels = rgba_pixels
            
            # Blender expects flattened RGBA data
            blender_image.pixels = pixels.flatten()
            blender_image.filepath_raw = filepath
            blender_image.file_format = props.export_format
            blender_image.save()
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to save texture: {str(e)}")
            return
    
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