import bpy
from bpy.types import Panel

class FACETEX_PT_main_panel(Panel):
    """Main panel for face texture generator"""
    bl_label = "AI Face Texture Generator"
    bl_idname = "FACETEX_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Face Texture"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.face_texture
        
        # Image section
        box = layout.box()
        box.label(text="Reference Image", icon='IMAGE_DATA')
        row = box.row()
        row.prop(props, "reference_image", text="")
        row.operator("facetex.load_image", text="", icon='FILEBROWSER')
        
        # Target object
        box = layout.box()
        box.label(text="Target Mesh", icon='MESH_DATA')
        box.prop(props, "target_object")
        box.prop(props, "texture_resolution")

class FACETEX_PT_landmarks_panel(Panel):
    """Panel for landmark controls"""
    bl_label = "Facial Landmarks"
    bl_idname = "FACETEX_PT_landmarks_panel" 
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Face Texture"
    bl_parent_id = "FACETEX_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.face_texture
        
        # Auto detection
        col = layout.column()
        col.prop(props, "auto_detect_landmarks")
        if props.auto_detect_landmarks:
            col.prop(props, "landmark_confidence", slider=True)
        
        # Detection buttons
        row = layout.row()
        row.operator("facetex.detect_landmarks", icon='ZOOM_SELECTED')
        row.operator("facetex.clear_landmarks", icon='PANEL_CLOSE')
        
        # Show landmark count
        landmark_count = len([obj for obj in bpy.data.objects if obj.name.startswith("Landmark_")])
        layout.label(text=f"Landmarks: {landmark_count}")

class FACETEX_PT_warping_panel(Panel):
    """Panel for warping controls"""
    bl_label = "Image Warping"
    bl_idname = "FACETEX_PT_warping_panel"
    bl_space_type = 'VIEW_3D' 
    bl_region_type = 'UI'
    bl_category = "Face Texture"
    bl_parent_id = "FACETEX_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.face_texture
        
        layout.prop(props, "warping_method")
        layout.prop(props, "warping_strength", slider=True)

class FACETEX_PT_ai_panel(Panel):
    """Panel for AI inpainting controls"""
    bl_label = "AI Inpainting"
    bl_idname = "FACETEX_PT_ai_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI' 
    bl_category = "Face Texture"
    bl_parent_id = "FACETEX_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.face_texture
        
        layout.prop(props, "enable_ai_inpainting")
        
        if props.enable_ai_inpainting:
            layout.prop(props, "ai_model")
            layout.prop(props, "inpainting_strength", slider=True)

class FACETEX_PT_output_panel(Panel):
    """Panel for output controls"""
    bl_label = "Output Settings"
    bl_idname = "FACETEX_PT_output_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Face Texture"
    bl_parent_id = "FACETEX_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.face_texture
        
        layout.prop(props, "output_path")
        layout.prop(props, "export_format")
        
        col = layout.column()
        col.prop(props, "generate_normal_map")
        col.prop(props, "generate_roughness_map")
        
        # Generate button
        layout.separator()
        layout.operator("facetex.generate_texture", text="Generate Texture", icon='TEXTURE')

# Registration
classes = [
    FACETEX_PT_main_panel,
    FACETEX_PT_landmarks_panel,
    FACETEX_PT_warping_panel, 
    FACETEX_PT_ai_panel,
    FACETEX_PT_output_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)