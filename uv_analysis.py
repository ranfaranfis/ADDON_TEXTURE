import bpy
import bmesh
import mathutils
import numpy as np
from typing import Dict, List, Tuple, Optional

class UVAnalyzer:
    """Analyze UV mapping of mesh objects"""
    
    def __init__(self):
        self.face_regions = {
            'forehead': [],
            'eyes': [],
            'nose': [],
            'mouth': [],
            'cheeks': [],
            'chin': [],
            'ears': [],
        }
    
    def analyze_mesh(self, obj: bpy.types.Object) -> Dict:
        """
        Analyze UV mapping of a mesh object
        
        Args:
            obj: Blender mesh object
            
        Returns:
            Dictionary containing UV analysis data
        """
        if obj.type != 'MESH':
            raise ValueError("Object must be a mesh")
        
        # Ensure we're in object mode
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Get mesh data
        mesh = obj.data
        
        if not mesh.uv_layers:
            raise ValueError("Mesh has no UV layers")
        
        # Use active UV layer or first available
        uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        
        # Analyze UV coordinates
        uv_data = self.extract_uv_coordinates(mesh, uv_layer)
        
        # Detect facial regions
        facial_regions = self.detect_facial_regions(mesh, uv_data)
        
        # Calculate UV statistics
        uv_stats = self.calculate_uv_statistics(uv_data)
        
        # Find seams and boundaries
        seams = self.find_uv_seams(mesh, uv_layer)
        
        return {
            'uv_coordinates': uv_data,
            'facial_regions': facial_regions,
            'statistics': uv_stats,
            'seams': seams,
            'uv_layer_name': uv_layer.name
        }
    
    def extract_uv_coordinates(self, mesh: bpy.types.Mesh, 
                               uv_layer: bpy.types.MeshUVLoopLayer) -> List[Dict]:
        """Extract UV coordinates for all faces"""
        uv_data = []
        
        # Iterate through all polygons
        for poly in mesh.polygons:
            face_uvs = []
            face_verts = []
            
            # Get UV coordinates for each vertex in the face
            for loop_idx in poly.loop_indices:
                uv = uv_layer.data[loop_idx].uv
                vert_idx = mesh.loops[loop_idx].vertex_index
                vertex = mesh.vertices[vert_idx]
                
                face_uvs.append((uv.x, uv.y))
                face_verts.append(vertex.co[:])
            
            # Calculate face center in UV and world space
            uv_center = self.calculate_center(face_uvs)
            world_center = self.calculate_center(face_verts)
            
            face_data = {
                'polygon_index': poly.index,
                'uv_coordinates': face_uvs,
                'uv_center': uv_center,
                'world_coordinates': face_verts,
                'world_center': world_center,
                'normal': poly.normal[:],
                'area': poly.area
            }
            
            uv_data.append(face_data)
        
        return uv_data
    
    def detect_facial_regions(self, mesh: bpy.types.Mesh, 
                              uv_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Detect different facial regions based on UV layout and vertex positions"""
        regions = {region: [] for region in self.face_regions.keys()}
        
        # Calculate bounding box in world space
        vertices = [v.co for v in mesh.vertices]
        if not vertices:
            return regions
            
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        min_z = min(v[2] for v in vertices)
        max_z = max(v[2] for v in vertices)
        
        width = max_x - min_x
        height = max_y - min_y
        depth = max_z - min_z
        
        # Avoid division by zero
        if width == 0 or height == 0 or depth == 0:
            # Fallback: assign all faces to 'face' region
            regions['face'] = uv_data
            return regions
        
        # Classify faces based on position and normal
        for face_data in uv_data:
            center = face_data['world_center']
            normal = face_data['normal']
            
            # Normalize position relative to bounding box
            rel_x = (center[0] - min_x) / width
            rel_y = (center[1] - min_y) / height
            rel_z = (center[2] - min_z) / depth
            
            # Classify based on position and normal direction
            region = self.classify_face_region(rel_x, rel_y, rel_z, normal)
            regions[region].append(face_data)
        
        return regions
    
    def classify_face_region(self, rel_x: float, rel_y: float, rel_z: float, 
                             normal: Tuple[float, float, float]) -> str:
        """Classify a face into a facial region"""
        # This is a simplified classification - you may need to adjust based on your specific mesh topology
        
        # Front-facing faces (normal pointing forward)
        if normal[1] > 0.3:  # Assuming Y is forward
            # Upper face
            if rel_z > 0.7:
                return 'forehead'
            # Middle face
            elif rel_z > 0.4:
                # Eye region
                if 0.2 < rel_x < 0.8:
                    return 'eyes'
                else:
                    return 'cheeks'
            # Lower middle
            elif rel_z > 0.2:
                # Nose area (center)
                if 0.4 < rel_x < 0.6:
                    return 'nose'
                else:
                    return 'cheeks'
            # Lower face
            else:
                # Mouth area
                if 0.3 < rel_x < 0.7 and rel_z > 0.1:
                    return 'mouth'
                else:
                    return 'chin'
        
        # Side faces
        elif abs(normal[0]) > 0.5:
            # Ears or side of face
            if rel_z > 0.4:
                return 'ears'
            else:
                return 'cheeks'
        
        # Default to cheeks for unclassified faces
        return 'cheeks'
    
    def calculate_uv_statistics(self, uv_data: List[Dict]) -> Dict:
        """Calculate UV mapping statistics"""
        all_uvs = []
        for face_data in uv_data:
            all_uvs.extend(face_data['uv_coordinates'])
        
        if not all_uvs:
            return {}
        
        u_coords = [uv[0] for uv in all_uvs]
        v_coords = [uv[1] for uv in all_uvs]
        
        stats = {
            'uv_count': len(all_uvs),
            'u_range': (min(u_coords), max(u_coords)),
            'v_range': (min(v_coords), max(v_coords)),
            'u_center': sum(u_coords) / len(u_coords),
            'v_center': sum(v_coords) / len(v_coords),
        }
        
        return stats
    
    def find_uv_seams(self, mesh: bpy.types.Mesh, 
                      uv_layer: bpy.types.MeshUVLoopLayer) -> List[Tuple]:
        """Find UV seams in the mesh"""
        seams = []
        
        try:
            # Create bmesh for easier edge analysis
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            
            uv_lay = bm.loops.layers.uv.active
            if not uv_lay:
                bm.free()
                return seams
            
            # Check each edge for UV seams
            for edge in bm.edges:
                if len(edge.link_faces) == 2:
                    # Get UV coordinates for both faces
                    face1, face2 = edge.link_faces
                    
                    # Find corresponding loops
                    loops1 = [l for l in face1.loops if l.edge == edge]
                    loops2 = [l for l in face2.loops if l.edge == edge]
                    
                    if len(loops1) == 2 and len(loops2) == 2:
                        # Check if UV coordinates are different
                        uv1_a = loops1[0][uv_lay].uv
                        uv1_b = loops1[1][uv_lay].uv
                        uv2_a = loops2[0][uv_lay].uv  
                        uv2_b = loops2[1][uv_lay].uv
                        
                        # If UV coordinates don't match, it's a seam
                        threshold = 0.001
                        if (abs(uv1_a.x - uv2_a.x) > threshold or abs(uv1_a.y - uv2_a.y) > threshold or
                            abs(uv1_b.x - uv2_b.x) > threshold or abs(uv1_b.y - uv2_b.y) > threshold):
                            seams.append((edge.verts[0].index, edge.verts[1].index))
            
            bm.free()
        except Exception as e:
            print(f"Error finding UV seams: {e}")
        
        return seams
    
    def calculate_center(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate center point of a list of coordinates"""
        if not points:
            return (0.0, 0.0)
        
        if len(points[0]) == 2:  # 2D points
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
            return (x, y)
        else:  # 3D points
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
            z = sum(p[2] for p in points) / len(points)
            return (x, y, z)