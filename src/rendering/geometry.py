import numpy as np

class GeometryGenerator:
    """
    Prosedürel 3D Geometri Oluşturucu.
    Basit uçak, küp, küre vb. mesh verilerini (vertex + normal) üretir.
    """
    
    @staticmethod
    def create_airplane_mesh(scale=1.0):
        """
        Basit bir uçak geometrisi oluşturur (Gövde, Kanatlar, Kuyruk).
        
        Returns:
            vertices (np.array): [x, y, z, nx, ny, nz] formatında vertex buffer
        """
        vertices = []
        
        # Renkler (Materyal ID gibi kullanılabilir ama şimdilik renk yok, shader uniform kullanacak)
        # Sadece Pos (3) + Normal (3) = 6 float per vertex
        
        # --- Helper: Kutu Ekleme ---
        def add_box(center, size, rot_y=0.0):
            # Basit Kutu (Cube)
            # Center: [x,y,z], Size: [w,h,d]
            cx, cy, cz = center
            w, h, d = size[0]/2, size[1]/2, size[2]/2
            
            # 8 Köşe
            p = np.array([
                [cx-w, cy-h, cz-d], [cx+w, cy-h, cz-d], [cx+w, cy+h, cz-d], [cx-w, cy+h, cz-d], # Arka (-Z)
                [cx-w, cy-h, cz+d], [cx+w, cy-h, cz+d], [cx+w, cy+h, cz+d], [cx-w, cy+h, cz+d]  # Ön (+Z)
            ])
            
            # Normaller
            norms = [
                [0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]
            ]
            
            # Yüzeyler (Quad -> 2 Triangles)
            indices = [
                [0, 1, 2, 3], # Back
                [4, 5, 6, 7], # Front (Reverse order for CCW) -> [7,6,5,4]
                [0, 4, 7, 3], # Left
                [1, 5, 6, 2], # Right
                [0, 1, 5, 4], # Bottom
                [3, 2, 6, 7]  # Top
            ]
            
            # Front face order fix
            indices[1] = [5, 4, 7, 6]
             # Bottom fix
            indices[4] = [4, 5, 1, 0]
            
            box_verts = []
            
            for face_idx, face in enumerate(indices):
                n = norms[face_idx]
                # Quad -> Tri 1
                box_verts.extend([*p[face[0]], *n])
                box_verts.extend([*p[face[1]], *n])
                box_verts.extend([*p[face[2]], *n])
                # Quad -> Tri 2
                box_verts.extend([*p[face[0]], *n])
                box_verts.extend([*p[face[2]], *n])
                box_verts.extend([*p[face[3]], *n])
                
            return box_verts

        # 1. Gövde (Fuselage) - Uzun kutu
        # X+ yönü ileri
        vertices.extend(add_box([0, 0, 0], [2.0, 0.4, 0.4]))
        
        # 2. Kanatlar (Wings)
        vertices.extend(add_box([0.2, 0, 0], [0.6, 2.4, 0.1]))
        
        # 3. Kuyruk (Tail - Vertical)
        vertices.extend(add_box([-0.8, 0, 0.4], [0.4, 0.1, 0.6]))
        
        # 4. Kuyruk (Tail - Horizontal)
        vertices.extend(add_box([-0.8, 0, 0], [0.4, 1.0, 0.1]))
        
        return np.array(vertices, dtype='f4') * scale
