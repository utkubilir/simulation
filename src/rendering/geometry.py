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
    
    @staticmethod
    def create_box_mesh(width: float = 1.0, height: float = 1.0, depth: float = 1.0):
        """
        Genel amaçlı kutu mesh'i oluşturur.
        
        Args:
            width, height, depth: Kutu boyutları
            
        Returns:
            vertices (np.array): [x, y, z, nx, ny, nz] formatında vertex buffer
        """
        w, h, d = width / 2, height / 2, depth / 2
        
        # 8 Köşe (merkez orijinde)
        p = np.array([
            [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],  # Arka (-Z)
            [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]       # Ön (+Z)
        ])
        
        # Yüz normalleri
        norms = [
            [0, 0, -1],   # Back
            [0, 0, 1],    # Front
            [-1, 0, 0],   # Left
            [1, 0, 0],    # Right
            [0, -1, 0],   # Bottom
            [0, 1, 0]     # Top
        ]
        
        # Yüz indeksleri (CCW winding)
        faces = [
            [0, 2, 1, 0, 3, 2],  # Back
            [4, 5, 6, 4, 6, 7],  # Front
            [0, 4, 7, 0, 7, 3],  # Left
            [1, 2, 6, 1, 6, 5],  # Right
            [0, 1, 5, 0, 5, 4],  # Bottom
            [3, 7, 6, 3, 6, 2]   # Top
        ]
        
        vertices = []
        for face_idx, face in enumerate(faces):
            n = norms[face_idx]
            for vi in face:
                vertices.extend([*p[vi], *n])
        
        return np.array(vertices, dtype='f4')
    
    @staticmethod
    def create_building_mesh(width: float = 30.0, height: float = 50.0, depth: float = 30.0):
        """
        Bina mesh'i oluştur (basit kutu + çatı detayları).
        
        Args:
            width, height, depth: Bina boyutları
            
        Returns:
            vertices (np.array): [x, y, z, nx, ny, nz] formatında vertex buffer
        """
        vertices = []
        
        # Ana bina gövdesi (y=0 tabanda, yukarı uzanır)
        base = GeometryGenerator.create_box_mesh(width, height, depth)
        # Y ekseninde yukarı kaydır (taban y=0 olsun)
        base_reshaped = base.reshape(-1, 6)
        base_reshaped[:, 1] += height / 2  # Y koordinatını kaydır
        vertices.extend(base_reshaped.flatten())
        
        return np.array(vertices, dtype='f4')
    
    @staticmethod
    def create_tree_mesh(trunk_height: float = 8.0, canopy_radius: float = 4.0):
        """
        Basit ağaç mesh'i (silindirik gövde + koni yaprak).
        
        Args:
            trunk_height: Gövde yüksekliği
            canopy_radius: Yaprak tacı yarıçapı
            
        Returns:
            vertices (np.array): [x, y, z, nx, ny, nz] formatında vertex buffer
        """
        vertices = []
        
        # Gövde (ince kutu olarak yaklaşık)
        trunk_width = canopy_radius * 0.15
        trunk = GeometryGenerator.create_box_mesh(trunk_width, trunk_height, trunk_width)
        trunk_reshaped = trunk.reshape(-1, 6)
        trunk_reshaped[:, 1] += trunk_height / 2  # Y yukarı kaydır
        vertices.extend(trunk_reshaped.flatten())
        
        # Yaprak tacı (koni yaklaşımı - çok basit piramit)
        canopy_height = canopy_radius * 2
        base_y = trunk_height
        top_y = base_y + canopy_height
        
        # Piramit - 4 yüzlü
        apex = [0, top_y, 0]
        r = canopy_radius
        corners = [
            [-r, base_y, -r],
            [r, base_y, -r],
            [r, base_y, r],
            [-r, base_y, r]
        ]
        
        # 4 üçgen yüz
        for i in range(4):
            c1 = corners[i]
            c2 = corners[(i + 1) % 4]
            
            # Yüz normali hesapla
            v1 = np.array(c2) - np.array(c1)
            v2 = np.array(apex) - np.array(c1)
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            
            vertices.extend([*c1, *normal])
            vertices.extend([*c2, *normal])
            vertices.extend([*apex, *normal])
        
        # Taban (2 üçgen)
        bottom_normal = [0, -1, 0]
        vertices.extend([*corners[0], *bottom_normal])
        vertices.extend([*corners[2], *bottom_normal])
        vertices.extend([*corners[1], *bottom_normal])
        vertices.extend([*corners[0], *bottom_normal])
        vertices.extend([*corners[3], *bottom_normal])
        vertices.extend([*corners[2], *bottom_normal])
        
        return np.array(vertices, dtype='f4')
    
    @staticmethod
    def create_terrain_mesh(heightmap: np.ndarray, size: tuple, resolution: int = 64):
        """
        Heightmap'ten terrain mesh oluştur.
        
        Args:
            heightmap: 2D numpy array (yükseklik değerleri)
            size: (width, depth) dünya boyutları
            resolution: Mesh çözünürlüğü (vertex sayısı her eksende)
            
        Returns:
            vertices (np.array): [x, y, z, nx, ny, nz] formatında vertex buffer
        """
        from scipy.ndimage import zoom
        
        # Heightmap'i istenen çözünürlüğe ölçekle
        if heightmap.shape[0] != resolution:
            scale_factor = resolution / heightmap.shape[0]
            heightmap = zoom(heightmap, scale_factor, order=1)
        
        vertices = []
        width, depth = size
        
        # Grid oluştur
        for z in range(resolution - 1):
            for x in range(resolution - 1):
                # Dünya koordinatları
                x0 = x * width / (resolution - 1)
                x1 = (x + 1) * width / (resolution - 1)
                z0 = z * depth / (resolution - 1)
                z1 = (z + 1) * depth / (resolution - 1)
                
                # Yükseklikler
                h00 = heightmap[z, x]
                h10 = heightmap[z, x + 1]
                h01 = heightmap[z + 1, x]
                h11 = heightmap[z + 1, x + 1]
                
                # Pozisyonlar (NED: X=North, Y=East, Z=Down - ama görsel için Z=up)
                # OpenGL'de Y yukarı, simülasyonda Z aşağı 
                # Terrain'de Y=height olarak kullanıyoruz
                p00 = [x0, h00, z0]
                p10 = [x1, h10, z0]
                p01 = [x0, h01, z1]
                p11 = [x1, h11, z1]
                
                # Normal hesaplama (cross product - sıra önemli!)
                # OpenGL'de +Y yukarı, normal yukarı bakmalı
                def calc_normal(p1, p2, p3):
                    v1 = np.array(p2) - np.array(p1)
                    v2 = np.array(p3) - np.array(p1)
                    # Cross product sırası: v2 x v1 = yukarı bakan normal (CCW winding)
                    n = np.cross(v2, v1)
                    norm = np.linalg.norm(n)
                    if norm < 1e-6:
                        return np.array([0.0, 1.0, 0.0])  # Default: yukarı
                    return n / norm
                
                # İki üçgen (quad) - CCW winding
                # Tri 1: p00, p10, p11
                n1 = calc_normal(p00, p10, p11)
                vertices.extend([*p00, *n1])
                vertices.extend([*p10, *n1])
                vertices.extend([*p11, *n1])
                
                # Tri 2: p00, p11, p01
                n2 = calc_normal(p00, p11, p01)
                vertices.extend([*p00, *n2])
                vertices.extend([*p11, *n2])
                vertices.extend([*p01, *n2])
        
        return np.array(vertices, dtype='f4')
