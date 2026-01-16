"""
3D Arazi Oluşturucu

Prosedürel arazi:
- Perlin noise ile tepeler/dağlar
- Binalar
- Pist
- Ağaçlar
"""

from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode, GeomLines,
    Vec3, Vec4, NodePath, CardMaker
)
import numpy as np


class TerrainGenerator:
    """Prosedürel arazi oluşturucu"""
    
    @staticmethod
    def create_terrain(render, size=2000, resolution=50):
        """
        Ana arazi oluştur
        
        Args:
            render: Panda3D render node
            size: Arazi boyutu (metre)
            resolution: Grid çözünürlüğü
        """
        root = render.attachNewNode("terrain")
        
        # Zemin mesh
        ground = TerrainGenerator._create_height_map(size, resolution)
        ground.reparentTo(root)
        
        # Dağlar
        mountains = TerrainGenerator._create_mountains()
        mountains.reparentTo(root)
        
        # Pist
        runway = TerrainGenerator._create_runway()
        runway.reparentTo(root)
        runway.setPos(500, 500, 0.1)
        
        # Binalar
        buildings = TerrainGenerator._create_buildings()
        buildings.reparentTo(root)
        
        # Ağaçlar
        trees = TerrainGenerator._create_trees()
        trees.reparentTo(root)
        
        return root
    
    @staticmethod
    def _perlin_noise(x, y, seed=42):
        """Basit Perlin benzeri noise"""
        np.random.seed(seed)
        
        # Çoklu frekans
        noise = 0
        amplitude = 1
        frequency = 0.002
        
        for _ in range(4):
            noise += amplitude * np.sin(x * frequency + np.random.rand() * 100) * \
                     np.cos(y * frequency + np.random.rand() * 100)
            amplitude *= 0.5
            frequency *= 2
            
        return noise
    
    @staticmethod
    def _create_height_map(size, resolution):
        """Yükseklik haritası ile zemin"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        step = size / resolution
        half = size / 2
        
        heights = np.zeros((resolution + 1, resolution + 1))
        
        # Yükseklikleri hesapla
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = -half + i * step
                y = -half + j * step
                
                # Merkeze yakın düz (pist için)
                dist_center = np.sqrt((x - 500)**2 + (y - 500)**2)
                if dist_center < 300:
                    h = 0
                else:
                    # Tepe/dağ yükseklikleri
                    h = TerrainGenerator._perlin_noise(x, y) * 30
                    h += TerrainGenerator._perlin_noise(x * 2, y * 2, seed=123) * 15
                    
                    # Uzaklara doğru dağlar yükselt
                    dist_edge = min(abs(x + half), abs(x - half), abs(y + half), abs(y - half))
                    if dist_edge < 400:
                        h += (400 - dist_edge) * 0.15
                        
                heights[i, j] = max(0, h)
        
        # Vertex'ler
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = -half + i * step
                y = -half + j * step
                z = heights[i, j]
                
                vertex.addData3(x, y, z)
                
                # Normal hesapla
                if 0 < i < resolution and 0 < j < resolution:
                    dzdx = (heights[i+1, j] - heights[i-1, j]) / (2 * step)
                    dzdy = (heights[i, j+1] - heights[i, j-1]) / (2 * step)
                    n = Vec3(-dzdx, -dzdy, 1).normalized()
                    normal.addData3(n)
                else:
                    normal.addData3(0, 0, 1)
                
                # Renk (yüksekliğe göre)
                if z < 5:
                    c = (0.25, 0.5, 0.2, 1)      # Yeşil çimen
                elif z < 20:
                    c = (0.35, 0.55, 0.25, 1)    # Açık yeşil
                elif z < 40:
                    c = (0.45, 0.4, 0.3, 1)      # Kahverengi
                else:
                    c = (0.6, 0.6, 0.6, 1)       # Gri (kaya)
                    
                color.addData4(*c)
        
        # Üçgenler
        prim = GeomTriangles(Geom.UHStatic)
        
        for i in range(resolution):
            for j in range(resolution):
                v0 = i * (resolution + 1) + j
                v1 = v0 + 1
                v2 = v0 + (resolution + 1)
                v3 = v2 + 1
                
                prim.addVertices(v0, v2, v1)
                prim.addVertices(v1, v2, v3)
        
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode('ground')
        node.addGeom(geom)
        
        return NodePath(node)
    
    @staticmethod
    def _create_mountains():
        """Büyük dağ tepeleri"""
        root = NodePath("mountains")
        
        # Birkaç büyük dağ
        mountain_positions = [
            (-800, -600, 0, 150),   # x, y, z_base, height
            (-600, 800, 0, 120),
            (900, -700, 0, 180),
            (700, 900, 0, 140),
            (-900, 200, 0, 100),
        ]
        
        for mx, my, mz, height in mountain_positions:
            mountain = TerrainGenerator._create_cone_mountain(height)
            mountain.reparentTo(root)
            mountain.setPos(mx, my, mz)
            
        return root
    
    @staticmethod
    def _create_cone_mountain(height, radius=None):
        """Konik dağ"""
        if radius is None:
            radius = height * 1.5
            
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('mountain', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        segments = 16
        rings = 8
        
        for ring in range(rings + 1):
            t = ring / rings
            r = radius * (1 - t * 0.9)  # Daralan yarıçap
            z = height * t
            
            # Renk
            if t < 0.3:
                c = (0.3, 0.5, 0.25, 1)   # Yeşil
            elif t < 0.6:
                c = (0.5, 0.4, 0.3, 1)    # Kahverengi
            elif t < 0.85:
                c = (0.55, 0.5, 0.45, 1)  # Açık kahve
            else:
                c = (0.9, 0.9, 0.95, 1)   # Kar
                
            for seg in range(segments):
                angle = 2 * np.pi * seg / segments
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                
                # Düzensizlik ekle
                noise = np.sin(angle * 5 + z * 0.1) * radius * 0.1
                x += noise * np.cos(angle)
                y += noise * np.sin(angle)
                
                vertex.addData3(x, y, z)
                
                nx = np.cos(angle)
                ny = np.sin(angle)
                nz = radius / height * 0.5
                n = Vec3(nx, ny, nz).normalized()
                normal.addData3(n)
                
                color.addData4(*c)
        
        prim = GeomTriangles(Geom.UHStatic)
        
        for ring in range(rings):
            for seg in range(segments):
                c0 = ring * segments + seg
                c1 = ring * segments + (seg + 1) % segments
                n0 = (ring + 1) * segments + seg
                n1 = (ring + 1) * segments + (seg + 1) % segments
                
                prim.addVertices(c0, c1, n0)
                prim.addVertices(c1, n1, n0)
        
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('mountain')).copyTo(NodePath('m'))
    
    @staticmethod
    def _create_runway():
        """Pist"""
        root = NodePath("runway")
        
        # Ana pist
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('runway', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        length = 400
        width = 30
        
        runway_color = (0.3, 0.3, 0.35, 1)
        
        pts = [
            (-length/2, -width/2, 0),
            (length/2, -width/2, 0),
            (-length/2, width/2, 0),
            (length/2, width/2, 0),
        ]
        
        for p in pts:
            vertex.addData3(*p)
            normal.addData3(0, 0, 1)
            color.addData4(*runway_color)
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(1, 3, 2)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        np_runway = NodePath(GeomNode('runway_surface'))
        np_runway.node().addGeom(geom)
        np_runway.reparentTo(root)
        
        # Pist çizgileri
        lines = TerrainGenerator._create_runway_markings(length, width)
        lines.reparentTo(root)
        lines.setZ(0.05)
        
        return root
    
    @staticmethod
    def _create_runway_markings(length, width):
        """Pist işaretleri"""
        from panda3d.core import LineSegs
        
        lines = LineSegs()
        lines.setColor(1, 1, 1, 1)
        lines.setThickness(3)
        
        # Orta çizgi (kesikli)
        for i in range(-int(length/2) + 10, int(length/2) - 10, 20):
            lines.moveTo(i, 0, 0)
            lines.drawTo(i + 10, 0, 0)
        
        # Kenar çizgileri
        lines.moveTo(-length/2, -width/2 + 1, 0)
        lines.drawTo(length/2, -width/2 + 1, 0)
        lines.moveTo(-length/2, width/2 - 1, 0)
        lines.drawTo(length/2, width/2 - 1, 0)
        
        # Threshold çizgileri
        for offset in range(-12, 13, 4):
            lines.moveTo(-length/2 + 5, offset, 0)
            lines.drawTo(-length/2 + 25, offset, 0)
            lines.moveTo(length/2 - 5, offset, 0)
            lines.drawTo(length/2 - 25, offset, 0)
        
        return NodePath(lines.create())
    
    @staticmethod
    def _create_buildings():
        """Binalar"""
        root = NodePath("buildings")
        
        # Bina konumları ve boyutları
        building_data = [
            # Kontrol kulesi
            (600, 450, 0, 8, 8, 25, (0.5, 0.5, 0.55, 1)),
            # Hangarlar
            (650, 520, 0, 40, 25, 12, (0.6, 0.55, 0.5, 1)),
            (650, 560, 0, 40, 25, 12, (0.55, 0.5, 0.45, 1)),
            # Depolar
            (700, 480, 0, 20, 15, 8, (0.5, 0.45, 0.4, 1)),
            (720, 520, 0, 15, 15, 6, (0.45, 0.4, 0.38, 1)),
        ]
        
        for x, y, z, w, d, h, c in building_data:
            building = TerrainGenerator._create_box_building(w, d, h, c)
            building.reparentTo(root)
            building.setPos(x, y, z)
            
        return root
    
    @staticmethod
    def _create_box_building(width, depth, height, color):
        """Kutu bina"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('building', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        w, d, h = width/2, depth/2, height
        
        # Köşeler ve normaller (points_list, normal_tuple)
        faces = [
            # Ön
            ([(-w, -d, 0), (w, -d, 0), (w, -d, h), (-w, -d, h)], (0, -1, 0)),
            # Arka
            ([(w, d, 0), (-w, d, 0), (-w, d, h), (w, d, h)], (0, 1, 0)),
            # Sol
            ([(-w, d, 0), (-w, -d, 0), (-w, -d, h), (-w, d, h)], (-1, 0, 0)),
            # Sağ
            ([(w, -d, 0), (w, d, 0), (w, d, h), (w, -d, h)], (1, 0, 0)),
            # Üst
            ([(-w, -d, h), (w, -d, h), (w, d, h), (-w, d, h)], (0, 0, 1)),
        ]
        
        roof_color = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, 1)
        
        for pts, n in faces:
            c = roof_color if n[2] == 1 else color
            for p in pts:
                vertex.addData3(*p)
                normal.addData3(*n)
                color_w.addData4(*c)
        
        prim = GeomTriangles(Geom.UHStatic)
        for face in range(5):
            base = face * 4
            prim.addVertices(base, base+1, base+2)
            prim.addVertices(base, base+2, base+3)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('building')).copyTo(NodePath('b'))
    
    @staticmethod
    def _create_trees():
        """Ağaçlar"""
        root = NodePath("trees")
        
        np.random.seed(42)
        
        # Rastgele ağaç konumları (pistden uzakta)
        for _ in range(100):
            x = np.random.uniform(-800, 1200)
            y = np.random.uniform(-800, 1200)
            
            # Pist ve bina alanını atla
            if 200 < x < 800 and 300 < y < 700:
                continue
                
            scale = np.random.uniform(0.7, 1.3)
            tree = TerrainGenerator._create_simple_tree(scale)
            tree.reparentTo(root)
            tree.setPos(x, y, 0)
            
        return root
    
    @staticmethod
    def _create_simple_tree(scale=1.0):
        """Basit ağaç (koni + silindir)"""
        root = NodePath("tree")
        
        format = GeomVertexFormat.get_v3n3c4()
        
        # Gövde
        trunk_vdata = GeomVertexData('trunk', format, Geom.UHStatic)
        tv = GeomVertexWriter(trunk_vdata, 'vertex')
        tn = GeomVertexWriter(trunk_vdata, 'normal')
        tc = GeomVertexWriter(trunk_vdata, 'color')
        
        trunk_color = (0.4, 0.25, 0.15, 1)
        trunk_h = 3 * scale
        trunk_r = 0.3 * scale
        
        for h in [0, trunk_h]:
            for i in range(8):
                angle = 2 * np.pi * i / 8
                tv.addData3(trunk_r * np.cos(angle), trunk_r * np.sin(angle), h)
                tn.addData3(np.cos(angle), np.sin(angle), 0)
                tc.addData4(*trunk_color)
        
        trunk_prim = GeomTriangles(Geom.UHStatic)
        for i in range(8):
            trunk_prim.addVertices(i, (i+1)%8, 8+i)
            trunk_prim.addVertices((i+1)%8, 8+(i+1)%8, 8+i)
        trunk_prim.closePrimitive()
        
        trunk_geom = Geom(trunk_vdata)
        trunk_geom.addPrimitive(trunk_prim)
        trunk_np = NodePath(GeomNode('trunk'))
        trunk_np.node().addGeom(trunk_geom)
        trunk_np.reparentTo(root)
        
        # Yapraklar (koni)
        leaf_vdata = GeomVertexData('leaves', format, Geom.UHStatic)
        lv = GeomVertexWriter(leaf_vdata, 'vertex')
        ln = GeomVertexWriter(leaf_vdata, 'normal')
        lc = GeomVertexWriter(leaf_vdata, 'color')
        
        leaf_color = (0.15, 0.45, 0.2, 1)
        leaf_h = 6 * scale
        leaf_r = 2 * scale
        
        # Tepe noktası
        lv.addData3(0, 0, trunk_h + leaf_h)
        ln.addData3(0, 0, 1)
        lc.addData4(*leaf_color)
        
        # Taban
        for i in range(12):
            angle = 2 * np.pi * i / 12
            lv.addData3(leaf_r * np.cos(angle), leaf_r * np.sin(angle), trunk_h)
            ln.addData3(np.cos(angle), np.sin(angle), 0.3)
            lc.addData4(*leaf_color)
        
        leaf_prim = GeomTriangles(Geom.UHStatic)
        for i in range(12):
            leaf_prim.addVertices(0, 1+i, 1+(i+1)%12)
        leaf_prim.closePrimitive()
        
        leaf_geom = Geom(leaf_vdata)
        leaf_geom.addPrimitive(leaf_prim)
        leaf_np = NodePath(GeomNode('leaves'))
        leaf_np.node().addGeom(leaf_geom)
        leaf_np.reparentTo(root)
        
        return root
