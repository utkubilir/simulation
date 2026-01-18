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
    def create_terrain(render, size=2000, resolution=200):
        """
        Ana arazi oluştur
        WorldMap kullanarak görsel mesh üretir.
        """
        from src.simulation.map_data import WorldMap

        root = render.attachNewNode("terrain")
        
        # Zemin mesh
        ground = TerrainGenerator._create_height_map(size, resolution)
        ground.reparentTo(root)
        
        # Pist
        runway = TerrainGenerator._create_runway()
        runway.reparentTo(root)
        # Pist merkezi MapData'dan alınır
        rc = WorldMap.RUNWAY_CENTER
        runway.setPos(rc[0], rc[1], 0.1)
        
        # Binalar (MapData'dan)
        buildings = TerrainGenerator._create_buildings()
        buildings.reparentTo(root)
        
        # Dağlar ve Ağaçlar artık Heightmap veya MapData'dan yönetiliyor.
        # Ayrı mesh olarak eklenmiyorlar (Z-fighting önlemek için).
        
        return root
    
    @staticmethod
    def _create_height_map(size, resolution):
        """Yükseklik haritası ile zemin"""
        from src.simulation.map_data import WorldMap

        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        step = size / resolution
        # 0..2000 sistemi
        
        heights = np.zeros((resolution + 1, resolution + 1))
        
        # Yükseklikleri hesapla (WorldMap'ten)
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = i * step
                y = j * step
                h = WorldMap.get_terrain_height(x, y)
                heights[i, j] = h
        
        # Vertex'ler
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = i * step
                y = j * step
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
        """Binalar (MapData'dan)"""
        from src.simulation.map_data import WorldMap
        
        root = NodePath("buildings")
        
        for obj in WorldMap.STATIC_OBJECTS:
            if obj.obj_type == "building":
                w, d, h = obj.size
                c = obj.color
                building = TerrainGenerator._create_box_building(w, d, h, c)
                building.reparentTo(root)
                building.setPos(*obj.position)
            
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
    
