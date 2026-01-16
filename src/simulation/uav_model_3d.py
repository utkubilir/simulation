"""
Gelişmiş Sabit Kanatlı İHA 3D Modeli

Detaylı prosedürel geometri:
- Aerodinamik gövde profili
- Kanat airfoil şekli
- Kontrol yüzeyleri (aileron, elevator, rudder)
- İniş takımı
- Motor ve pervane
- Anten ve sensörler
"""

from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode, GeomLines,
    Vec3, Vec4, Point3,
    Material, NodePath, TransparencyAttrib
)
import numpy as np


class DetailedUAVModel:
    """
    Yüksek detaylı sabit kanatlı İHA modeli
    
    Boyutlar (gerçekçi ölçek):
    - Kanat açıklığı: 6m
    - Gövde uzunluğu: 4m
    - Ağırlık temsili: 5kg sınıfı
    """
    
    @staticmethod
    def create(render, name="uav", team="blue", scale=1.0):
        """Detaylı İHA modeli oluştur"""
        root = render.attachNewNode(name)
        
        colors = DetailedUAVModel._get_colors(team)
        
        # === GÖVDE ===
        fuselage = DetailedUAVModel._create_detailed_fuselage(colors)
        fuselage.reparentTo(root)
        
        # === KANATLAR ===
        # Sol kanat
        left_wing = DetailedUAVModel._create_detailed_wing(colors, left=True)
        left_wing.reparentTo(root)
        
        # Sağ kanat
        right_wing = DetailedUAVModel._create_detailed_wing(colors, left=False)
        right_wing.reparentTo(root)
        
        # === KUYRUK ===
        # Yatay stabilizatör
        h_stab = DetailedUAVModel._create_h_stabilizer(colors)
        h_stab.reparentTo(root)
        
        # Dikey stabilizatör (rudder)
        v_stab = DetailedUAVModel._create_v_stabilizer(colors)
        v_stab.reparentTo(root)
        
        # === MOTOR VE PERVANE ===
        motor = DetailedUAVModel._create_motor()
        motor.reparentTo(root)
        motor.setPos(2.2, 0, 0)
        
        propeller = DetailedUAVModel._create_detailed_propeller()
        propeller.reparentTo(root)
        propeller.setPos(2.4, 0, 0)
        
        # === İNİŞ TAKIMI ===
        landing_gear = DetailedUAVModel._create_landing_gear()
        landing_gear.reparentTo(root)
        
        # === KOKPIT/KAMERA ===
        camera_dome = DetailedUAVModel._create_camera_dome()
        camera_dome.reparentTo(root)
        camera_dome.setPos(1.5, 0, -0.15)
        
        # === ANTEN ===
        antenna = DetailedUAVModel._create_antenna()
        antenna.reparentTo(root)
        antenna.setPos(-0.5, 0, 0.3)
        
        # === WINGLET (Kanat uçları) ===
        left_winglet = DetailedUAVModel._create_winglet(colors, left=True)
        left_winglet.reparentTo(root)
        
        right_winglet = DetailedUAVModel._create_winglet(colors, left=False)
        right_winglet.reparentTo(root)
        
        root.setScale(scale)
        return root
    
    @staticmethod
    def _get_colors(team):
        """Takım renk paleti"""
        palettes = {
            'blue': {
                'body_main': (0.15, 0.35, 0.65, 1),
                'body_accent': (0.2, 0.45, 0.8, 1),
                'wing_top': (0.18, 0.4, 0.72, 1),
                'wing_bottom': (0.12, 0.3, 0.55, 1),
                'tail': (0.2, 0.42, 0.75, 1),
                'stripe': (1, 0.8, 0, 1),
            },
            'red': {
                'body_main': (0.7, 0.15, 0.15, 1),
                'body_accent': (0.85, 0.2, 0.2, 1),
                'wing_top': (0.75, 0.18, 0.18, 1),
                'wing_bottom': (0.55, 0.12, 0.12, 1),
                'tail': (0.78, 0.2, 0.2, 1),
                'stripe': (1, 1, 1, 1),
            },
            'green': {
                'body_main': (0.15, 0.55, 0.3, 1),
                'body_accent': (0.2, 0.7, 0.4, 1),
                'wing_top': (0.18, 0.6, 0.35, 1),
                'wing_bottom': (0.12, 0.45, 0.25, 1),
                'tail': (0.2, 0.62, 0.38, 1),
                'stripe': (1, 1, 0.3, 1),
            }
        }
        return palettes.get(team, palettes['blue'])
    
    @staticmethod
    def _create_detailed_fuselage(colors):
        """Detaylı gövde - pürüzsüz profil"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('fuselage', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        # Gövde kesit profili (x, yarıçap, renk_index)
        # 0=main, 1=accent
        profiles = [
            (2.0, 0.02, 1),   # Burun ucu
            (1.8, 0.08, 1),   # Burun konik
            (1.5, 0.15, 1),   # Burun
            (1.0, 0.22, 0),   # Ön gövde
            (0.3, 0.25, 0),   # Orta gövde max
            (-0.3, 0.25, 0),  # Orta gövde
            (-0.8, 0.22, 0),  # Arka gövde
            (-1.3, 0.18, 0),  # Kuyruk geçişi
            (-1.8, 0.12, 1),  # Kuyruk
            (-2.0, 0.08, 1),  # Kuyruk ucu
        ]
        
        segments = 16
        
        # Vertex'ler
        for px, pr, ci in profiles:
            c = colors['body_main'] if ci == 0 else colors['body_accent']
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                y = pr * np.cos(angle)
                z = pr * np.sin(angle)
                
                vertex.addData3(px, y, z)
                normal.addData3(0, np.cos(angle), np.sin(angle))
                color.addData4(*c)
        
        # Üçgenler
        prim = GeomTriangles(Geom.UHStatic)
        n_rings = len(profiles)
        
        for ring in range(n_rings - 1):
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
        
        node = GeomNode('fuselage')
        node.addGeom(geom)
        
        return NodePath(node)
    
    @staticmethod
    def _create_detailed_wing(colors, left=True):
        """Detaylı kanat - airfoil profili"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('wing', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        # Kanat parametreleri
        span = 3.0           # Yarım kanat açıklığı
        root_chord = 0.7     # Kök veter
        tip_chord = 0.35     # Uç veter
        sweep = 0.25         # Sweep
        dihedral = 0.15      # Dihedral
        
        side = 1 if left else -1
        
        # Airfoil profili (NACA 2412 benzeri basitleştirilmiş)
        def airfoil_y(x, chord):
            """Airfoil kalınlık profili"""
            t = 0.12  # %12 kalınlık
            xn = x / chord
            return 5 * t * chord * (0.2969*np.sqrt(xn) - 0.126*xn - 0.3516*xn**2 + 0.2843*xn**3 - 0.1015*xn**4)
        
        # Kanat kesitleri (root'tan tip'e)
        sections = [
            (0, 0, root_chord),        # Kök
            (0.3, side*1.0, 0.6),      # İç
            (0.5, side*2.0, 0.5),      # Orta
            (0.7, side*2.7, 0.4),      # Dış
            (0.9, side*3.0, tip_chord) # Uç
        ]
        
        chord_pts = 8
        
        for ratio, span_pos, chord in sections:
            x_offset = 0.35 - sweep * ratio
            z_base = dihedral * abs(span_pos) * side
            
            # Üst yüzey
            for i in range(chord_pts):
                x_local = chord * i / (chord_pts - 1)
                thickness = airfoil_y(x_local, chord) if i > 0 and i < chord_pts-1 else 0
                
                vertex.addData3(x_offset - x_local, span_pos, z_base + thickness)
                normal.addData3(0, 0, 1)
                color_w.addData4(*colors['wing_top'])
            
            # Alt yüzey
            for i in range(chord_pts):
                x_local = chord * i / (chord_pts - 1)
                thickness = airfoil_y(x_local, chord) * 0.4 if i > 0 and i < chord_pts-1 else 0
                
                vertex.addData3(x_offset - x_local, span_pos, z_base - thickness)
                normal.addData3(0, 0, -1)
                color_w.addData4(*colors['wing_bottom'])
        
        prim = GeomTriangles(Geom.UHStatic)
        
        n_sections = len(sections)
        pts_per_section = chord_pts * 2
        
        for sec in range(n_sections - 1):
            for p in range(chord_pts - 1):
                # Üst yüzey
                c0 = sec * pts_per_section + p
                c1 = sec * pts_per_section + p + 1
                n0 = (sec + 1) * pts_per_section + p
                n1 = (sec + 1) * pts_per_section + p + 1
                
                prim.addVertices(c0, n0, c1)
                prim.addVertices(c1, n0, n1)
                
                # Alt yüzey
                c0 = sec * pts_per_section + chord_pts + p
                c1 = sec * pts_per_section + chord_pts + p + 1
                n0 = (sec + 1) * pts_per_section + chord_pts + p
                n1 = (sec + 1) * pts_per_section + chord_pts + p + 1
                
                prim.addVertices(c0, c1, n0)
                prim.addVertices(c1, n1, n0)
        
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode('wing')
        node.addGeom(geom)
        
        return NodePath(node)
    
    @staticmethod
    def _create_h_stabilizer(colors):
        """Yatay kuyruk"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('h_stab', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        span = 0.9
        chord = 0.35
        thickness = 0.025
        x_pos = -1.7
        
        pts = [
            (x_pos + 0.05, -span, 0),
            (x_pos - chord, -span * 0.9, 0),
            (x_pos + 0.05, span, 0),
            (x_pos - chord, span * 0.9, 0),
        ]
        
        for p in pts:
            vertex.addData3(p[0], p[1], p[2] + thickness)
            normal.addData3(0, 0, 1)
            color_w.addData4(*colors['tail'])
        
        for p in pts:
            vertex.addData3(p[0], p[1], p[2] - thickness)
            normal.addData3(0, 0, -1)
            color_w.addData4(*colors['tail'])
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(1, 3, 2)
        prim.addVertices(4, 6, 5)
        prim.addVertices(5, 6, 7)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('h_stab')).copyTo(NodePath('h_stab'))
    
    @staticmethod
    def _create_v_stabilizer(colors):
        """Dikey kuyruk ve rudder"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('v_stab', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        height = 0.55
        x_pos = -1.65
        
        pts = [
            (x_pos + 0.05, 0, 0.08),
            (x_pos - 0.35, 0, 0.08),
            (x_pos, 0, height),
            (x_pos - 0.2, 0, height),
        ]
        
        for p in pts:
            vertex.addData3(p[0], p[1] - 0.02, p[2])
            normal.addData3(0, -1, 0)
            color_w.addData4(*colors['tail'])
        
        for p in pts:
            vertex.addData3(p[0], p[1] + 0.02, p[2])
            normal.addData3(0, 1, 0)
            color_w.addData4(*colors['tail'])
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 2, 1)
        prim.addVertices(1, 2, 3)
        prim.addVertices(4, 5, 6)
        prim.addVertices(5, 7, 6)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('v_stab')).copyTo(NodePath('v_stab'))
    
    @staticmethod
    def _create_motor():
        """Motor/nacelle"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('motor', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        color = (0.25, 0.25, 0.25, 1)
        segments = 12
        length = 0.2
        radius = 0.08
        
        # Silindir
        for x in [0, length]:
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                y = radius * np.cos(angle)
                z = radius * np.sin(angle)
                
                vertex.addData3(x, y, z)
                normal.addData3(0, np.cos(angle), np.sin(angle))
                color_w.addData4(*color)
        
        prim = GeomTriangles(Geom.UHStatic)
        for i in range(segments):
            c0 = i
            c1 = (i + 1) % segments
            n0 = segments + i
            n1 = segments + (i + 1) % segments
            prim.addVertices(c0, c1, n0)
            prim.addVertices(c1, n1, n0)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('motor')).copyTo(NodePath('motor'))
    
    @staticmethod
    def _create_detailed_propeller():
        """3 kanatlı pervane"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('prop', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        color = (0.15, 0.15, 0.15, 1)
        blades = 3
        length = 0.35
        width = 0.06
        
        for b in range(blades):
            base_angle = b * 2 * np.pi / blades
            
            for r in [0.04, length]:
                for twist in [-1, 1]:
                    angle = base_angle + twist * 0.1
                    y = r * np.cos(angle)
                    z = r * np.sin(angle)
                    w_off = twist * width / 2
                    
                    vertex.addData3(0, y + w_off * np.sin(angle), z - w_off * np.cos(angle))
                    normal.addData3(1, 0, 0)
                    color_w.addData4(*color)
        
        prim = GeomTriangles(Geom.UHStatic)
        for b in range(blades):
            base = b * 4
            prim.addVertices(base, base+1, base+2)
            prim.addVertices(base+1, base+3, base+2)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('prop')).copyTo(NodePath('prop'))
    
    @staticmethod
    def _create_landing_gear():
        """İniş takımı"""
        root = NodePath('landing_gear')
        
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('gear', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        color = (0.3, 0.3, 0.3, 1)
        
        # Ön tekerlek desteği
        gear_pts = [
            # Ön
            (1.2, 0, -0.25), (1.2, 0, -0.4),
            # Sol
            (-0.3, -0.3, -0.25), (-0.3, -0.3, -0.4),
            # Sağ
            (-0.3, 0.3, -0.25), (-0.3, 0.3, -0.4),
        ]
        
        for p in gear_pts:
            vertex.addData3(*p)
            normal.addData3(0, 0, -1)
            color_w.addData4(*color)
        
        prim = GeomLines(Geom.UHStatic)
        prim.addVertices(0, 1)
        prim.addVertices(2, 3)
        prim.addVertices(4, 5)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        np_gear = NodePath(GeomNode('gear_lines'))
        np_gear.node().addGeom(geom)
        np_gear.reparentTo(root)
        
        return root
    
    @staticmethod
    def _create_camera_dome():
        """Kamera kubbesi (gimbal)"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('cam', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        color = (0.1, 0.1, 0.1, 0.8)
        segments = 10
        radius = 0.08
        
        # Yarım küre
        for lat in range(5):
            phi = np.pi / 2 * lat / 4
            r = radius * np.cos(phi)
            z = -radius * np.sin(phi)
            
            for lon in range(segments):
                theta = 2 * np.pi * lon / segments
                y = r * np.cos(theta)
                x = r * np.sin(theta) * 0.5
                
                vertex.addData3(x, y, z)
                normal.addData3(x, y, z)
                color_w.addData4(*color)
        
        prim = GeomTriangles(Geom.UHStatic)
        for lat in range(4):
            for lon in range(segments):
                c0 = lat * segments + lon
                c1 = lat * segments + (lon + 1) % segments
                n0 = (lat + 1) * segments + lon
                n1 = (lat + 1) * segments + (lon + 1) % segments
                
                prim.addVertices(c0, c1, n0)
                prim.addVertices(c1, n1, n0)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        np_cam = NodePath(GeomNode('camera_dome'))
        np_cam.node().addGeom(geom)
        np_cam.setTransparency(TransparencyAttrib.MAlpha)
        
        return np_cam
    
    @staticmethod
    def _create_antenna():
        """GPS/Telemetri anteni"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('ant', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        color = (0.2, 0.2, 0.2, 1)
        
        # Anten direği
        pts = [
            (0, 0, 0), (0, 0, 0.15),
            (-0.03, -0.03, 0.15), (0.03, 0.03, 0.15),
        ]
        
        for p in pts:
            vertex.addData3(*p)
            normal.addData3(0, 0, 1)
            color_w.addData4(*color)
        
        prim = GeomLines(Geom.UHStatic)
        prim.addVertices(0, 1)
        prim.addVertices(2, 3)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('antenna')).copyTo(NodePath('antenna'))
    
    @staticmethod
    def _create_winglet(colors, left=True):
        """Kanat ucu winglet"""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('winglet', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_w = GeomVertexWriter(vdata, 'color')
        
        side = 1 if left else -1
        span = 3.0 * side
        x_base = 0.35 - 0.25 * 0.9
        
        pts = [
            (x_base, span, 0.15 * side),
            (x_base - 0.15, span, 0.15 * side),
            (x_base - 0.05, span, 0.35 * side),
            (x_base - 0.12, span, 0.3 * side),
        ]
        
        for p in pts:
            vertex.addData3(*p)
            normal.addData3(0, side, 0)
            color_w.addData4(*colors['stripe'])
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(1, 3, 2)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        return NodePath(GeomNode('winglet')).copyTo(NodePath('winglet'))
