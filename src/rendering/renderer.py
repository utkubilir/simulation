import moderngl
import numpy as np
import os
from src.rendering.camera import GLCamera

class GLRenderer:
    """
    Simülasyon için ModernGL tabanlı OpenGL Render Motoru
    """
    def __init__(self, width: int = 640, height: int = 480):
        # Pygame tarafından oluşturulan OpenGL context'ini bul
        try:
            # Standalone context to avoid conflict with Pygame UI window (which might not be OpenGL)
            self.ctx = moderngl.create_context(standalone=True)
        except Exception as e:
            print(f"Standalone context failed ({e}), attempting to attach to existing context...")
            try:
                self.ctx = moderngl.create_context()
            except Exception as e2:
                print(f"OpenGL Context Hatası (Fallback): {e2}")
                raise e2
            
        # OpenGL Ayarları
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.width = width
        self.height = height
        
        # Kamera Sistemi
        self.camera = GLCamera(fov=60.0, aspect_ratio=width/height)
        
        # --- SHADERS ---
        # 1. Grid Shader
        self.prog_grid = self._load_program('grid_vs.glsl', 'grid_fs.glsl')
        
        # --- GEOMETRY ---
        # Grid için Full Screen Quad (NDC coordinates)
        # Z=1 (far plane), Z=-1 (near plane)
        grid_quad = np.array([
            -1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
        ], dtype='f4')
        
        self.vbo_grid = self.ctx.buffer(grid_quad.tobytes())
        self.vao_grid = self.ctx.simple_vertex_array(self.prog_grid, self.vbo_grid, 'in_position')
        
        
        # 2. Aircraft Shader (Phong) - Reusing box shader logic for now
        self.prog_aircraft = self._load_program('box_vs.glsl', 'box_fs.glsl')
        
        # Aircraft Geometry
        from src.rendering.geometry import GeometryGenerator
        aircraft_data = GeometryGenerator.create_airplane_mesh(scale=1.0)
        
        self.vbo_aircraft = self.ctx.buffer(aircraft_data.tobytes())
        # (Pos 3, Normal 3)
        self.vao_aircraft = self.ctx.simple_vertex_array(self.prog_aircraft, self.vbo_aircraft, 'in_position', 'in_normal')
        
        # --- POST PROCESSING ---
        # 1. Framebuffer (Off-screen render target)
        self.texture = self.ctx.texture((width, height), 3)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_buffer = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(self.texture, self.depth_buffer)
        
        # 1.1 Final Output Framebuffer (Post-Process result will be drawn here)
        # Standalone context has no screen, so we need a destination for the final quad draw.
        self.texture_post = self.ctx.texture((width, height), 3)
        self.texture_post.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_post = self.ctx.framebuffer(self.texture_post)
        
        # 2. Screen Quad Shader
        self.prog_post = self._load_program('screen_vs.glsl', 'post_fs.glsl')
        
        # 3. Screen Quad Geometry
        # X, Y, U, V
        quad_data = np.array([
            -1.0, 1.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, -1.0, 1.0, 0.0,
        ], dtype='f4')
        
        self.vbo_quad = self.ctx.buffer(quad_data.tobytes())
        self.vao_quad = self.ctx.simple_vertex_array(self.prog_post, self.vbo_quad, 'in_pos', 'in_uv')

    def _load_program(self, vs_name, fs_name):
        """Shader programını yükle ve derle"""
        shader_dir = os.path.dirname(__file__) + '/shaders'
        
        try:
            with open(os.path.join(shader_dir, vs_name), 'r') as f:
                vs_source = f.read()
                
            with open(os.path.join(shader_dir, fs_name), 'r') as f:
                fs_source = f.read()
                
            return self.ctx.program(vertex_shader=vs_source, fragment_shader=fs_source)
        except Exception as e:
            print(f"Shader yükleme hatası ({vs_name}, {fs_name}): {e}")
            raise e
        
    def update_camera(self, position, rotation):
        """Kamera durumunu güncelle"""
        self.camera.set_transform(position, rotation)
        
    def render_aircraft(self, position, heading=0.0, roll=0.0, pitch=0.0, color=(1.0, 0.0, 0.0)):
        """
        Belirtilen konumda bir uçak çiz.
        Args:
            position: [x, y, z]
            heading, roll, pitch: in radians (Euler angles)
            color: [r, g, b]
        """
        import pyrr
        
        # Model Matrisi
        # Rotasyon sırası: Roll -> Pitch -> Yaw (Heading)
        rot_r = pyrr.matrix44.create_from_x_rotation(roll, dtype='f4')
        rot_p = pyrr.matrix44.create_from_y_rotation(pitch, dtype='f4')
        rot_y = pyrr.matrix44.create_from_z_rotation(heading, dtype='f4')
        
        rot = pyrr.matrix44.multiply(rot_p, rot_r)
        rot = pyrr.matrix44.multiply(rot_y, rot)
        
        # Translate
        trans = pyrr.matrix44.create_from_translation(position, dtype='f4')
        
        model = pyrr.matrix44.multiply(rot, trans)
        
        self.prog_aircraft['m_model'].write(model.tobytes())
        self.prog_aircraft['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
        self.prog_aircraft['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
        self.prog_aircraft['u_color'].value = tuple(color)
        self.prog_aircraft['viewPos'].value = tuple(self.camera.position)
        
        self.vao_aircraft.render()

    def render(self):
        """Sahne render döngüsü"""
        # --- PASS 1: Off-screen Render ---
        self.fbo.use()
        self.ctx.clear(0.5, 0.7, 0.9) # Sky color
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # 1. Grid Çizimi
        self.prog_grid['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
        self.prog_grid['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
        self.vao_grid.render()
        
        # NOT: render_aircraft çağrıları bu araya girmeli.
        # Ancak render() metodu scene managment yapmadığı için 
        # buradaki tasarımda render_aircraft() çağrıları FBO bağlıyken yapılmalı.
        # Bu yüzden render() metodunu 'begin_frame' ve 'end_frame' olarak ayırmalıyız
        # Veya render() metodu FBO bind/unbind işlemlerini yönetmeli ve draw_queued_objects gibi çalışmalı.
        
        # Design Update:
        # GLRenderer.begin_frame() -> FBO bind
        # ... render_aircraft() ...
        # GLRenderer.end_frame() -> FBO unbind & Post Process Draw
        
    def begin_frame(self):
        """Render döngüsünü başlat (FBO Bind)"""
        self.fbo.use()
        self.ctx.clear(0.5, 0.7, 0.9)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Grid'i her zaman en başta çizelim
        self.prog_grid['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
        self.prog_grid['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
        self.vao_grid.render()
        
    def end_frame(self):
        """Render döngüsünü bitir (Post-Process & Draw to Output FBO)"""
        # --- PASS 2: Post Processing ---
        # Output framebuffer'a geç (Ekran DEĞİL, fbo_post)
        self.fbo_post.use()
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.disable(moderngl.DEPTH_TEST) # Depth test gerekmez, sadece quad çiziyoruz
        
        # Texture'ları bağla
        self.texture.use(location=0)
        self.depth_buffer.use(location=1)
        
        self.prog_post['screenTexture'].value = 0
        self.prog_post['depthTexture'].value = 1
        
        # DoF Parameters (Sabit veya dinamik)
        # Focus distance: Kamera önündeki hedef (örn: 20m)
        self.prog_post['focusDistance'].value = 20.0 
        self.prog_post['focusRange'].value = 30.0
        self.prog_post['dofEnabled'].value = True
        
        # Quad çiz
        self.vao_quad.render(moderngl.TRIANGLE_STRIP)

    def read_pixels(self):
        """
        Oluşturulan görüntüyü CPU'ya (Numpy Array) geri okur.
        OpenCV (BGR) formatına dönüştürür.
        """
        # Read from framebuffer (RGB)
        # Read from FINAL framebuffer (RGB)
        # fbo_post kullanıyoruz çünkü post-process sonucu orada.
        buffer = self.fbo_post.read(components=3, alignment=1)
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape((self.height, self.width, 3))
        
        # Flip Y (OpenGL orijini sol alt, Image sol üst)
        image = np.flipud(image)
        
        # RGB -> BGR (OpenCV için)
        image = image[..., ::-1].copy() # Copy to make it contiguous
        
        return image
