import moderngl
import numpy as np
import os
import cv2  # Asset loading
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
        # 1. Skybox Shader
        self.prog_sky = self._load_program('sky_vs.glsl', 'sky_fs.glsl')
        
        # 2. Ground Shader (Textured) (Replaces Grid)
        self.prog_ground = self._load_program('ground_vs.glsl', 'ground_fs.glsl')
        
        # --- GEOMETRY ---
        # Full Screen Quad (Reusable for Sky, Ground, Post)
        # We need UVs for Post, but Sky/Ground just need positions.
        # Let's verify `ground_vs` uses `in_position` (vec3) and `sky_vs` uses `in_pos` (vec2).
        # To simplify, let's just make a generic Quad VBO with 2D positions for Sky, 3D for Ground?
        # Actually Ground VS expects Z=0 in shader or does it? 
        # Ground VS: "gl_Position = vec4(in_position, 1.0);" and raycasts from -1..1 NDC.
        # So we can just use a fullscreen quad geometry (2D or 3D).
        
        # Generic Quad (NDC -1..1)
        quad_verts = np.array([
            -1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
        ], dtype='f4')
        
        self.vbo_common_quad = self.ctx.buffer(quad_verts.tobytes())
        # Bindings
        self.vao_sky = self.ctx.simple_vertex_array(self.prog_sky, self.vbo_common_quad, 'in_pos') # Sky takes vec2, implies stride check?
        # If sky takes vec2 but buffer has vec3, we need to specify format carefully or just use vec3 in Sky VS.
        # Let's assume I updated Sky VS to take vec3 or just ignore z?
        # My Sky VS said 'in vec2 in_pos'. Using '3f4' buffer with '2f4' input might stride weirdly in simple_vertex_array.
        # Safer to make Sky VS take vec3 or create proper VBO.
        # Let's create a specific Quad for Sky/Ground to be safe.
        
        # Ground VS takes 'in_position' (vec3). Quad above matches.
        self.vao_ground = self.ctx.simple_vertex_array(self.prog_ground, self.vbo_common_quad, 'in_position')
        
        # Sky VS takes 'in_pos' (vec2). 
        # Let's just create a 2D quad buffer for Sky/Post.
        quad_2d = np.array([
            -1.0, 1.0,
            -1.0, -1.0,
            1.0, 1.0,
            1.0, 1.0,
            -1.0, -1.0,
            1.0, -1.0,
        ], dtype='f4')
        self.vbo_quad_2d = self.ctx.buffer(quad_2d.tobytes())
        self.vao_sky = self.ctx.simple_vertex_array(self.prog_sky, self.vbo_quad_2d, 'in_pos')
        
        # --- TEXTURES ---
        self._load_assets()
        
        
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
            raise e
            
    def _load_assets(self):
        """Load Textures"""
        import pygame
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        
        # 1. Sky Texture
        sky_path = os.path.join(assets_dir, 'sky_texture.png')
        if os.path.exists(sky_path):
            try:
                # Load with CV2 to ensure RGB/BGR correctness or Pygame
                # Pygame loads as Surface. Moderngl needs bytes.
                # Use CV2 for consistency
                img = cv2.imread(sky_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # GL expects RGB
                # Flip Y? OpenGL textures usually originate bottom-left.
                # Images usually top-left.
                img = cv2.flip(img, 0) 
                
                self.tex_sky = self.ctx.texture((img.shape[1], img.shape[0]), 3, img.tobytes())
                self.tex_sky.filter = (moderngl.LINEAR, moderngl.LINEAR) # Linear interpolation
                
                # Bind to Texture Unit 2 (0=Screen, 1=Depth)
                self.prog_sky['skyTexture'] = 2 
            except Exception as e:
                print(f"Failed to load sky texture: {e}")
                self.tex_sky = None
        else:
            print("Sky texture not found.")
            self.tex_sky = None
            
        # 2. Ground Texture
        ground_path = os.path.join(assets_dir, 'ground_texture.png')
        if os.path.exists(ground_path):
            try:
                img = cv2.imread(ground_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # No flip needed for top-down map if UVs align? 
                # World X,Y maps to UV.
                
                self.tex_ground = self.ctx.texture((img.shape[1], img.shape[0]), 3, img.tobytes())
                self.tex_ground.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                self.tex_ground.build_mipmaps()
                self.tex_ground.repeat_x = True
                self.tex_ground.repeat_y = True
                
                self.prog_ground['groundTexture'] = 3
            except Exception as e:
                 print(f"Failed to load ground texture: {e}")
                 self.tex_ground = None
        else:
            self.tex_ground = None
            
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
        
        self.ctx.clear(0.5, 0.7, 0.9)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # 1. Render Skybox (Background)
        # Disable Depth Write so Sky is always "behind" everything
        if self.tex_sky:
            self.ctx.disable(moderngl.DEPTH_TEST) # Or glDepthMask(False)
            self.tex_sky.use(location=2)
            self.prog_sky['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
            # Skybox needs special projection matrix? Or same?
            # Standard View matrix has translation. Skybox should center on camera (remove translation).
            # But my shader does "invView" logic. If viewPos is 0,0,0 it works.
            # Effectively, SkyVS uses "invView * ...". If 'm_view' has translation, invView calculates world pos.
            # We want direction. 
            # Vector (0,0,1) * invView_rot_only -> World Dir.
            # My SkyVS logic:
            # vec4 viewPos = invProj * clipPos; viewPos.w = 0;
            # vec3 worldDir = (invView * viewPos).xyz; 
            # Since viewPos.w = 0, the translation part of invView is multiplied by 0.
            # So translation doesn't matter! Correct.
            
            self.prog_sky['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
            self.vao_sky.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST)
            
        # 2. Render Ground (Geometry)
        if self.tex_ground:
            self.tex_ground.use(location=3)
            self.prog_ground['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
            self.prog_ground['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
            self.vao_ground.render(moderngl.TRIANGLES)
        
    def end_frame(self, time=0.0):
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
        
        # Update Time for Noise
        if 'time' in self.prog_post:
            self.prog_post['time'].value = time

        # DoF Parameters (Sabit veya dinamik)
        # Focus distance: Sonsuza odakla (Netlik için)
        self.prog_post['focusDistance'].value = 500.0 
        self.prog_post['focusRange'].value = 1000.0
        self.prog_post['dofEnabled'].value = False # Blur kapalı
        
        # Vignette azalt
        if 'vignetteStrength' in self.prog_post:
            self.prog_post['vignetteStrength'].value = 0.3
        
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
