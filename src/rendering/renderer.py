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
        
        # 3. Instanced Aircraft Shader
        self.prog_instanced = self._load_program('instanced_vs.glsl', 'box_fs.glsl')
        
        # Aircraft Geometry
        from src.rendering.geometry import GeometryGenerator
        aircraft_data = GeometryGenerator.create_airplane_mesh(scale=1.0)
        
        self.vbo_aircraft = self.ctx.buffer(aircraft_data.tobytes())
        # (Pos 3, Normal 3)
        self.vao_aircraft = self.ctx.simple_vertex_array(self.prog_aircraft, self.vbo_aircraft, 'in_position', 'in_normal')
        
        # --- INSTANCING SETUP ---
        # Max instances buffer (Matrix 16f + Color 3f = 19 floats per instance)
        self.max_instances = 4096
        self.instance_data_size = 19 * 4 # bytes per instance
        self.vbo_instance = self.ctx.buffer(reserve=self.max_instances * self.instance_data_size)
        
        # Instanced VAO
        # Note: '16f 3f /i' means 16 floats and 3 floats per instance
        self.vao_aircraft_instanced = self.ctx.vertex_array(self.prog_instanced, [
            (self.vbo_aircraft, '3f 3f', 'in_position', 'in_normal'),
            (self.vbo_instance, '16f 3f /i', 'in_instanceMatrix', 'in_instanceColor')
        ])
        
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
        
        # --- BLOOM PIPELINE ---
        # Bloom extraction FBO (half resolution for performance)
        bloom_w, bloom_h = width // 2, height // 2
        self.texture_bloom_bright = self.ctx.texture((bloom_w, bloom_h), 3)
        self.texture_bloom_bright.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_bloom_bright = self.ctx.framebuffer(self.texture_bloom_bright)
        
        # Bloom blur ping-pong FBOs
        self.texture_bloom_blur1 = self.ctx.texture((bloom_w, bloom_h), 3)
        self.texture_bloom_blur1.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_bloom_blur1 = self.ctx.framebuffer(self.texture_bloom_blur1)
        
        self.texture_bloom_blur2 = self.ctx.texture((bloom_w, bloom_h), 3)
        self.texture_bloom_blur2.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_bloom_blur2 = self.ctx.framebuffer(self.texture_bloom_blur2)
        
        # Bloom shaders
        self.prog_bloom_bright = self._load_program('screen_vs.glsl', 'bloom_bright_fs.glsl')
        self.prog_blur = self._load_program('screen_vs.glsl', 'blur_fs.glsl')
        self.prog_bloom_combine = self._load_program('screen_vs.glsl', 'bloom_combine_fs.glsl')
        
        self.bloom_enabled = True
        self.bloom_intensity = 0.5
        self.bloom_threshold = 0.8
        
        # --- SHADOW MAPPING SETUP ---
        self.shadow_res = 2048
        self.tex_shadow = self.ctx.depth_texture((self.shadow_res, self.shadow_res))
        self.tex_shadow.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_shadow.compare_func = '>' # Needed for shadow comparison? Usually '' or 'func'
        # ModernGL depth sampling defaults to (D < R ? 1 : 0) if compare_func set?
        # Let's keep default and use sampler2DShadow in GLSL or standard sampler2D.
        # Im using sampler2D in shader with custom comparison.
        self.fbo_shadow = self.ctx.framebuffer(depth_attachment=self.tex_shadow)
        
        self.prog_shadow = self._load_program('shadow_depth_vs.glsl', 'shadow_depth_fs.glsl')
        self.light_pos = (100.0, -200.0, -200.0) # Fixed Light Position
        
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
        
        # VAO for bloom shaders (reuse quad)
        self.vao_bloom_bright = self.ctx.simple_vertex_array(self.prog_bloom_bright, self.vbo_quad, 'in_pos', 'in_uv')
        self.vao_blur = self.ctx.simple_vertex_array(self.prog_blur, self.vbo_quad, 'in_pos', 'in_uv')
        self.vao_bloom_combine = self.ctx.simple_vertex_array(self.prog_bloom_combine, self.vbo_quad, 'in_pos', 'in_uv')

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
            import warnings
            warnings.warn(f"Sky texture not found at {sky_path}. Rendering may be affected.", UserWarning)
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
            import warnings
            warnings.warn(f"Ground texture not found at {ground_path}. Rendering may be affected.", UserWarning)
            self.tex_ground = None
            
    def update_camera(self, position, rotation):
        """Kamera durumunu güncelle"""
        self.camera.set_transform(position, rotation)
        
    def render_aircraft(self, position, heading=0.0, roll=0.0, pitch=0.0, color=(1.0, 0.0, 0.0), program=None):
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
        
        # Choose program (Default or Shadow)
        prog = program if program else self.prog_aircraft
        
        if 'm_model' in prog:
            prog['m_model'].write(model.tobytes())
            
        if program is None:
            # Standard Rendering
            prog['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
            prog['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
            prog['u_color'].value = tuple(color)
            if 'viewPos' in prog:
                prog['viewPos'].value = tuple(self.camera.position)
            if 'lightSpaceMatrix' in prog and hasattr(self, 'light_space_matrix'):
                prog['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())
        
        self.vao_aircraft.render()

    def render_instanced_aircraft(self, positions, directions, colors=None, program=None):
        """
        Render multiple aircraft using instancing.
        Args:
            program: Optional shader program (e.g. shadow depth shader)
        """
        count = len(positions)
        if count == 0:
            return
        if count > self.max_instances:
            # print(f"Warning: Truncating instances {count} -> {self.max_instances}")
            count = self.max_instances
            positions = positions[:count]
            directions = directions[:count]
            colors = colors[:count] if colors is not None else None
            
        prog = program if program else self.prog_instanced
        
        if program is None:
            # Camera Uniforms for standard pass
            prog['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
            prog['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
            if 'lightPos' in prog:
                prog['lightPos'].value = self.light_pos
            if 'viewPos' in prog:
                prog['viewPos'].value = tuple(self.camera.position)
            if 'lightSpaceMatrix' in prog and hasattr(self, 'light_space_matrix'):
                prog['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())
        
        # Prepare Instance Data
        import pyrr
        instance_data = np.zeros(count * 19, dtype='f4')
        def_color = np.array([1.0, 0.0, 0.0], dtype='f4')
        
        for i in range(count):
            pos = positions[i]
            direction = np.array(directions[i])
            length = np.linalg.norm(direction)
            if length < 0.01:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / length
            
            up = np.array([0.0, 0.0, -1.0]) 
            if abs(np.dot(direction, up)) > 0.99:
                right = np.array([0.0, 1.0, 0.0])
            else:
                right = np.cross(direction, up)
                right = right / np.linalg.norm(right)
            real_up = np.cross(right, direction)
            
            mat = np.eye(4, dtype='f4')
            mat[0, :3] = direction
            mat[1, :3] = right
            mat[2, :3] = real_up 
            mat[3, :3] = pos
            
            offset = i * 19
            instance_data[offset:offset+16] = mat.flatten() 
            col = colors[i] if colors is not None else def_color
            instance_data[offset+16:offset+19] = col
            
        self.vbo_instance.write(instance_data.tobytes())
        
        # Use instanced VAO but with shadow program?
        # Important: vao_aircraft_instanced is linked to prog_instanced.
        # To reuse it with prog_shadow, we rely on attribute location compatibility.
        # prog_instanced attributes: 0=pos, 1=norm, 2=instMat, 6=instCol
        # prog_shadow attributes: 0=pos, 2=instMat (if instanced)
        # We need a new VAO for instanced shadows if the locations/formats differ significantly,
        # OR we ensure compat.
        # My shadow_vertex shader (shadow_depth_vs.glsl) currently only takes:
        # layout (location = 0) in vec3 in_position;
        # It assumes SINGLE model matrix uniform. IT DOES NOT SUPPORT INSTANCING YET.
        # I need an instanced version of shadow shader.
        # For now, let's skip instanced shadows render or create 'shadow_depth_instanced_vs.glsl'.
        # Assuming we just update standard render first.
        
        if program:
             # If using shadow program, we can't use vao_aircraft_instanced easily if it doesn't match.
             # Let's skip instanced shadows for this MVP step or just render them as normal loop.
             pass
        else: 
             self.vao_aircraft_instanced.render(moderngl.TRIANGLES, instances=count)

    def render_aircraft(self, position, heading=0.0, roll=0.0, pitch=0.0, color=(1.0, 0.0, 0.0), program=None):
        """
        Belirtilen konumda bir uçak çiz.
        Args:
           program: Optional override shader (e.g. for shadow pass)
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
        
        prog = program if program else self.prog_aircraft
        
        if 'm_model' in prog:
            prog['m_model'].write(model.tobytes())
            
        if program is None:
            prog['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
            prog['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
            prog['u_color'].value = tuple(color)
            if 'viewPos' in prog:
                prog['viewPos'].value = tuple(self.camera.position)
            if 'lightSpaceMatrix' in prog and hasattr(self, 'light_space_matrix'):
                prog['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())
        
        self.vao_aircraft.render()

    # NOTE: render() method removed - use begin_frame() / render_aircraft() / end_frame() pattern instead
        
    def render_instanced_aircraft(self, positions, directions, colors=None):
        """
        Render multiple aircraft using instancing.
        Args:
            positions: List or array of (N, 3) positions
            directions: List or array of (N, 3) direction vectors (velocity/heading)
            colors: Optional List or array of (N, 3) colors
        """
        count = len(positions)
        if count == 0:
            return
        if count > self.max_instances:
            print(f"Warning: Truncating instances {count} -> {self.max_instances}")
            count = self.max_instances
            positions = positions[:count]
            directions = directions[:count]
            if colors is not None:
                colors = colors[:count]
                
        # Camera Uniforms
        self.prog_instanced['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
        self.prog_instanced['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
        
        # Prepare Instance Data
        import pyrr
        instance_data = np.zeros(count * 19, dtype='f4')
        def_color = np.array([1.0, 0.0, 0.0], dtype='f4')
        
        for i in range(count):
            pos = positions[i]
            direction = np.array(directions[i])
            length = np.linalg.norm(direction)
            if length < 0.01:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / length
            
            # X=Forward, Y=Right, Z=Up (aligned)
            up = np.array([0.0, 0.0, -1.0]) # Sim Up is -Z
            if abs(np.dot(direction, up)) > 0.99:
                right = np.array([0.0, 1.0, 0.0])
            else:
                right = np.cross(direction, up)
                right = right / np.linalg.norm(right)
            real_up = np.cross(right, direction)
            
            mat = np.eye(4, dtype='f4')
            mat[0, :3] = direction
            mat[1, :3] = right
            mat[2, :3] = real_up 
            mat[3, :3] = pos
            
            offset = i * 19
            instance_data[offset:offset+16] = mat.flatten() 
            col = colors[i] if colors is not None else def_color
            instance_data[offset+16:offset+19] = col
            
        self.vbo_instance.write(instance_data.tobytes())
        self.vao_aircraft_instanced.render(moderngl.TRIANGLES, instances=count)

    # --- ENVIRONMENT RENDERING ---
    
    def init_environment(self, environment):
        """
        Environment için GPU kaynaklarını hazırla.
        
        Args:
            environment: Environment objesi (terrain + objects)
        """
        from src.rendering.geometry import GeometryGenerator
        
        # Terrain mesh oluştur
        terrain = environment.terrain
        terrain_mesh = GeometryGenerator.create_terrain_mesh(
            terrain.heightmap,
            tuple(terrain.size),
            resolution=64  # Performance için düşük
        )
        
        self.vbo_terrain = self.ctx.buffer(terrain_mesh.tobytes())
        self.vao_terrain = self.ctx.simple_vertex_array(
            self.prog_aircraft,  # box shader'ı tekrar kullan
            self.vbo_terrain, 
            'in_position', 'in_normal'
        )
        self.terrain_vertex_count = len(terrain_mesh) // 6  # 6 float per vertex
        
        # World objects için mesh'leri hazırla
        self._init_object_meshes()
        
        # Environment referansını sakla
        self.environment = environment
        
    def _init_object_meshes(self):
        """Object mesh'lerini oluştur (building, tree)"""
        from src.rendering.geometry import GeometryGenerator
        
        # Building mesh
        building_mesh = GeometryGenerator.create_building_mesh(1.0, 1.0, 1.0)  # Unit size, scale at render
        self.vbo_building = self.ctx.buffer(building_mesh.tobytes())
        self.vao_building = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_building, 
            'in_position', 'in_normal'
        )
        
        # Tree mesh
        tree_mesh = GeometryGenerator.create_tree_mesh(1.0, 0.5)  # Unit size
        self.vbo_tree = self.ctx.buffer(tree_mesh.tobytes())
        self.vao_tree = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_tree,
            'in_position', 'in_normal'
        )
    
    def render_terrain(self, terrain_color=(0.15, 0.25, 0.1)):
        """
        Terrain mesh'i render et.
        
        Args:
            terrain_color: Terrain rengi (r, g, b)
        """
        if not hasattr(self, 'vao_terrain'):
            return
        
        import pyrr
        
        # NED → OpenGL koordinat dönüşümü 
        # GLCamera.SIM_TO_GL ile aynı - terrain mesh NED'de oluşturuldu
        # Sim: X=Forward, Y=Right, Z=Down  → GL: X=Right, Y=Up, Z=-Forward
        # Terrain mesh: X=position(0-2000), Y=height(0), Z=position(0-2000)
        # Bu aslında NED değil, XYZ world coords. Terrain'i dönüştürmeliyiz.
        sim_to_gl = np.array([
            [0, 1, 0, 0],   # GL.X = Sim.Y (mesh X -> GL X) 
            [0, 0, -1, 0],  # GL.Y = -Sim.Z 
            [-1, 0, 0, 0],  # GL.Z = -Sim.X
            [0, 0, 0, 1]
        ], dtype='f4')
        
        # Terrain mesh uses: X=world_x, Y=height, Z=world_z (already OpenGL-like)
        # But the world coords need to match NED world where camera is
        # Actually terrain is drawn in "raw" coords while camera views in transformed coords
        # Apply inverse transform: GL -> Sim so terrain appears in camera's view
        # Or just apply same transform so both are in same space
        model = sim_to_gl
        
        self.prog_aircraft['m_model'].write(model.tobytes())
        self.prog_aircraft['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
        self.prog_aircraft['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
        self.prog_aircraft['u_color'].value = terrain_color
        
        if 'viewPos' in self.prog_aircraft:
            self.prog_aircraft['viewPos'].value = tuple(self.camera.position)
        if 'lightSpaceMatrix' in self.prog_aircraft and hasattr(self, 'light_space_matrix'):
            self.prog_aircraft['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())
        
        self.vao_terrain.render()
    
    def render_world_objects(self, objects):
        """
        World object'lerini render et.
        
        Args:
            objects: WorldObject listesi
        """
        if not hasattr(self, 'vao_building'):
            return
            
        import pyrr
        
        # NED → GL transform (same as terrain)
        sim_to_gl = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype='f4')
        
        for obj in objects:
            # Object tipine göre VAO seç
            if obj.obj_type == 'building':
                vao = self.vao_building
            elif obj.obj_type == 'tree':
                vao = self.vao_tree
            else:
                continue  # Bilinmeyen tip
            
            # Model matrix oluştur (scale + rotate + translate + coord transform)
            scale_mat = pyrr.matrix44.create_from_scale(
                [obj.size[0], obj.size[1], obj.size[2]], dtype='f4'
            )
            rot_mat = pyrr.matrix44.create_from_y_rotation(obj.rotation, dtype='f4')
            trans_mat = pyrr.matrix44.create_from_translation(obj.position, dtype='f4')
            
            model = pyrr.matrix44.multiply(scale_mat, rot_mat)
            model = pyrr.matrix44.multiply(model, trans_mat)
            model = pyrr.matrix44.multiply(model, sim_to_gl)  # Apply coord transform
            
            # Shader uniforms
            self.prog_aircraft['m_model'].write(model.tobytes())
            self.prog_aircraft['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
            self.prog_aircraft['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
            self.prog_aircraft['u_color'].value = obj.color
            
            if 'viewPos' in self.prog_aircraft:
                self.prog_aircraft['viewPos'].value = tuple(self.camera.position)
            if 'lightSpaceMatrix' in self.prog_aircraft and hasattr(self, 'light_space_matrix'):
                self.prog_aircraft['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())
            
            vao.render()
    
    def render_environment(self):
        """
        Tüm environment'ı render et (terrain + objects).
        init_environment() çağrıldıktan sonra kullanılır.
        """
        if not hasattr(self, 'environment'):
            return
        
        # Terrain
        self.render_terrain()
        
        # World objects
        self.render_world_objects(self.environment.get_all_objects())

    def begin_shadow_pass(self):
        """Start Shadow Map Rendering Pass"""
        self.fbo_shadow.use()
        self.ctx.clear()
        
        # Light setup
        import pyrr
        light_projection = pyrr.matrix44.create_orthogonal_projection(
            -500, 500, -500, 500, 1.0, 2000.0, dtype='f4'
        )
        light_view = pyrr.matrix44.create_look_at(
            np.array(self.light_pos, dtype='f4'),
            np.array([0.0, 0.0, 0.0], dtype='f4'),
            np.array([0.0, 0.0, -1.0], dtype='f4'),
            dtype='f4'
        )
        self.light_space_matrix = pyrr.matrix44.multiply(light_view, light_projection)
        self.prog_shadow['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())

    def begin_frame(self):
        """Render döngüsünü başlat (Main FBO Bind)"""
        
        # 1. Main Pass
        self.fbo.use()
        self.ctx.clear(0.4, 0.6, 0.8)  # Açık mavi gökyüzü arka planı
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Bind Shadow Map
        self.tex_shadow.use(location=5)
        self.prog_aircraft['shadowMap'] = 5
        
        # Environment rendering (terrain + objects) - Skybox/Ground texture yerine
        if hasattr(self, 'environment') and self.environment is not None:
            # Real environment rendering
            self.render_environment()
        else:
            # Fallback: Eski skybox/ground texture sistemi
            if self.tex_sky:
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.tex_sky.use(location=2)
                self.prog_sky['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
                self.prog_sky['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
                self.vao_sky.render(moderngl.TRIANGLES)
                self.ctx.enable(moderngl.DEPTH_TEST)
                
            if self.tex_ground:
                self.tex_ground.use(location=3)
                self.prog_ground['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
                self.prog_ground['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
                self.vao_ground.render(moderngl.TRIANGLES)
        
    def end_frame(self, time=0.0):
        """Render döngüsünü bitir (Post-Process & Bloom & Draw to Output FBO)"""
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # --- BLOOM PASS (if enabled) ---
        if self.bloom_enabled:
            # Pass 1: Extract bright pixels
            self.fbo_bloom_bright.use()
            self.ctx.clear(0.0, 0.0, 0.0)
            self.texture.use(location=0)
            self.prog_bloom_bright['screenTexture'].value = 0
            if 'threshold' in self.prog_bloom_bright:
                self.prog_bloom_bright['threshold'].value = self.bloom_threshold
            self.vao_bloom_bright.render(moderngl.TRIANGLE_STRIP)
            
            # Pass 2: Horizontal blur
            self.fbo_bloom_blur1.use()
            self.ctx.clear(0.0, 0.0, 0.0)
            self.texture_bloom_bright.use(location=0)
            self.prog_blur['inputTexture'].value = 0
            self.prog_blur['direction'].value = (1.0, 0.0)
            if 'blurSize' in self.prog_blur:
                self.prog_blur['blurSize'].value = 2.0
            self.vao_blur.render(moderngl.TRIANGLE_STRIP)
            
            # Pass 3: Vertical blur
            self.fbo_bloom_blur2.use()
            self.ctx.clear(0.0, 0.0, 0.0)
            self.texture_bloom_blur1.use(location=0)
            self.prog_blur['inputTexture'].value = 0
            self.prog_blur['direction'].value = (0.0, 1.0)
            self.vao_blur.render(moderngl.TRIANGLE_STRIP)
            
            # Pass 4: Combine bloom with scene
            self.fbo_post.use()
            self.ctx.clear(0.0, 0.0, 0.0)
            self.texture.use(location=0)
            self.texture_bloom_blur2.use(location=1)
            self.prog_bloom_combine['sceneTexture'].value = 0
            self.prog_bloom_combine['bloomTexture'].value = 1
            if 'bloomIntensity' in self.prog_bloom_combine:
                self.prog_bloom_combine['bloomIntensity'].value = self.bloom_intensity
            self.vao_bloom_combine.render(moderngl.TRIANGLE_STRIP)
        else:
            # --- PASS 2: Post Processing (no bloom) ---
            self.fbo_post.use()
            self.ctx.clear(0.0, 0.0, 0.0)
            
            self.texture.use(location=0)
            self.depth_buffer.use(location=1)
            
            self.prog_post['screenTexture'].value = 0
            self.prog_post['depthTexture'].value = 1
            
            if 'time' in self.prog_post:
                self.prog_post['time'].value = time
            
            self.prog_post['focusDistance'].value = 500.0 
            self.prog_post['focusRange'].value = 1000.0
            self.prog_post['dofEnabled'].value = False
            
            if 'vignetteStrength' in self.prog_post:
                self.prog_post['vignetteStrength'].value = 0.3
            
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
