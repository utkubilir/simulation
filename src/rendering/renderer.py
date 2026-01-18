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
        # 1. Skybox Shader
        self.prog_sky = self._load_program('sky_vs.glsl', 'sky_fs.glsl')
        
        # 2. Ground Shader (Textured) (Replaces Grid)
        self.prog_ground = self._load_program('ground_vs.glsl', 'ground_fs.glsl')
        
        # 3. Terrain Shader (Realistic)
        self.prog_terrain = self._load_program('terrain_vs.glsl', 'terrain_fs.glsl')
        
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
        self.bloom_intensity = 0.3
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
        self.light_pos = (100.0, 1000.0, 100.0) # Fixed Light Position (Top-Down for correct diffuse)
        
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
        """
        Create dummy textures for shader fallback system.
        
        Real rendering uses Environment (terrain mesh + world objects).
        Sky and ground shaders have procedural fallbacks when textures return black.
        """
        # Create 1x1 black dummy textures - this triggers shader fallback colors
        # (shader checks if texture luminance < 0.01 and uses procedural colors)
        black_pixel = bytes([0, 0, 0])
        
        # Dummy sky texture
        self.tex_sky = self.ctx.texture((1, 1), 3, black_pixel)
        self.tex_sky.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.prog_sky['skyTexture'] = 2
        
        # Dummy ground texture
        self.tex_ground = self.ctx.texture((1, 1), 3, black_pixel)
        self.tex_ground.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.prog_ground['groundTexture'] = 3
        
        # Generate procedural terrain texture
        self._generate_terrain_texture()
            
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


    # NOTE: render() method removed - use begin_frame() / render_aircraft() / end_frame() pattern instead

    # --- ENVIRONMENT RENDERING ---
    
    def init_environment(self, environment):
        """
        Environment için GPU kaynaklarını hazırla.
        
        Args:
            environment: Environment objesi (terrain + objects)
        """
        # World objects için mesh'leri hazırla
        self._init_object_meshes()
        
        # Environment referansını sakla
        self.environment = environment
        
        if hasattr(environment.chunk_manager, 'add_callbacks'):
            environment.chunk_manager.add_callbacks(
                on_load=self._on_chunk_loaded,
                on_unload=self._on_chunk_unloaded
            )
        else:
            environment.chunk_manager.on_chunk_loaded = self._on_chunk_loaded
            environment.chunk_manager.on_chunk_unloaded = self._on_chunk_unloaded
        
        self._chunk_vaos = {}  # {coords: vao}
        self._chunk_shadow_vaos = {} # {coords: vao_shadow}
        self._chunk_vbos = {}  # {coords: vbo}


    
    def _on_chunk_loaded(self, chunk):
        """Yeni chunk yüklendiğinde VAO oluştur"""
        from src.rendering.geometry import GeometryGenerator
        
        # Chunk mesh oluştur
        mesh = GeometryGenerator.create_terrain_mesh(
            chunk.heightmap,
            (chunk.chunk_size, chunk.chunk_size),
            resolution=chunk.resolution
        )
        
        vbo = self.ctx.buffer(mesh.tobytes())
        vao = self.ctx.simple_vertex_array(
            self.prog_terrain, vbo,
            'in_position', 'in_normal', 'in_texcoord'
        )
        
        self._chunk_vbos[chunk.coords] = vbo
        self._chunk_vaos[chunk.coords] = vao
        
        # Shadow VAO (Manual definition to handle stride)
        # Terrain VBO format: 3f (pos), 3f (norm), 2f (tex)
        # Shadow shader only needs pos. Use stride to skip norm/tex.
        vao_shadow = self.ctx.vertex_array(self.prog_shadow, [
            (vbo, '3f 20x', 'in_position')
        ])
        
        self._chunk_shadow_vaos[chunk.coords] = vao_shadow
        chunk.vertex_count = len(mesh) // 6
    
    def _on_chunk_unloaded(self, chunk):
        """Chunk kaldırıldığında GPU kaynaklarını temizle"""
        coords = chunk.coords
        
        if coords in self._chunk_vaos:
            self._chunk_vaos[coords].release()
            del self._chunk_vaos[coords]
            
        if coords in self._chunk_shadow_vaos:
            self._chunk_shadow_vaos[coords].release()
            del self._chunk_shadow_vaos[coords]
            
        if coords in self._chunk_vbos:
            self._chunk_vbos[coords].release()
            del self._chunk_vbos[coords]
    
    def render_terrain_chunks(self, program=None):
        """Sonsuz dünya: Görünür chunk'ları render et"""
        if not hasattr(self, 'environment') or not hasattr(self.environment, 'chunk_manager'):
            return
        
        import pyrr
        chunks = self.environment.chunk_manager.get_visible_chunks()
        
        # Use provided program (shadow pass) or default terrain shader
        prog = program if program else self.prog_terrain
        
        for chunk in chunks:
            # Check local VAO storage
            vao = self._chunk_vaos.get(chunk.coords)
            if not vao:
                continue
            
            # Her chunk için model matrisi
            ox, oy = chunk.world_origin
            model = np.eye(4, dtype='f4')
            
            # Sim (x, y, z) -> GL (x, z, -y) transformation logic
            # Chunk is on XY plane in Sim. In GL it should be on XZ plane.
            # But the chunk mesh itself is built on XY plane?
            # Wait, chunk VAO contains vertices. MapData generates heights.
            # Usually chunk vertices are (x, y, z).
            # If we want to rotate them:
            # We can use a rotation matrix OR set the position columns manually.
            
            # Legacy logic used model[3,0] type setting.
            # Let's use proper transformation.
            # Sim Origin (ox, oy)
            # GL Position (ox, 0, -oy)
            
            # Coordinate Transform Matrix (Z-Up -> Y-Up)
            # [1, 0, 0, 0]
            # [0, 0,-1, 0]
            # [0, 1, 0, 0]
            # [0, 0, 0, 1]
            
            # Translate to (ox, oy, 0) then Rotate?
            # Or just Translate in GL coords?
            
            # Let's construct the matrix:
            # Translation
            trans = pyrr.matrix44.create_from_translation([ox, 0, -oy], dtype='f4')
            
            # Rotation of the MESH itself? 
            # If mesh is (x, y, z) where z is height.
            # We want z -> y (up), y -> -z (north).
            # So X->X, Y->-Z, Z->Y.
            
            # Create rotation matrix
            # Col 0: (1, 0, 0)
            # Col 1: (0, 0, -1) -> Sim Y maps to GL -Z
            # Col 2: (0, 1, 0) -> Sim Z maps to GL Y
            
            rot = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype='f4')
            
            # Combine: Translate * Rotate * Vertex? Or Rotate * Translate?
            # Vertex is local (0..chunk_size). 
            # We want to move it to (ox, oy) then rotate?
            # No, if we rotate, (x, y, z) becomes (x, z, -y).
            # Then we add offset (ox, 0, -oy).
            
            model = pyrr.matrix44.multiply(rot, trans)
            
            prog['m_model'].write(model.tobytes())
            
            # View/Proj only if not shadow pass (shadow pass handles these globally? No.)
            # If program is passed, caller might have set uniforms?
            # BUT render_instanced_aircraft says "program: Optional shader".
            # Shadow pass sets 'lightSpaceMatrix' globally.
            # But m_model is per object.
            # m_view/m_proj are camera specific. In shadow pass, we don't need camera view/proj?
            # Wait. Shadow mapping needs Light View/Proj. 
            # That is 'lightSpaceMatrix = light_proj * light_view'.
            # Vertex Shader usually does: gl_Position = lightSpaceMatrix * model * vec4(pos, 1.0);
            
            # So for shadow pass, we DO NOT write camera m_view/m_proj.
            # We assume 'program' (shadow shader) has 'm_model' and 'lightSpaceMatrix'.
            # 'lightSpaceMatrix' is set in begin_shadow_pass globally? 
            # No, uniform must be set for the program.
            # Check render_instanced_aircraft implementation.
            
            if not program:
                prog['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
                prog['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
                prog['u_color'].value = (1.0, 1.0, 1.0)
                if 'viewPos' in prog:
                    prog['viewPos'].value = tuple(self.camera.position)
            
            # Select proper VAO
            if program == self.prog_shadow:
                vao_shadow = self._chunk_shadow_vaos.get(chunk.coords)
                if vao_shadow:
                    vao_shadow.render()
            else:
                vao.render(prog if program else None)
        
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
        
        
        # Building Shadow VAO (3f pos, 3f norm -> skip norm)
        self.vao_building_shadow = self.ctx.vertex_array(self.prog_shadow, [
            (self.vbo_building, '3f 12x', 'in_position')
        ])
        
        # Tree mesh
        tree_mesh = GeometryGenerator.create_tree_mesh(1.0, 0.5)  # Unit size
        self.vbo_tree = self.ctx.buffer(tree_mesh.tobytes())
        self.vao_tree = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_tree,
            'in_position', 'in_normal'
        )
        
        # Tree Shadow VAO
        self.vao_tree_shadow = self.ctx.vertex_array(self.prog_shadow, [
            (self.vbo_tree, '3f 12x', 'in_position')
        ])
        
        # Arena pole mesh (for boundary markers)
        pole_mesh = GeometryGenerator.create_pole_mesh(height=1.0, radius=0.5)  # Unit height
        self.vbo_pole = self.ctx.buffer(pole_mesh.tobytes())
        self.vao_pole = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_pole,
            'in_position', 'in_normal'
        )
        self.pole_vertex_count = len(pole_mesh) // 6
        
        # Arena cone mesh (for edge markers)
        cone_mesh = GeometryGenerator.create_cone_mesh(height=1.0, radius=0.5)
        self.vbo_cone = self.ctx.buffer(cone_mesh.tobytes())
        self.vao_cone = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_cone,
            'in_position', 'in_normal'
        )
        self.cone_vertex_count = len(cone_mesh) // 6

        # Arena helipad mesh (ring)
        helipad_mesh = GeometryGenerator.create_ring_mesh(outer_radius=1.0, inner_radius=0.8)
        self.vbo_helipad = self.ctx.buffer(helipad_mesh.tobytes())
        self.vao_helipad = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_helipad,
            'in_position', 'in_normal'
        )
        self.helipad_vertex_count = len(helipad_mesh) // 6
        
        # Arena ground marker mesh (for safe zones)
        marker_mesh = GeometryGenerator.create_ground_marker(1.0, 1.0)  # Unit size
        self.vbo_ground_marker = self.ctx.buffer(marker_mesh.tobytes())
        self.vao_ground_marker = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_ground_marker,
            'in_position', 'in_normal'
        )
        self.ground_marker_vertex_count = len(marker_mesh) // 6
        
        # Tent mesh
        tent_mesh = GeometryGenerator.create_tent_mesh(1.0, 1.0, 1.0)
        self.vbo_tent = self.ctx.buffer(tent_mesh.tobytes())
        self.vao_tent = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_tent,
            'in_position', 'in_normal'
        )
        
        # Box mesh
        box_mesh = GeometryGenerator.create_box_mesh(1.0, 1.0, 1.0)
        self.vbo_box = self.ctx.buffer(box_mesh.tobytes())
        self.vao_box = self.ctx.simple_vertex_array(
            self.prog_aircraft, self.vbo_box,
            'in_position', 'in_normal'
        )
    
    def render_terrain(self, terrain_color=(0.34, 0.55, 0.27)):  # Green terrain
        """
        Terrain mesh'i render et.
        
        Args:
            terrain_color: Terrain rengi (r, g, b)
        """
        if not hasattr(self, 'vao_terrain'):
            return
        
        import pyrr
        
        # Terrain mesh: X=0..2000, Y=height (0..30), Z=0..2000
        # Simple model matrix: just center the terrain
        model = np.eye(4, dtype='f4')
        # Center terrain around origin: mesh goes 0..2000, shift by -1000
        model[3, 0] = -1000.0  # Translate X
        model[3, 2] = -1000.0  # Translate Z
        
        self.prog_terrain['m_model'].write(model.tobytes())
        self.prog_terrain['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
        self.prog_terrain['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
        self.prog_terrain['u_color'].value = (1.0, 1.0, 1.0) # White base for texture
        
        if 'viewPos' in self.prog_terrain:
            self.prog_terrain['viewPos'].value = tuple(self.camera.position)
        if 'lightSpaceMatrix' in self.prog_terrain and hasattr(self, 'light_space_matrix'):
            self.prog_terrain['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())
        
        if hasattr(self, 'tex_terrain'):
            self.prog_terrain['u_texture'] = 0
            self.tex_terrain.use(location=0)
            
        self.vao_terrain.render()
        
    def render_sky(self):
        """Render Skybox (Procedural or Textured)"""
        # Disable Depth Write (Sky is background)
        self.ctx.enable(moderngl.DEPTH_TEST) # Ensure depth test needed? 
        # Usually skybox is drawn last at max depth OR first with depth write off.
        # Let's draw FIRST (after clear) with Depth Write OFF.
        # But we are calling this inside render_environment which might be after objects?
        # Ideally: Clear -> Render Sky -> Render Scene
        
        self.ctx.depth_mask = False
        
        # Use Sky Shader
        self.prog_sky['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
        self.prog_sky['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
        
        if 'lightPos' in self.prog_sky:
            self.prog_sky['lightPos'].value = self.light_pos
            
        if hasattr(self, 'tex_sky'):
            self.prog_sky['skyTexture'] = 2
            self.tex_sky.use(location=2)
            
        # self.vao_sky uses vbo_quad_2d (-1..1), shader sets Z=0.9999
        self.vao_sky.render(moderngl.TRIANGLES)
        
        self.ctx.depth_mask = True

    def init_arena(self, arena):
        """
        Arena için GPU kaynaklarını hazırla.
        
        Args:
            arena: TeknofestArena objesi
        """
        self.arena = arena
        self._init_object_meshes()  # Ensure meshes are ready
    
    def render_arena(self):
        """
        Arena sınır işaretçilerini ve güvenli bölgeleri render et.
        """
        if not hasattr(self, 'arena') or self.arena is None:
            return
        
        if not hasattr(self, 'vao_pole'):
            return
        
        import pyrr
        
        view = self.camera.get_view_matrix().astype('f4')
        proj = self.camera.get_projection_matrix().astype('f4')
        
        self.prog_aircraft['m_view'].write(view.tobytes())
        self.prog_aircraft['m_proj'].write(proj.tobytes())
        
        if 'viewPos' in self.prog_aircraft:
            self.prog_aircraft['viewPos'].value = tuple(self.camera.position)
        
        # Render boundary markers
        for marker in self.arena.markers:
            pos = marker['position'] # [East, North, Alt]
            height = marker.get('height', 1.0)
            color = marker['color']
            marker_type = marker.get('type', 'corner_pole')
            radius = marker.get('radius', 0.5)
            
            model = np.eye(4, dtype='f4')
            
            # Coordinate Transform: Sim [x, y, z] -> GL [x, z, -y]
            gl_x = pos[0]
            gl_y = pos[2] # Altitude -> Up
            gl_z = -pos[1] # North -> -Back
            
            if marker_type == 'corner_pole':
                # Cylinder pole
                model[0, 0] = 1.0 
                model[1, 1] = height 
                model[2, 2] = 1.0 
                model[3, 0] = gl_x
                model[3, 1] = gl_y
                model[3, 2] = gl_z
                
                self.prog_aircraft['m_model'].write(model.tobytes())
                self.prog_aircraft['u_color'].value = color
                self.vao_pole.render()
                
            elif marker_type == 'cone':
                # Cone marker
                model[0, 0] = 1.0 
                model[1, 1] = height
                model[2, 2] = 1.0
                model[3, 0] = gl_x
                model[3, 1] = gl_y
                model[3, 2] = gl_z
                
                self.prog_aircraft['m_model'].write(model.tobytes())
                self.prog_aircraft['u_color'].value = color
                if hasattr(self, 'vao_cone'):
                    self.vao_cone.render()
                    
            elif marker_type == 'helipad':
                scale = radius
                model[0, 0] = scale
                model[1, 1] = 1.0 
                model[2, 2] = scale
                model[3, 0] = gl_x
                model[3, 1] = gl_y
                model[3, 2] = gl_z
                
                self.prog_aircraft['m_model'].write(model.tobytes())
                self.prog_aircraft['u_color'].value = color
                if hasattr(self, 'vao_helipad'):
                    self.vao_helipad.render()
        
        # Render safe zone ground markers
        for zone in self.arena.safe_zones:
            center = zone.center # [x, y, z]
            size = zone.size # [width, depth, height]
            
            # GL Transform
            gl_x = center[0]
            gl_y = 0.5 # Ground level + offset
            gl_z = -center[1] 
            
            # Scale mapped to Zone Size
            # size[0] = Width (X)
            # size[1] = Depth (Y)
            # size[2] = Height (Z) -> ignored for ground marker
            
            model = np.eye(4, dtype='f4')
            model[0, 0] = size[0]  # Width
            model[1, 1] = 1.0  # Thin
            model[2, 2] = size[1]  # Depth (Sim Y mapped to GL Z)
            model[3, 0] = gl_x
            model[3, 1] = gl_y
            model[3, 2] = gl_z
            
            self.prog_aircraft['m_model'].write(model.tobytes())
            self.prog_aircraft['u_color'].value = zone.color
            self.vao_ground_marker.render()
            
        # Render decorative detail objects (tents, boxes)
        if hasattr(self.arena, 'detail_objects'):
            for obj in self.arena.detail_objects:
                pos = obj['position']
                color = obj['color']
                scale = obj.get('scale', (1.0, 1.0, 1.0)) # [sx, sy, sz]
                rotation = obj.get('rotation', 0.0)
                obj_type = obj['type']
                
                # Transform Scale: Sim [x, y, z] -> GL [x, z, y]
                # Because Sim Y is Depth (GL Z), Sim Z is Height (GL Y)
                gl_scale = [scale[0], scale[2], scale[1]]
                
                # Transform Pos
                gl_pos = [pos[0], pos[2], -pos[1]]
                
                import pyrr
                s_mat = pyrr.matrix44.create_from_scale(gl_scale, dtype='f4')
                r_mat = pyrr.matrix44.create_from_y_rotation(rotation, dtype='f4')
                t_mat = pyrr.matrix44.create_from_translation(gl_pos, dtype='f4')
                
                model = pyrr.matrix44.multiply(s_mat, r_mat)
                model = pyrr.matrix44.multiply(model, t_mat)
                
                self.prog_aircraft['m_model'].write(model.tobytes())
                self.prog_aircraft['u_color'].value = color
                
                if obj_type == 'tent' and hasattr(self, 'vao_tent'):
                    self.vao_tent.render()
                elif obj_type == 'box' and hasattr(self, 'vao_box'):
                    self.vao_box.render()

    def render_environment_objects_only(self):
        """
        Called if we want strict separation, but render_environment calls all.
        """
        if hasattr(self, 'arena'):
             self.render_arena()

    def _generate_terrain_texture(self):
        """Generate procedural grass texture."""
        size = 512
        # Base green color (Forest Green)
        base_color = np.array([34, 139, 34], dtype=np.float32)
        
        # Create noise
        texture_data = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Simple noise: random variations
        noise = np.random.normal(0, 15, (size, size, 3))
        
        # Apply base color
        final_color = base_color + noise
        texture_data = np.clip(final_color, 0, 255).astype(np.uint8)
        
        self.tex_terrain = self.ctx.texture((size, size), 3, texture_data.tobytes())
        self.tex_terrain.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.tex_terrain.build_mipmaps()
    def _is_in_frustum(self, position, radius=50.0):
        """Basit frustum culling: Obje kamera görüş alanında mı?"""
        cam_pos = np.array(self.camera.position)
        obj_pos = np.array(position)
        distance = np.linalg.norm(obj_pos - cam_pos)
        
        # Çok uzaktaki objeleri atla (2km görüş mesafesi)
        if distance > 2000.0 + radius:
            return False
        return True
    
    def render_world_objects(self, objects, program=None):
        """
        World object'lerini render et.
        
        Args:
            objects: WorldObject listesi
            program: Optional shader program override (for shadow pass)
        """
        if not hasattr(self, 'vao_building'):
            return
            
        import pyrr
        
        # Sim (Z-Up) to GL (Y-Up) Transformation
        # X -> X
        # Y -> -Z (North is forward/into screen)
        # Z -> Y (Altitude is Up)
        sim_to_gl = np.array([
            [1, 0, 0, 0],   # Col 0 (X) maps to X
            [0, 0, -1, 0],  # Col 1 (Y) maps to -Z
            [0, 1, 0, 0],   # Col 2 (Z) maps to Y
            [0, 0, 0, 1]
        ], dtype='f4')
        
        for obj in objects:
            # Frustum culling
            if not self._is_in_frustum(obj.position, max(obj.size)):
                continue
            
            # Object tipine göre VAO seç
            if obj.obj_type == 'building':
                vao = self.vao_building
                # Default colors if not shadow pass
                prog = program if program else self.prog_aircraft # Uses simple shader
                
            elif obj.obj_type == 'tree':
                vao = self.vao_tree
                prog = program if program else self.prog_aircraft
            else:
                continue
            
            # Model matrix oluştur
            # scale -> rotate -> translate -> coord transform
            scale_mat = pyrr.matrix44.create_from_scale(
                [obj.size[0], obj.size[1], obj.size[2]], dtype='f4'
            )
            # Sim Rotation is usually yaw around Z axis.
            # In GL, Z maps to Y. So rotation around Sim-Z is rotation around GL-Y.
            rot_mat = pyrr.matrix44.create_from_y_rotation(np.radians(obj.rotation), dtype='f4')
            
            trans_mat = pyrr.matrix44.create_from_translation(obj.position, dtype='f4')
            
            model = pyrr.matrix44.multiply(scale_mat, rot_mat)
            model = pyrr.matrix44.multiply(model, trans_mat)
            model = pyrr.matrix44.multiply(model, sim_to_gl)
            
            # Shader uniforms
            prog['m_model'].write(model.tobytes())
            
            if not program:
                prog['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
                prog['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
                prog['u_color'].value = obj.color
                
                if 'viewPos' in prog:
                    prog['viewPos'].value = tuple(self.camera.position)
                if 'lightSpaceMatrix' in prog and hasattr(self, 'light_space_matrix'):
                    prog['lightSpaceMatrix'].write(self.light_space_matrix.tobytes())
            
            # Use Shadow VAO if doing shadow pass
            if program == self.prog_shadow:
                if obj.obj_type == 'building' and hasattr(self, 'vao_building_shadow'):
                    self.vao_building_shadow.render()
                elif obj.obj_type == 'tree' and hasattr(self, 'vao_tree_shadow'):
                    self.vao_tree_shadow.render()
                else:
                    # Fallback (might fail if shader mismatch, but better than nothing)
                    vao.render() 
            else:
                vao.render(prog if program else None)
    
    def render_environment(self, shadow_pass=False):
        """
        Tüm environment'ı render et (sky + terrain + arena + objects).
        
        Args:
            shadow_pass: If True, renders for shadow map (depth only).
        """
        # Select shader program for shadow pass
        program = self.prog_shadow if shadow_pass else None
        
        if not shadow_pass:
            # 1. Skybox (No shadows for sky)
            self.render_sky()
        
        # 2. Terrain (Chunk-based veya static)
        if hasattr(self, '_chunk_vaos') and self._chunk_vaos:
            # Sonsuz dünya: Chunk'ları render et
            self.render_terrain_chunks(program=program)
        else:
            # Static terrain (origin area)
            # render_terrain needs update too if we want shadows on legacy terrain
            # But legacy terrain is deprecated.
            if not shadow_pass:
                self.render_terrain()
        
        if not shadow_pass:
            # 3. Arena (poles, markers) - Markers usually don't cast shadows
            if hasattr(self, 'arena'):
                self.render_arena()
            
        # 4. World objects (trees, buildings)
        if hasattr(self, 'environment'):
            self.render_world_objects(self.environment.get_all_objects(), program=program)

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
        
        # Render scene for shadows
        # Instanced aircraft are rendered by caller (render_instanced_aircraft(program=prog_shadow))
        # But we must render static environment here if we want terrain/buildings to cast shadows
        self.render_environment(shadow_pass=True)

    def begin_frame(self):
        """Render döngüsünü başlat (Main FBO Bind)"""
        
        # 1. Main Pass
        self.fbo.use()
        self.ctx.clear(0.53, 0.81, 0.92)  # Sky Blue Background
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Bind Shadow Map
        self.tex_shadow.use(location=5)
        self.prog_aircraft['shadowMap'] = 5
        
        # Environment rendering (terrain + objects) - Skybox/Ground texture yerine
        # Always render sky first (with procedural fallback if texture unavailable)
        if self.tex_sky:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.tex_sky.use(location=2)
            self.prog_sky['m_view'].write(self.camera.get_view_matrix().astype('f4').tobytes())
            self.prog_sky['m_proj'].write(self.camera.get_projection_matrix().astype('f4').tobytes())
            self.vao_sky.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Render environment terrain OR fallback ground texture
        if hasattr(self, 'environment') and self.environment is not None:
            # Real environment rendering (terrain mesh + objects)
            self.render_environment()
        elif self.tex_ground:
            # Fallback: ground texture
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
        # Use persistent buffer to avoid allocation
        expected_size = self.width * self.height * 3
        if not hasattr(self, '_read_buffer') or self._read_buffer.size != expected_size:
            self._read_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)

        # Read from post-processed FBO (includes bloom and other effects)
        # If bloom is disabled, fbo_post still contains the final output
        self.fbo_post.read_into(self._read_buffer, components=3, alignment=1)
        
        # Flip Y (OpenGL orijini sol alt, Image sol üst)
        # RGB -> BGR (OpenCV için)
        # Copy to make it contiguous
        image = self._read_buffer[::-1, :, ::-1].copy()
        
        return image
