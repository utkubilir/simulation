#version 330

// Post-Process Screen Quad Vertex Shader
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;

out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
