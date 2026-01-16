#version 330 core

layout (location = 0) in vec2 in_pos; // Fullscreen Quad (-1..1)

out vec3 viewDir;

uniform mat4 m_view;
uniform mat4 m_proj;

void main() {
    // Standard Fullscreen Quad z=1.0 (Far Plane)
    gl_Position = vec4(in_pos, 0.9999, 1.0);
    
    // Inverse View-Projection to find Ray Direction
    mat4 invProj = inverse(m_proj);
    mat4 invView = inverse(m_view);
    
    // Clip coordinates for this vertex (at far plane)
    vec4 clipPos = vec4(in_pos, 1.0, 1.0);
    
    // View Space
    vec4 viewPos = invProj * clipPos;
    viewPos /= viewPos.w; // Perspective divide
    viewPos = vec4(viewPos.xyz, 0.0); // Direction vector (w=0)
    
    // World Space Direction
    vec3 worldDir = (invView * viewPos).xyz;
    viewDir = normalize(worldDir);
}
