#version 330 core

layout (location = 0) in vec3 in_position;

uniform mat4 m_view;
uniform mat4 m_proj;

out vec3 nearPoint;
out vec3 farPoint;

// Unproject clip space points to world space
vec3 UnprojectPoint(float x, float y, float z, mat4 view, mat4 proj) {
    mat4 viewInv = inverse(view);
    mat4 projInv = inverse(proj);
    vec4 unprojectedPoint =  viewInv * projInv * vec4(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main() {
    // Quad is drawn at Z=0 in NDC? Or passed as -1..1 quad
    gl_Position = vec4(in_position, 1.0);
    
    // Determine near and far points of the view frustum ray for this pixel
    nearPoint = UnprojectPoint(in_position.x, in_position.y, -1.0, m_view, m_proj);
    farPoint  = UnprojectPoint(in_position.x, in_position.y,  1.0, m_view, m_proj);
}
