#version 330

// Grid Vertex Shader
// Sonsuz zemin ızgarası için

layout (location = 0) in vec3 in_position;

uniform mat4 m_view;
uniform mat4 m_proj;

out vec3 nearPoint;
out vec3 farPoint;
out mat4 view;
out mat4 proj;

vec3 UnprojectPoint(float x, float y, float z, mat4 view, mat4 proj) {
    mat4 viewInv = inverse(view);
    mat4 projInv = inverse(proj);
    vec4 unprojectedPoint =  viewInv * projInv * vec4(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main() {
    view = m_view;
    proj = m_proj;
    
    // Grid düzlemi (XY veya XZ) - Genelde XZ düzlemi zemin olur ama
    // Simülasyonumuzda Z aşağı, X ileri, Y sağ. Yani XY düzlemi yatay.
    // Ancak OpenGL'de XZ yataydır. Shader içinde dönüşüm yapacağız.
    
    // Full screen quad koordinatlarını world space'e unproject ediyoruz
    nearPoint = UnprojectPoint(in_position.x, in_position.y, 0.0, m_view, m_proj).xyz;
    farPoint = UnprojectPoint(in_position.x, in_position.y, 1.0, m_view, m_proj).xyz;
    
    gl_Position = vec4(in_position, 1.0);
}
