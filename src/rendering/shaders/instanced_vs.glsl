#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

// Instance attributes (per-instance)
// mat4 takes 4 attribute slots (locations 2,3,4,5)
layout (location = 2) in mat4 in_instanceMatrix; 
layout (location = 6) in vec3 in_instanceColor;

uniform mat4 m_view;
uniform mat4 m_proj;
uniform mat4 lightSpaceMatrix;

out vec3 v_normal;
out vec3 v_fragPos;
out vec3 v_color;
out vec4 v_fragPosLightSpace;

void main() {
    v_color = in_instanceColor;
    mat4 model = in_instanceMatrix;
    
    // Normal matrix calculation (expensive per vertex, but improved by not doing it in CPU)
    v_normal = mat3(transpose(inverse(model))) * in_normal;
    
    v_fragPos = vec3(model * vec4(in_position, 1.0));
    v_fragPosLightSpace = lightSpaceMatrix * vec4(v_fragPos, 1.0);
    
    gl_Position = m_proj * m_view * vec4(v_fragPos, 1.0);
}
