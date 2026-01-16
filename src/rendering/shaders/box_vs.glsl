#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

uniform mat4 m_view;
uniform mat4 m_proj;
uniform mat4 m_model;
uniform vec3 u_color;

uniform mat4 lightSpaceMatrix;

out vec3 v_normal;
out vec3 v_fragPos;
out vec3 v_color;
out vec4 v_fragPosLightSpace;

void main() {
    v_color = u_color;
    v_normal = mat3(transpose(inverse(m_model))) * in_normal; 
    v_fragPos = vec3(m_model * vec4(in_position, 1.0));
    v_fragPosLightSpace = lightSpaceMatrix * vec4(v_fragPos, 1.0);
    
    gl_Position = m_proj * m_view * vec4(v_fragPos, 1.0);
}
