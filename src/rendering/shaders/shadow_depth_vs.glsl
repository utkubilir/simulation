#version 330 core
layout (location = 0) in vec3 in_position;

uniform mat4 lightSpaceMatrix;
uniform mat4 m_model;

void main()
{
    gl_Position = lightSpaceMatrix * m_model * vec4(in_position, 1.0);
}
