#version 330 core

// Bloom extraction shader - extracts bright pixels

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D screenTexture;
uniform float threshold = 0.8;

void main() {
    vec3 color = texture(screenTexture, v_uv).rgb;
    
    // Calculate luminance
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Extract bright pixels
    if (luminance > threshold) {
        f_color = vec4(color, 1.0);
    } else {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
