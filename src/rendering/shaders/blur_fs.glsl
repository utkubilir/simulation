#version 330 core

// Gaussian blur shader for bloom

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D inputTexture;
uniform vec2 direction; // (1,0) for horizontal, (0,1) for vertical
uniform float blurSize = 1.0;

// 9-tap Gaussian weights
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 texelSize = 1.0 / textureSize(inputTexture, 0);
    vec3 result = texture(inputTexture, v_uv).rgb * weights[0];
    
    for (int i = 1; i < 5; i++) {
        vec2 offset = direction * texelSize * float(i) * blurSize;
        result += texture(inputTexture, v_uv + offset).rgb * weights[i];
        result += texture(inputTexture, v_uv - offset).rgb * weights[i];
    }
    
    f_color = vec4(result, 1.0);
}
