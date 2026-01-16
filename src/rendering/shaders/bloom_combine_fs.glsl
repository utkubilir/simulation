#version 330 core

// Bloom combine shader - adds blurred bloom to original

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D sceneTexture;
uniform sampler2D bloomTexture;
uniform float bloomIntensity = 0.5;
uniform float exposure = 1.0;

// ACES Tone Mapping
vec3 aces_tone_mapping(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 scene = texture(sceneTexture, v_uv).rgb;
    vec3 bloom = texture(bloomTexture, v_uv).rgb;
    
    // Combine with HDR
    vec3 hdr = scene + bloom * bloomIntensity;
    
    // Exposure
    hdr *= exposure;
    
    // Tone mapping
    vec3 ldr = aces_tone_mapping(hdr);
    
    // Gamma correction
    ldr = pow(ldr, vec3(1.0 / 2.2));
    
    f_color = vec4(ldr, 1.0);
}
