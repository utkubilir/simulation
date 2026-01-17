#version 330 core

in vec3 nearPoint;
in vec3 farPoint;

uniform mat4 m_view;
uniform mat4 m_proj;

out vec4 outColor;

uniform sampler2D groundTexture;
uniform float textureScale = 200.0; // Size of one texture tile in meters (Smaller = More repeats = More detail)

// Helpers for depth/fog
float computeDepth(vec3 pos, mat4 view, mat4 proj) {
    vec4 clip_space_pos = proj * view * vec4(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w);
}

void main() {
    // Intersect ray with Ground Plane (Z = 0)
    // Ray: P = near + t * (far - near)
    // P.z = 0 -> near.z + t * (far.z - near.z) = 0
    // t = -near.z / (far.z - near.z)
    
    float t = -nearPoint.z / (farPoint.z - nearPoint.z);
    
    if (t < 0.0) {
        discard; // Ray doesn't hit ground (looking up)
    }
    
    vec3 fragPos3D = nearPoint + t * (farPoint - nearPoint);
    
    // Depth for Z-buffer
    gl_FragDepth = computeDepth(fragPos3D, m_view, m_proj);
    // Actually standard method is to calculate depth from fragPos3D manually or let gl_FragDepth update
    // But since we are drawing a fullscreen quad, we MUST update gl_FragDepth manually.
    
    // UV Mapping
    // Map World X, Y to UV
    // Simple repeating tile
    vec2 uv = fragPos3D.xy / textureScale;
    
    // Sample texture
    vec4 texColor = texture(groundTexture, uv);
    
    // Fallback: If texture is unavailable (returns near-black), use a default green terrain color
    float texLuminance = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
    if (texLuminance < 0.01) {
        // Default green terrain color (grass-like)
        texColor = vec4(0.34, 0.49, 0.27, 1.0);
    }
    
    // Fog Logic (reuse from box/grid)
    float dist = length(fragPos3D - nearPoint); // approximate distance from camera
    float fogDensity = 0.002; // Less fog because we have huge view distance now (Haze=500m logic)
    // Real haze: 500m -> visibility drops.
    // Exp2 fog
    float fogFactor = 1.0 - exp(-(dist * fogDensity));
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    // Fog color should match sky bottom color?
    // Let's use a warm gray/blue
    vec3 fogColor = vec3(0.7, 0.75, 0.8);
    
    outColor = mix(texColor, vec4(fogColor, 1.0), fogFactor);
}
