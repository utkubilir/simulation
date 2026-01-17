#version 330 core

in vec3 viewDir;
out vec4 outColor;

uniform sampler2D skyTexture;
uniform vec3 lightPos; // World space light position

const float PI = 3.14159265359;

void main() {
    vec3 d = normalize(viewDir);
    vec3 L = normalize(lightPos);
    
    // Convert 3D direction to Spherical Coordinates (Equirectangular) for texture lookup
    // d = (x, y, z) where +Z is Down (NED)
    // Pitch (Elevation) - Angle from horizontal plane (Z=0)
    // Up is -Z.
    float pitch = asin(-d.z); 
    float yaw = atan(d.y, d.x); 
    
    float u = 0.5 + yaw / (2.0 * PI);
    float v = 0.5 + pitch / PI;
    
    // Sample texture
    vec4 texColor = texture(skyTexture, vec2(u, v));
    
    // Fallback: If texture is unavailable (returns near-black), use procedural sky
    float texLuminance = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
    
    if (texLuminance < 0.01) {
        // Procedural sky gradient based on look direction vertical component (-d.z)
        // -d.z is 1.0 (Zenith), 0.0 (Horizon), -1.0 (Nadir) assuming Up is -Z
        float elevation = -d.z;
        
        vec3 horizonColor = vec3(0.6, 0.7, 0.9);   // Hazy blue/white
        vec3 zenithColor = vec3(0.1, 0.4, 0.8);    // Deep blue
        vec3 groundColor = vec3(0.3, 0.3, 0.3);    // Dark grey for bottom
        
        vec3 skyColor;
        if (elevation > 0.0) {
            skyColor = mix(horizonColor, zenithColor, pow(elevation, 0.5));
        } else {
            skyColor = mix(horizonColor, groundColor, clamp(-elevation * 5.0, 0.0, 1.0));
        }
        
        // Sun Disk
        // Angle between view direction and light direction
        float cosTheta = dot(d, L);
        
        // Sun Glow (Mie Scattering approximation)
        float sunGlow = exp(-0.1 / (1.001 - cosTheta)) * 1.0; 
        // Sun Disk (Sharp)
        float sunDisk = step(0.9995, cosTheta) * 5.0; // Very bright
        
        vec3 sunColor = vec3(1.0, 0.9, 0.7);
        vec3 finalColor = skyColor + sunColor * (sunGlow * 0.5 + sunDisk);
        
        texColor = vec4(finalColor, 1.0);
    }
    
    outColor = texColor;
}
