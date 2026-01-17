#version 330 core

in vec3 viewDir;
out vec4 outColor;

uniform sampler2D skyTexture;

const float PI = 3.14159265359;

void main() {
    vec3 d = normalize(viewDir);
    
    // Convert 3D direction to Spherical Coordinates (Equirectangular)
    // d = (x, y, z) -> NED: x=North, y=East, z=Down?
    // Simulation Frame: x=Forward, y=Right, z=Down
    // Standard Math: x=Right, y=Up, z=Back
    
    // Let's assume input 'viewDir' is in the Simulation World Frame.
    // We map this to UVs [0..1] x [0..1]
    
    // Pitch (Elevation) - Angle from horizontal plane
    // dot(d, up). Up is -Z in NED? or +Z?
    // Sim: +Z is Down. Up is -Z.
    float pitch = asin(-d.z); // -d.z is projection onto Up (+Z world is down)
    
    // Yaw (Azimuth)
    float yaw = atan(d.y, d.x); // standard atan2(y, x)
    
    // Map to UV
    // u = (yaw + PI) / (2PI)
    // v = (pitch + PI/2) / PI -> 0 at bottom (-Z), 1 at top (+Z)?
    // pitch range [-PI/2, PI/2]
    
    float u = 0.5 + yaw / (2.0 * PI);
    float v = 0.5 + pitch / PI;
    
    // Sample texture
    vec4 texColor = texture(skyTexture, vec2(u, v));
    
    // Fallback: If texture is unavailable (returns near-black), use procedural sky gradient
    float texLuminance = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
    if (texLuminance < 0.01) {
        // Procedural sky gradient based on elevation
        // v ranges from 0 (bottom/horizon) to 1 (top/zenith)
        vec3 horizonColor = vec3(0.53, 0.81, 0.92);  // Light blue at horizon
        vec3 zenithColor = vec3(0.25, 0.52, 0.96);   // Deeper blue at zenith
        vec3 skyGradient = mix(horizonColor, zenithColor, clamp(v, 0.0, 1.0));
        texColor = vec4(skyGradient, 1.0);
    }
    
    outColor = texColor;
    
    // Add fake sun glare if needed, but texture has it.
}
