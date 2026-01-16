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
    outColor = texture(skyTexture, vec2(u, v));
    
    // Add fake sun glare if needed, but texture has it.
}
