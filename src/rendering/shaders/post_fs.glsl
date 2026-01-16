#version 330

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D screenTexture;
uniform sampler2D depthTexture;

uniform float distortionK1 = -0.2;
uniform float distortionK2 = 0.05;
uniform float chromaticAberration = 0.005;

// New Realism Parameters
uniform float time = 0.0;
uniform float vignetteStrength = 0.5;
uniform float noiseStrength = 0.05;

// DoF Parameters
uniform float focusDistance = 0.5; // NDC: 0 (near) to 1 (far) or Linear Z
uniform float focusRange = 20.0;   // In meters (requires linearization)
uniform bool dofEnabled = true;

// Pseudo-random number generator
float rand(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// Linearize Depth
float linearize_depth(float d, float zNear, float zFar) {
    return (2.0 * zNear) / (zFar + zNear - d * (zFar - zNear));
}

// Brown-Conrady Distortion
vec2 distort(vec2 uv) {
    vec2 pos = uv * 2.0 - 1.0;
    float r2 = dot(pos, pos);
    float r4 = r2 * r2;
    float factor = 1.0 + distortionK1 * r2 + distortionK2 * r4;
    pos *= factor;
    return (pos + 1.0) * 0.5;
}

// ACES Tone Mapping (Standard for realistic rendering)
vec3 aces_tone_mapping(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec2 dist_uv = distort(v_uv);
    
    // Black borders for distorted UVs outside [0,1]
    if (dist_uv.x < 0.0 || dist_uv.x > 1.0 || dist_uv.y < 0.0 || dist_uv.y > 1.0) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Depth Fetch
    float depth = texture(depthTexture, dist_uv).r;
    
    // DoF Calculation (Basit Blur)
    vec3 color = vec3(0.0);
    
    if (dofEnabled) {
        float zNear = 0.1;
        float zFar = 1000.0;
        float z_ndc = depth * 2.0 - 1.0;
        float linear_z = (2.0 * zNear * zFar) / (zFar + zNear - z_ndc * (zFar - zNear));
        
        float blurAmount = abs(linear_z - focusDistance) / focusRange;
        blurAmount = clamp(blurAmount, 0.0, 1.0);
        
        float offset = blurAmount * 0.005; 
        
        vec2 offsets[9] = vec2[](
            vec2(-1, -1), vec2(0, -1), vec2(1, -1),
            vec2(-1,  0), vec2(0,  0), vec2(1,  0),
            vec2(-1,  1), vec2(0,  1), vec2(1,  1)
        );
        
        vec3 sum = vec3(0.0);
        for(int i=0; i<9; i++) {
            vec2 sample_uv = dist_uv + offsets[i] * offset;
            
            // Chromatic Aberration inside blur
            vec2 center = vec2(0.5);
            vec2 d_vec = sample_uv - center;
            
            float r = texture(screenTexture, sample_uv + d_vec * chromaticAberration).r;
            float g = texture(screenTexture, sample_uv).g;
            float b = texture(screenTexture, sample_uv - d_vec * chromaticAberration).b;
            sum += vec3(r, g, b);
        }
        color = sum / 9.0;
        
    } else {
        // No DoF, just Chromatic Aberration
        vec2 center = vec2(0.5);
        vec2 dist_vec = dist_uv - center;
        float r = texture(screenTexture, dist_uv + dist_vec * chromaticAberration).r;
        float g = texture(screenTexture, dist_uv).g;
        float b = texture(screenTexture, dist_uv - dist_vec * chromaticAberration).b;
        color = vec3(r, g, b);
    }

    // --- NEW EFFECTS ---

    // 1. Film Grain / Noise
    float noise = rand(dist_uv + time) * noiseStrength;
    color += noise;

    // 2. Vignette
    vec2 center = vec2(0.5);
    float dist = distance(dist_uv, center);
    float vignette = smoothstep(0.8, 0.8 - vignetteStrength, dist * (1.0 + vignetteStrength));
    color *= vignette;
    
    // 3. Tone Mapping (ACES)
    color = aces_tone_mapping(color);

    // Gamma Correction (Linear -> SRGB)
    color = pow(color, vec3(1.0/2.2));
    
    f_color = vec4(color, 1.0);
}
