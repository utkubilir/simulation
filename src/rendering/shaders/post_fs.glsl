#version 330

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D screenTexture;
uniform sampler2D depthTexture;

uniform float distortionK1 = -0.2;
uniform float distortionK2 = 0.05;
uniform float chromaticAberration = 0.005;

// DoF Parameters
uniform float focusDistance = 0.5; // NDC: 0 (near) to 1 (far) or Linear Z
uniform float focusRange = 20.0;   // In meters (requires linearization)
uniform bool dofEnabled = true;

// Linearize Depth
float linearize_depth(float d, float zNear, float zFar) {
    return zNear * zFar / (zFar * d - zFar + zNear); // Basit versiyon (Projection matrisine bagli)
    // Standart OpenGL: z_ndc = 2.0 * d - 1.0;
    // z_eye = 2.0 * zNear * zFar / (zFar + zNear - z_ndc * (zFar - zNear));
    // Basit lineerizasyon (Perspective projection için):
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

void main() {
    vec2 dist_uv = distort(v_uv);
    
    if (dist_uv.x < 0.0 || dist_uv.x > 1.0 || dist_uv.y < 0.0 || dist_uv.y > 1.0) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Depth Fetch
    float depth = texture(depthTexture, dist_uv).r;
    
    // DoF Calculation (Basit Blur)
    vec3 color = vec3(0.0);
    
    if (dofEnabled) {
        // Lineer derinlik hesabı (Near=0.1, Far=1000.0)
        float zNear = 0.1;
        float zFar = 1000.0;
        // Depth texture [0, 1] range. OpenGL default is non-linear.
        // Convert to NDC [-1, 1]
        float z_ndc = depth * 2.0 - 1.0;
        float linear_z = (2.0 * zNear * zFar) / (zFar + zNear - z_ndc * (zFar - zNear));
        
        // Odak mesafesi farkı (Mutlak)
        float blurAmount = abs(linear_z - focusDistance) / focusRange;
        blurAmount = clamp(blurAmount, 0.0, 1.0);
        
        // Blur Kernel (3x3 veya 5x5 - Performans için küçük tutalım)
        // Blur miktarına göre offset ayarla
        float offset = blurAmount * 0.005; // Max blur radius
        
        vec2 offsets[9] = vec2[](
            vec2(-1, -1), vec2(0, -1), vec2(1, -1),
            vec2(-1,  0), vec2(0,  0), vec2(1,  0),
            vec2(-1,  1), vec2(0,  1), vec2(1,  1)
        );
        
        // Renk Kanalları (CA ile birleştirilmiş)
        vec3 sum = vec3(0.0);
        for(int i=0; i<9; i++) {
            vec2 sample_uv = dist_uv + offsets[i] * offset;
            
            // CA Logic inside blur loop (Heavy!) -> Move out or simplify
            // Simplify: Just blur the base texture, ignore CA for blurred parts?
            // Or better: Apply CA to the coordinate, then Blur.
            
            // Basit CA uygulanmış örnekleme
            vec2 center = vec2(0.5);
            vec2 d_vec = sample_uv - center;
            
            float r = texture(screenTexture, sample_uv + d_vec * chromaticAberration).r;
            float g = texture(screenTexture, sample_uv).g;
            float b = texture(screenTexture, sample_uv - d_vec * chromaticAberration).b;
            
            sum += vec3(r, g, b);
        }
        color = sum / 9.0;
        
    } else {
        // No DoF
        vec2 center = vec2(0.5);
        vec2 dist_vec = dist_uv - center;
        float r = texture(screenTexture, dist_uv + dist_vec * chromaticAberration).r;
        float g = texture(screenTexture, dist_uv).g;
        float b = texture(screenTexture, dist_uv - dist_vec * chromaticAberration).b;
        color = vec3(r, g, b);
    }
    
    f_color = vec4(color, 1.0);
}
