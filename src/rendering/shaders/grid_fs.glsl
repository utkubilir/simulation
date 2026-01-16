#version 330

in vec3 nearPoint;
in vec3 farPoint;
in mat4 view;
in mat4 proj;

out vec4 outColor;

uniform float gridScale = 50.0; // Metre cinsinden grid aralığı

vec4 grid(vec3 fragPos3D, float scale) {
    vec2 coord = fragPos3D.xy / scale; // X ve Y koordinatlarını kullan (Simülasyon Z=Aşağı, XY yatay)
    vec2 derivative = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    float line = min(grid.x, grid.y);
    float minimumz = min(derivative.y, 1.0);
    float minimumx = min(derivative.x, 1.0);
    
    vec4 color = vec4(0.2, 0.2, 0.2, 1.0 - min(line, 1.0));
    
    // X ekseni (Kırmızı)
    if(abs(fragPos3D.y) < 2.0 * minimumx)
        color.x = 1.0;
        
    // Y ekseni (Yeşil) - Zemin ekseni
    if(abs(fragPos3D.x) < 2.0 * minimumz)
        color.y = 1.0;
        
    return color;
}

float computeDepth(vec3 pos) {
    vec4 clip_space_pos = proj * view * vec4(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w);
}

void main() {
    float t = -nearPoint.z / (farPoint.z - nearPoint.z);
    
    // Z = 0 düzlemi (yerden yükseklik?)
    // Simülasyon Z ekseni yükseklik (negatif yukarı, pozitif aşağı).
    // Varsayım: Zemin Z=0 değil, belki Z=H.
    // Şimdilik Z=0 düzlemini çizelim.
    
    // Ground plane intersection
    // Simülasyon: Z ekseni dikey. O yüzden Z bileşeni ile intersection bakıyoruz.
    vec3 fragPos3D = nearPoint + t * (farPoint - nearPoint);
    
    gl_FragDepth = computeDepth(fragPos3D);
    
    // Fade out distance (Fog ile birleştirilecek)
    float linearDepth = computeDepth(fragPos3D);
    float fogDensity = 0.04; // Fog yoğunluğu
    float fogFactor = 1.0 - exp(-linearDepth * fogDensity * 1.5);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    vec3 result = (grid(fragPos3D, gridScale) + grid(fragPos3D, gridScale/10.0)*0.5).rgb;
    result *= float(t > 0);
    
    vec3 fogColor = vec3(0.5, 0.7, 0.9); // Sky color
    vec3 finalColor = mix(result, fogColor, fogFactor);
    
    outColor = vec4(finalColor, max(0, (1.0 - linearDepth)) * (1.0 - fogFactor)); // Alpha fade + fog fade
}
