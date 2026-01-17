#version 330

in vec3 v_normal;
in vec3 v_fragPos;
in vec3 v_color;
in vec4 v_fragPosLightSpace;
in vec2 v_texcoord;

out vec4 f_color;

uniform vec3 lightPos = vec3(100.0, 1000.0, 100.0); // Sun (Top-Down)
uniform vec3 viewPos;
uniform sampler2D shadowMap;
uniform sampler2D u_texture;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        return 0.0;
        
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // PCF (Percentage-closer filtering)
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - 0.005 > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    return shadow;
}

void main() {
    // Texture sampling
    vec4 texColor = texture(u_texture, v_texcoord);
    
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    
    // Diffuse
    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(lightPos - v_fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 0.9);
    
    // Specular (Terrain is usually low specular)
    float specularStrength = 0.1;
    vec3 viewDir = normalize(viewPos - v_fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 8); // Lower shininess for dirt/grass
    vec3 specular = specularStrength * vec3(1.0, 1.0, 1.0) * spec;  
    
    // Shadow
    float shadow = ShadowCalculation(v_fragPosLightSpace, norm, lightDir);
    
    // Combine (Texture * Lighting)
    // We multiply result by v_color (tint) AND texColor
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular));
    
    // Simple AO: Darken valleys slightly
    float ao = clamp(v_fragPos.y / 15.0 + 0.7, 0.7, 1.0);
    vec3 result = lighting * v_color * texColor.rgb * ao;
    
    // Fog - Realistic Atmospheric Scattering
    float dist = length(viewPos - v_fragPos);
    float fogDensity = 0.0012;
    float fogFactor = 1.0 - exp(-dist * fogDensity);
    
    // Mix horizon haze color (matching sky_fs horizon)
    vec3 horizonHaze = vec3(0.6, 0.7, 0.9);
    result = mix(result, horizonHaze, clamp(fogFactor, 0.0, 1.0));
    
    f_color = vec4(result, 1.0);
}
