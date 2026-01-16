#version 330


in vec3 v_normal;
in vec3 v_fragPos;
in vec3 v_color;
in vec4 v_fragPosLightSpace;

out vec4 f_color;

uniform vec3 lightPos = vec3(100.0, -200.0, -200.0); // Güneş gibi (yukarıdan)
uniform vec3 viewPos;
uniform sampler2D shadowMap;

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
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    
    // Diffuse
    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(lightPos - v_fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 0.9);
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - v_fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * vec3(1.0, 1.0, 1.0);  
    
    // Shadow
    float shadow = ShadowCalculation(v_fragPosLightSpace, norm, lightDir);
    vec3 result = (ambient + (1.0 - shadow) * (diffuse + specular)) * v_color;
    
    // Fog
    float dist = length(viewPos - v_fragPos);
    float fogDensity = 0.04;
    float fogFactor = 1.0 - exp(-dist * fogDensity);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    vec3 fogColor = vec3(0.5, 0.7, 0.9);
    result = mix(result, fogColor, fogFactor);
    
    f_color = vec4(result, 1.0);
}
