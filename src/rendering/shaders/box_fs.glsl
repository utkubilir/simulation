#version 330

in vec3 v_normal;
in vec3 v_fragPos;
in vec3 v_color;

out vec4 f_color;

uniform vec3 lightPos = vec3(100.0, -200.0, -200.0); // Güneş gibi (yukarıdan)
uniform vec3 viewPos;

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
    
    vec3 result = (ambient + diffuse + specular) * v_color;
    
    // Fog
    float dist = length(viewPos - v_fragPos);
    float fogDensity = 0.04;
    float fogFactor = 1.0 - exp(-dist * fogDensity);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    vec3 fogColor = vec3(0.5, 0.7, 0.9);
    result = mix(result, fogColor, fogFactor);
    
    f_color = vec4(result, 1.0);
}
