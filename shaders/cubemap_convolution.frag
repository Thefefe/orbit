#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant, std430) uniform PushConstants {
    mat4 matrix;
    uint image_index;
};

layout(location = 0) in vec3 in_local_pos;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 normal = normalize(in_local_pos);
    vec3 irradiance = vec3(0.0);

    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up         = normalize(cross(normal, right));

    float sampleDelta = 0.025;
    float nrSamples = 0.0; 
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * in_local_pos; 

            irradiance += texture(GetSampledTextureCube(image_index), sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples += 1;
        }
    }
    irradiance = PI * irradiance * (1.0 / float(nrSamples));

    out_color = vec4(irradiance, 1.0);
}