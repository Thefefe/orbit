#include "../include/common.glsl"
#include "../include/types.glsl"

layout(push_constant, std430) uniform PushConstants {
    mat4 matrix;
    uint image_index;
};

layout(location = 0) in vec3 in_local_pos;
layout(location = 0) out vec4 out_color;

const vec2 inv_atan = vec2(0.1591, 0.3183);
vec2 sample_spherical_map(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= inv_atan;
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

void main() {
    vec3 normal = normalize(in_local_pos);
    vec2 uv = sample_spherical_map(normal);
    out_color = vec4(texture(GetSampledTexture2D(image_index), uv).rgb, 1.0);
}