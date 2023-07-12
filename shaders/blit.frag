#include "include/common.glsl"
#include "include/types.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant, std430) uniform PushConstants {
    uint image_index;
    float exposure;
};

void main() {
    vec3 hdr_color = texture(GetSampledTexture2D(image_index), in_uv).rgb;
    vec3 mapped = vec3(1.0) - exp(-hdr_color * exposure);
    out_color = vec4(mapped, 1.0);
}