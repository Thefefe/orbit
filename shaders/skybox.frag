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
    out_color = vec4(texture(GetSampledTextureCube(image_index), normal).rgb, 1.0);
}