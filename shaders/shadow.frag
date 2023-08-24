#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands_buffer;
    uint material_buffer;
};

layout(location = 0) in VertexOutput {
    vec2 uv;
    float alpha_cutoff;
    uint texture_index;
} vout;


void main() {
    float alpha = 1.0;
    
    if (vout.texture_index != TEXTURE_NONE) {
        alpha = texture(GetSampledTexture2D(vout.texture_index), vout.uv).a;
    }

    if (alpha < vout.alpha_cutoff) {
        discard;
    }
}