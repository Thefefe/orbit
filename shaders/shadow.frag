#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    uint vertex_buffer;
    uint meshlet_buffer;
    uint meshlet_data_buffer;
    uint entity_buffer;
    uint draw_commands_buffer;
    uint materials_buffer;
};

layout(location = 0) in VertexOutput {
    vec2 uv;
    flat uint material_index;
} vout;


void main() {
    if (GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index].alpha_mode == 1) {
        float alpha_cutoff = GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index].alpha_cutoff;
        float alpha = GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index].base_color.a;

        if (alpha <= alpha_cutoff) discard;

        uint texture_index = GetBuffer(MaterialsBuffer, materials_buffer)
            .materials[vout.material_index]
            .base_texture_index;
        if (texture_index != TEXTURE_NONE) {
            alpha *= texture(GetSampledTexture2D(texture_index), vout.uv).a;
            if (alpha <= alpha_cutoff) discard;
        }
    }
}