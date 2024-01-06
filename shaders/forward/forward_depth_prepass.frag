#version 460

#include "../include/common.glsl"
#include "../include/types.glsl"
#include "../include/functions.glsl"

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    uint draw_command_buffer;
    uint cull_info_buffer;
    uint vertex_buffer;
    uint meshlet_buffer;
    uint meshlet_data_buffer;
    uint entity_buffer;
    uint materials_buffer;
};

layout(location = 0) in VertexOutput {
    vec2 uv;
    flat uint material_index;
} vout;

layout(location = 0) out vec4 out_color;

#define MIP_SCALE 0.25

float calc_mip_level(vec2 texture_coord) {
    vec2 dx = dFdx(texture_coord);
    vec2 dy = dFdy(texture_coord);
    float delta_max_sqr = max(dot(dx, dx), dot(dy, dy));
    
    return max(0.0, 0.5 * log2(delta_max_sqr));
}

void main() {
    out_color = vec4(1.0);

    if (GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index].alpha_mode == 1) {
        float alpha_cutoff = GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index].alpha_cutoff;
        float alpha = GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index].base_color.a;
        uint texture_index = GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index].base_texture_index;
        
        // avoid artifacts at edge cases
        if (alpha_cutoff >= 1.0 || (texture_index == TEXTURE_NONE && alpha <= alpha_cutoff)) {
            discard;
        }
        
        if (texture_index != TEXTURE_NONE) {
            alpha *= texture(GetSampledTexture2D(texture_index), vout.uv).a;
            
            // minor bias to minimize some totally random artifact that
            // i spent half a week debuging and don't want to spend more
            alpha -= 0.001;

            vec2 texture_size = textureSize(GetSampledTexture2D(texture_index), 0);
            alpha *= 1 + max(0, calc_mip_level(vout.uv * texture_size)) * MIP_SCALE;
            out_color.a = (alpha - alpha_cutoff) / max(fwidth(alpha), 0.0001) + 0.5;
        }
    }
}