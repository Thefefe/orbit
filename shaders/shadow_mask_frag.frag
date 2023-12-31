#version 460

#include "include/common.glsl"
#include "include/types.glsl"

RegisterFloatImageFormat(r8);

RegisterBuffer(ShadowMaskFragData, {
    mat4 view_projection_matrix;
    mat4 reprojection_matrix;
    uint vertex_buffer;
    uint meshlet_buffer;
    uint meshlet_data_buffer;
    uint entity_buffer;
    uint draw_commands_buffer;
    uint materials_buffer;
    uint camera_depth_buffer;
    uint shadow_mask;
    uint _padding[2];
});

layout(push_constant) uniform PushConstants {
   uint data_buffer;
};

layout(location = 0) in VertexOutput {
    vec3 pos;
    vec3 camera_pos;
    vec2 uv;
    flat uint material_index;
} vout;

void main() {
    uint materials_buffer = GetBuffer(ShadowMaskFragData, data_buffer).materials_buffer;
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

    vec3 camera_pos = vout.camera_pos;
    float camera_depth = texture(GetSampledTexture2D(GetBuffer(ShadowMaskFragData, data_buffer).camera_depth_buffer), camera_pos.xy).x;
    if (
        camera_pos.z >= 0.01 / (0.01 / camera_depth + 0.1)
        && 0.0 <= camera_pos.x && camera_pos.x <= 1.0
        && 0.0 <= camera_pos.y && camera_pos.y <= 1.0
        && 0.0 <= camera_pos.z && camera_pos.z <= 1.0
    ) {
        float mask_size = textureSize(GetSampledTexture2D(GetBuffer(ShadowMaskFragData, data_buffer).shadow_mask), 0).x;
        vec2 norm_mask_pos = vout.pos.xy * 0.5 + 0.5;
        norm_mask_pos.y = 1.0 - norm_mask_pos.y;
        ivec2 mask_pos = ivec2(norm_mask_pos * mask_size);
        imageStore(GetImage2D(r8, GetBuffer(ShadowMaskFragData, data_buffer).shadow_mask), mask_pos, vec4(0.0));
    }
}