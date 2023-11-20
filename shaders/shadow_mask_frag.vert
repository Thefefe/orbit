#version 460

#include "include/common.glsl"
#include "include/types.glsl"

RegisterBuffer(ShadowMaskFragData, {
    mat4 view_projection_matrix;
    mat4 reprojection_matrix;
    uint vertex_buffer;
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

layout(location = 0) out VertexOutput {
    vec3 pos;
    vec3 camera_pos;
    vec2 uv;
    flat uint material_index;
} vout;

void main() {
    float pos_array[3] = GetBuffer(VertexBuffer, GetBuffer(ShadowMaskFragData, data_buffer).vertex_buffer).vertices[gl_VertexIndex].position;
    vec3 pos = vec3(pos_array[0], pos_array[1], pos_array[2]);
    mat4 model_matrix = GetBuffer(EntityBuffer, GetBuffer(ShadowMaskFragData, data_buffer).entity_buffer).entities[gl_InstanceIndex].model_matrix;
    gl_Position = GetBuffer(ShadowMaskFragData, data_buffer).view_projection_matrix * model_matrix * vec4(pos, 1.0);

    float uv_array[2] = GetBuffer(VertexBuffer, GetBuffer(ShadowMaskFragData, data_buffer).vertex_buffer).vertices[gl_VertexIndex].uv_coord;
    vout.uv = vec2(uv_array[0], uv_array[1]);
    vout.material_index = GetBuffer(DrawCommandsBuffer, GetBuffer(ShadowMaskFragData, data_buffer).draw_commands_buffer)
        .commands[gl_DrawID].material_index;

    vout.pos = gl_Position.xyz / gl_Position.w;
    vec4 camera_pos = GetBuffer(ShadowMaskFragData, data_buffer).reprojection_matrix * vec4(vout.pos, 1.0);
    camera_pos /= camera_pos.w;
    camera_pos.xy = camera_pos.xy * 0.5 + 0.5;
    camera_pos.y = 1.0 - camera_pos.y;
    vout.camera_pos = camera_pos.xyz;
}