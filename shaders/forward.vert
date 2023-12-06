#version 460

#include "include/common.glsl"
#include "include/types.glsl"
#include "include/functions.glsl"

layout(push_constant, std430) uniform PushConstants {
    uint per_frame_buffer;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands;
    uint materials_buffer;
    uint cluster_buffer;
    uint light_count;
    uint light_data_buffer;
    uint shadow_data_buffer;
    uint shadow_settings_buffer;
    uint selected_light;
    uint brdf_integration_map_index;
    uint jitter_texture_index;
};

layout(location = 0) out VertexOutput {
    vec4 world_pos;
    vec2 uv;
    vec3 normal;
    vec4 tangent;
    flat uint material_index;
} vout;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = model_matrix * vec4(vertex.position[0], vertex.position[1], vertex.position[2], 1.0);

    gl_Position = GetBuffer(PerFrameBuffer, per_frame_buffer).view_projection * vout.world_pos;

    vout.uv = vec2(vertex.uv_coord[0], vertex.uv_coord[1]);
    
    mat3 normal_matrix = mat3(GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].normal_matrix);

    unpack_normal_tangent(vertex.packed_normals, vout.normal, vout.tangent);
    vout.normal  = normalize(normal_matrix * vout.normal);
    vout.tangent = vec4(normalize(normal_matrix * vout.tangent.xyz), vout.tangent.w);

    vout.material_index = GetBuffer(DrawCommandsBuffer, draw_commands).commands[gl_DrawID].material_index;
}