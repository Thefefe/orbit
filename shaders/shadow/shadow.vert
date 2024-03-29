#version 460

#include "../include/common.glsl"
#include "../include/types.glsl"

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

layout(location = 0) out VertexOutput {
    vec2 uv;
    flat uint material_index;
} vout;

void main() {
    uint vertex_index =
        GetBuffer(MeshletDrawCommandBuffer, draw_command_buffer).draws[gl_DrawID].meshlet_vertex_offset +
        GetBuffer(MeshletDataBuffer, meshlet_data_buffer).vertex_indices[gl_VertexIndex];

    vec3 pos = GetBuffer(VertexBuffer, vertex_buffer).vertices[vertex_index].position;
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    gl_Position = view_proj * model_matrix * vec4(pos, 1.0);

    vout.uv = GetBuffer(VertexBuffer, vertex_buffer).vertices[vertex_index].uv;
    uint meshlet_index = GetBuffer(MeshletDrawCommandBuffer, draw_command_buffer).draws[gl_DrawID].meshlet_index;
    vout.material_index = GetBuffer(MeshletBuffer, meshlet_buffer).meshlets[meshlet_index].material_index;
}