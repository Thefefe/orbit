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

layout(location = 0) out VertexOutput {
    vec2 uv;
    flat uint material_index;
} vout;

void main() {
    uint vertex_index =
        GetBuffer(MeshletDrawCommandBuffer, draw_commands_buffer).draws[gl_DrawID].meshlet_vertex_offset +
        GetBuffer(MeshletDataBuffer, meshlet_data_buffer).vertex_indices[gl_VertexIndex];
    float pos_array[3] = GetBuffer(VertexBuffer, vertex_buffer).vertices[vertex_index].position;
    vec3 pos = vec3(pos_array[0], pos_array[1], pos_array[2]);
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    gl_Position = view_proj * model_matrix * vec4(pos, 1.0);

    float uv_array[2] = GetBuffer(VertexBuffer, vertex_buffer).vertices[vertex_index].uv_coord;
    vout.uv = vec2(uv_array[0], uv_array[1]);
    uint meshlet_index = GetBuffer(MeshletDrawCommandBuffer, draw_commands_buffer).draws[gl_DrawID].meshlet_index;
    vout.material_index = GetBuffer(MeshletBuffer, meshlet_buffer).meshlets[meshlet_index].material_index;
}