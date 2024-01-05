#version 460

#include "../include/common.glsl"
#include "../include/types.glsl"
#include "../include/functions.glsl"

#include "forward_common.glsl"

layout(location = 0) out VERTEX_OUTPUT vout;

void main() {
    uint vertex_index =
        GetBuffer(MeshletDrawCommandBuffer, draw_commands_buffer).draws[gl_DrawID].meshlet_vertex_offset +
        GetBuffer(MeshletDataBuffer, meshlet_data_buffer).vertex_indices[gl_VertexIndex];
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[vertex_index];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = model_matrix * vec4(vertex.position[0], vertex.position[1], vertex.position[2], 1.0);

    gl_Position = GetBuffer(PerFrameBuffer, per_frame_buffer).view_projection * vout.world_pos;

    vout.uv = vec2(vertex.uv_coord[0], vertex.uv_coord[1]);
    
    mat3 normal_matrix = mat3(GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].normal_matrix);

    unpack_normal_tangent(vertex.packed_normals, vout.normal, vout.tangent);
    vout.normal  = normalize(normal_matrix * vout.normal);
    vout.tangent = vec4(normalize(normal_matrix * vout.tangent.xyz), vout.tangent.w);

    uint meshlet_index = GetBuffer(MeshletDrawCommandBuffer, draw_commands_buffer).draws[gl_DrawID].meshlet_index;
    uint material_index = GetBuffer(MeshletBuffer, meshlet_buffer).meshlets[meshlet_index].material_index;
    vout.material_index = GetBuffer(PerFrameBuffer, per_frame_buffer).render_mode == 9 ? meshlet_index : material_index;
}