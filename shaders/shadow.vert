#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    uint vertex_buffer;
    uint entity_buffer;
};

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;

    gl_Position =
        view_proj *
        model_matrix *
        vec4(vertex.pos, 1.0);
}