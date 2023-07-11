#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    VertexBuffer vertex_buffer;
    EntityBuffer entity_buffer;
};

void main() {
    MeshVertex vertex = vertex_buffer.vertices[gl_VertexIndex];
    mat4 model_matrix = entity_buffer.entities[gl_InstanceIndex].model_matrix;

    gl_Position =
        view_proj *
        model_matrix *
        vec4(vertex.pos, 1.0);
}