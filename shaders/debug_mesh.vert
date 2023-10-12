#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    mat4 view_projection_matrix;
    uint vertex_buffer;
    uint instance_buffer;
    float color_multiplier;
};

void main() {
    vec3 position = vec3(
        GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex].position[0],
        GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex].position[1],
        GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex].position[2]
    );
    DebugMeshInstance instance = GetBuffer(DebugMeshInstanceBuffer, instance_buffer).instances[gl_InstanceIndex];
    gl_Position = view_projection_matrix * instance.matrix * vec4(position, 1.0);
    gl_Position.z += 0.00001;
    out_color = vec4(instance.color.xyz * color_multiplier, 1.0);
}