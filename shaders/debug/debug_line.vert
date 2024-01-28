#version 460

#include "../include/common.glsl"
#include "../include/types.glsl"

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    mat4 view_projection;
    float multiplier;
    uint vertex_buffer;
    uint vertex_offset;
};

void main() {
    DebugLineVertex vertex = GetBuffer(DebugLineVertexBuffer, vertex_buffer).vertices[gl_VertexIndex + vertex_offset];
    vec4 position = view_projection * vec4(vertex.position, 1.0);
    gl_Position = position;
    out_color = vec4(vertex.color) / 255.0 * multiplier;
}