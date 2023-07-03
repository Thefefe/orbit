#version 460

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec4 vertex_color;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    mat4 view_projection;
    float multiplier;
};

void main() {
    vec4 position = view_projection * vec4(vertex_position, 1.0);
    gl_Position = position;
    out_color = vertex_color;
}