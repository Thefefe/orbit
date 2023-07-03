#version 460

layout(push_constant) uniform PushConstants {
    mat4 view_projection;
    float multiplier;
};

layout(location = 0) in vec4 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = in_color * multiplier;
}