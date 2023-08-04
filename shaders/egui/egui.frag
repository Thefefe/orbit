#version 460

#include "../include/common.glsl"

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(push_constant, std430) uniform PushConstants {
    vec2 screen_size;
    uint input_texture;
};

void main() {
    vec4 tex_color = texture(sampler2D(GetTexture2D(input_texture), GetSampler(0)), in_uv);
    out_color = in_color * tex_color;
}