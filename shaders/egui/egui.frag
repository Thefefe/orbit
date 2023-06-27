#version 460

#include "../common.glsl"

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

BindSlot(input_texture, 2);

void main() {
    vec4 tex_color = texture(sampler2D(GetTexture(input_texture), _uSamplers[0]), in_uv);
    if (in_uv == vec2(0.0, 0.0)) { // HACK: this should use clamped samplers
        tex_color = vec4(1.0, 1.0, 1.0, 1.0);
    }
    out_color = in_color * tex_color;
    // outColor = vec4(inUV, 0.0, 1.0);
}