#version 460

#include "../common.glsl"

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec4 outColor;

// layout(binding = 0, set = 0) uniform sampler2D font_texture;

BindSlot(input_texture, 4);

void main() {
    // outColor = inColor * texture(sampler2D(GetTexture(input_texture), _uSamplers[0]), inUV);
    outColor = inColor;
}