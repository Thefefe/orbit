#version 460

#include "common.glsl"

struct Vertex {
    vec2 pos;
    vec2 uv;
    vec4 color;
};

RegisterBuffer(VertexBuffer, std430, readonly, {
    Vertex vertices[];
});

BindSlot(VertexBuffer, 0);

layout(location = 0) out vec4 outColor;

void main() {
    vec2 position = GetBuffer(VertexBuffer).vertices[gl_VertexIndex].pos;
    vec4 color = GetBuffer(VertexBuffer).vertices[gl_VertexIndex].color;

    gl_Position = vec4(position, 0.0, 1.0);
    outColor = color;
}