#version 460

#include "common.glsl"

struct MeshVertex {
    vec3 pos;
    uint _padding0;
    vec2 uv;
    uint _padding1;
    uint _padding2;
    vec3 norm;
    uint _padding3;
    vec4 tang;
};

RegisterBuffer(VertexBuffer, std430, readonly, {
    MeshVertex vertices[];
});

RegisterBuffer(Globals, std430, readonly, {
    mat4 viewProj;
});

BindSlot(VertexBuffer, 0);
BindSlot(Globals, 1);

layout(location = 0) out vec4 outColor;

void main() {
    vec3 world_pos = GetBuffer(VertexBuffer).vertices[gl_VertexIndex].pos;
    vec3 norm = GetBuffer(VertexBuffer).vertices[gl_VertexIndex].norm;

    gl_Position = GetBuffer(Globals).viewProj * vec4(world_pos, 1.0);
    // outColor = vec4((normalize(world_pos) + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
    outColor = vec4((norm + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
}