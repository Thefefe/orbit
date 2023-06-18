#version 460

#include "common.glsl"

RegisterBuffer(PerFrameData, std430, readonly, {
    mat4 view_proj;
    uint render_mode;
});

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

struct EntityInstance {
    mat4 model_matrix;
};


RegisterBuffer(InstanceBuffer, std430, readonly, {
    EntityInstance instances[];
});

BindSlot(PerFrameData, 0);
BindSlot(VertexBuffer, 1);
BindSlot(InstanceBuffer, 2);

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_tangent;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer).vertices[gl_VertexIndex];

    gl_Position =
        GetBuffer(PerFrameData).view_proj *
        GetBuffer(InstanceBuffer).instances[gl_InstanceIndex].model_matrix *
        vec4(vertex.pos, 1.0);
    // outColor = vec4((normalize(world_pos) + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
    

    out_uv = vertex.uv;
    out_normal = vertex.norm;
    out_tangent = vertex.tang;
}