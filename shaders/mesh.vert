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

RegisterBuffer(EntityIndexBuffer, std430, readonly, {
    uint indices[];
});

struct DrawData {
    // DrawIndiexedIndirectCommand
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;

    uint first_instance;
    // other per-draw data
    uint material_index;
    vec2 _padding;
};

RegisterBuffer(DrawCommands, std430, readonly, {
    DrawData draw_commands[];
});

BindSlot(PerFrameData, 0);
BindSlot(VertexBuffer, 1);
BindSlot(InstanceBuffer, 2);
BindSlot(EntityIndexBuffer, 3);
BindSlot(DrawCommands, 4);

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_tangent;
layout(location = 3) flat out uint out_material_index;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer).vertices[gl_VertexIndex];
    uint entity_index = GetBuffer(EntityIndexBuffer).indices[gl_InstanceIndex];

    gl_Position =
        GetBuffer(PerFrameData).view_proj *
        GetBuffer(InstanceBuffer).instances[entity_index].model_matrix *
        vec4(vertex.pos, 1.0);

    out_uv = vertex.uv;
    out_normal = vertex.norm;
    out_tangent = vertex.tang;
    out_material_index = GetBuffer(DrawCommands).draw_commands[gl_DrawID].material_index; 
}