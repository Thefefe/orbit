#version 460

#include "common.glsl"
#include "types.glsl"

RegisterBuffer(PerFrameData, std430, readonly, {
    mat4 view_proj;
    uint render_mode;
});

RegisterBuffer(VertexBuffer, std430, readonly, {
    MeshVertex vertices[];
});

RegisterBuffer(InstanceBuffer, std430, readonly, {
    EntityInstance instances[];
});

RegisterBuffer(DrawCommands, std430, readonly, {
	uint draw_command_count;
    DrawCommand draw_commands[];
});

BindSlot(PerFrameData, 0);
BindSlot(VertexBuffer, 1);
BindSlot(InstanceBuffer, 2);
BindSlot(DrawCommands, 3);

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_tangent;
layout(location = 3) flat out uint out_material_index;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer).vertices[gl_VertexIndex];

    gl_Position =
        GetBuffer(PerFrameData).view_proj *
        GetBuffer(InstanceBuffer).instances[gl_InstanceIndex].model_matrix *
        vec4(vertex.pos, 1.0);

    out_uv = vertex.uv;
    out_normal = vertex.norm;
    out_tangent = vertex.tang;
    out_material_index = GetBuffer(DrawCommands).draw_commands[gl_DrawID].material_index; 
}