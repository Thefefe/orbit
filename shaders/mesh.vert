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

RegisterBuffer(EntityBuffer, std430, readonly, {
    EntityData entities[];
});

RegisterBuffer(DrawCommands, std430, readonly, {
	uint draw_command_count;
    DrawCommand draw_commands[];
});

BindSlot(PerFrameData, 0);
BindSlot(VertexBuffer, 1);
BindSlot(EntityBuffer, 2);
BindSlot(DrawCommands, 3);

layout(location = 0) out VertexOutput {
    vec2 uv;
    mat3 TBN;
    flat uint material_index;
} vout;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer).entities[gl_InstanceIndex].model_matrix;

    gl_Position =
        GetBuffer(PerFrameData).view_proj *
        model_matrix *
        vec4(vertex.pos, 1.0);

    vout.uv = vertex.uv;

    vec3 normal = vertex.norm;
    vec3 tangent = vertex.tang.xyz;
    vec3 bitangent = cross(normal, tangent) * vertex.tang.w;
    
    mat3 normal_matrix = mat3(GetBuffer(EntityBuffer).entities[gl_InstanceIndex].normal_matrix);

    vout.TBN = mat3(
        normalize(vec3(normal_matrix * tangent)),
        normalize(vec3(normal_matrix * bitangent)),
        normalize(vec3(normal_matrix * normal))
    );

    vout.material_index = GetBuffer(DrawCommands).draw_commands[gl_DrawID].material_index;
}