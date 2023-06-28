#version 460

#include "common.glsl"
#include "types.glsl"

RegisterBuffer(PerFrameBuffer, std430, readonly, {
    PerFrameData data;
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

BindSlot(PerFrameBuffer, 0);
BindSlot(VertexBuffer, 1);
BindSlot(EntityBuffer, 2);
BindSlot(DrawCommands, 3);

layout(location = 0) out VertexOutput {
    vec3 world_pos;
    vec2 uv;
    mat3 TBN;
    flat uint material_index;
} vout;

vec4 pbr() {
    return vec4(1.0);
}

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer).entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = vec3(model_matrix * vec4(vertex.pos, 1.0));

    gl_Position =
        GetBuffer(PerFrameBuffer).data.view_proj *
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