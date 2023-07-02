#version 460

#include "common1.glsl"
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

layout(push_constant, std430) uniform PushConstants {
    uint per_frame_buffer;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands;
    uint materials;
    
    uint _padding0;
    uint _padding1;
    uint _padding2;

    mat4 light_projection;
    vec3 light_direction;
    uint light_shadow_map;
};

layout(location = 0) out VertexOutput {
    vec4 world_pos;
    vec2 uv;
    mat3 TBN;
    flat uint material_index;
    vec4 light_space_frag_pos;
} vout;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = model_matrix * vec4(vertex.pos, 1.0);

    gl_Position = GetBuffer(PerFrameBuffer, per_frame_buffer).data.view_projection * vout.world_pos;

    vout.uv = vertex.uv;
    vout.light_space_frag_pos = light_projection * vout.world_pos;

    vec3 normal = vertex.norm;
    vec3 tangent = vertex.tang.xyz;
    vec3 bitangent = cross(normal, tangent) * vertex.tang.w;
    
    mat3 normal_matrix = mat3(GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].normal_matrix);

    vout.TBN = mat3(
        normalize(vec3(normal_matrix * tangent)),
        normalize(vec3(normal_matrix * bitangent)),
        normalize(vec3(normal_matrix * normal))
    );

    vout.material_index = GetBuffer(DrawCommands, draw_commands).draw_commands[gl_DrawID].material_index;
}