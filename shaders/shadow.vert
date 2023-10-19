#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands_buffer;
    uint materials_buffer;
};

layout(location = 0) out VertexOutput {
    vec2 uv;
    flat uint material_index;
} vout;

void main() {
    float pos_array[3] = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex].position;
    vec3 pos = vec3(pos_array[0], pos_array[1], pos_array[2]);
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    gl_Position = view_proj * model_matrix * vec4(pos, 1.0);

    float uv_array[2] = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex].uv_coord;
    vout.uv = vec2(uv_array[0], uv_array[1]);
    vout.material_index = GetBuffer(DrawCommandsBuffer, draw_commands_buffer)
        .commands[gl_DrawID].material_index;
}