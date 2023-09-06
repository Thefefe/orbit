#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant, std430) uniform PushConstants {
    ivec2 screen_size;
    uint per_frame_buffer;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands;
    uint materials_buffer;
    uint directional_light_buffer;
    uint light_count;
    uint light_data_buffer;
    uint irradiance_image_index;
    uint prefiltered_env_map_index;
    uint brdf_integration_map_index;
    uint jitter_texture_index;
};

layout(location = 0) out VertexOutput {
    vec4 world_pos;
    vec2 uv;
    vec3 normal;
    vec4 tangent;
    flat uint material_index;
} vout;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = model_matrix * vec4(vertex.pos, 1.0);

    gl_Position = GetBuffer(PerFrameBuffer, per_frame_buffer).data.view_projection * vout.world_pos;

    vout.uv = vertex.uv;
    
    mat3 normal_matrix = mat3(GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].normal_matrix);

    vout.normal = normalize(normal_matrix * vertex.norm);
    vout.tangent = vec4(normalize(normal_matrix * vertex.tang.xyz), vertex.tang.w);

    vout.material_index = GetBuffer(DrawCommandsBuffer, draw_commands).commands[gl_DrawID].material_index;
}