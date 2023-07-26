#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant, std430) uniform PushConstants {
    uint per_frame_buffer;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands;
    uint materials_buffer;
    uint directional_light_buffer;
    uint irradiance_image_index;
    uint prefiltered_env_map_index;
    uint brdf_integration_map_index;
};

layout(location = 0) out VertexOutput {
    vec4 world_pos;
    vec4 view_pos;
    vec2 uv;
    mat3 TBN;
    flat uint material_index;
    vec4 cascade_map_coords[MAX_SHADOW_CASCADE_COUNT];
} vout;

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = model_matrix * vec4(vertex.pos, 1.0);

    gl_Position = GetBuffer(PerFrameBuffer, per_frame_buffer).data.view_projection * vout.world_pos;

    vout.uv = vertex.uv;
    vout.view_pos = GetBuffer(PerFrameBuffer, per_frame_buffer).data.view * -vout.world_pos;

    vec3 normal = vertex.norm;
    vec3 tangent = vertex.tang.xyz;
    vec3 bitangent = cross(normal, tangent) * vertex.tang.w;
    
    mat3 normal_matrix = mat3(GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].normal_matrix);

    vout.TBN = mat3(
        normalize(vec3(normal_matrix * tangent)),
        normalize(vec3(normal_matrix * bitangent)),
        normalize(vec3(normal_matrix * normal))
    );

    vout.material_index = GetBuffer(DrawCommandsBuffer, draw_commands).commands[gl_DrawID].material_index;

    for (uint i = 0; i < MAX_SHADOW_CASCADE_COUNT; ++i) {
        mat4 light_proj_matrix =
            GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.projection_matrices[i] *
            model_matrix;
        vout.cascade_map_coords[i] = light_proj_matrix * vec4(vertex.pos, 1.0);
    }
}