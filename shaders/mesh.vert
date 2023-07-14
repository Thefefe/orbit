#version 460

#include "include/common.glsl"
#include "include/types.glsl"

layout(push_constant, std430) uniform PushConstants {
    PerFrameBuffer per_frame_buffer;
    VertexBuffer vertex_buffer;
    EntityBuffer entity_buffer;
    DrawCommandsBuffer draw_commands;
    MaterialsBuffer materials_buffer;
    DirectionalLightBuffer directional_light_buffer;
    uint irradiance_image_index;
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
    MeshVertex vertex = vertex_buffer.vertices[gl_VertexIndex];
    mat4 model_matrix = entity_buffer.entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = model_matrix * vec4(vertex.pos, 1.0);

    gl_Position = per_frame_buffer.data.view_projection * vout.world_pos;

    vout.uv = vertex.uv;
    vout.view_pos = per_frame_buffer.data.view * -vout.world_pos;

    vec3 normal = vertex.norm;
    vec3 tangent = vertex.tang.xyz;
    vec3 bitangent = cross(normal, tangent) * vertex.tang.w;
    
    mat3 normal_matrix = mat3(entity_buffer.entities[gl_InstanceIndex].normal_matrix);

    vout.TBN = mat3(
        normalize(vec3(normal_matrix * tangent)),
        normalize(vec3(normal_matrix * bitangent)),
        normalize(vec3(normal_matrix * normal))
    );

    vout.material_index = draw_commands.commands[gl_DrawID].material_index;

    for (uint i = 0; i < MAX_SHADOW_CASCADE_COUNT; ++i) {
        vout.cascade_map_coords[i] = directional_light_buffer.data.projection_matrices[i] * vec4(vout.world_pos.xyz, 1.0);
    }
}