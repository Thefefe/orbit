layout(push_constant, std430) uniform PushConstants {
    uint per_frame_buffer;
    uint vertex_buffer;
    uint meshlet_buffer;
    uint meshlet_data_buffer;
    uint entity_buffer;
    uint draw_commands_buffer;
    uint materials_buffer;
    uint cluster_buffer;
    uint light_count;
    uint light_data_buffer;
    uint shadow_data_buffer;
    uint shadow_settings_buffer;
    uint selected_light;
    uint brdf_integration_map_index;
    uint jitter_texture_index;
};

#define VERTEX_OUTPUT VertexOutput { \
    vec4 world_pos;                  \
    vec2 uv;                         \
    vec3 normal;                     \
    vec4 tangent;                    \
    flat uint material_index;        \
}