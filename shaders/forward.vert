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

// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
vec3 reference_orthonormal_vector(vec3 v) {
    float signum = v.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (signum + v.z);
    float b = v.x * v.y * a;
    return vec3(b, signum + v.y * v.y * a, -v.y);
}

vec3 octahedron_decode(vec2 f) {
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z , 0.0);
    n.xy += mix(vec2(t), vec2(-t), greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}

void unpack_normal_tangent(i8vec4 packed, out vec3 n, out vec4 t) {
    vec4 unpacked = vec4(packed) / 127.0;
    
    n = octahedron_decode(unpacked.xy);

    vec3 reference_tangent = reference_orthonormal_vector(n);
    float tangent_alpha = unpacked.z * PI;
    
    vec3 tangent = reference_tangent * cos(tangent_alpha) + cross(reference_tangent, n) * sin(tangent_alpha);
    t = vec4(normalize(tangent), unpacked.w);
}

void main() {
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    vout.world_pos = model_matrix * vec4(vertex.pos, 1.0);

    gl_Position = GetBuffer(PerFrameBuffer, per_frame_buffer).data.view_projection * vout.world_pos;

    vout.uv = vertex.uv;
    
    mat3 normal_matrix = mat3(GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].normal_matrix);

    unpack_normal_tangent(vertex.packed_normals, vout.normal, vout.tangent);
    vout.normal  = normalize(normal_matrix * vout.normal);
    vout.tangent = vec4(normalize(normal_matrix * vout.tangent.xyz), vout.tangent.w);

    vout.material_index = GetBuffer(DrawCommandsBuffer, draw_commands).commands[gl_DrawID].material_index;
}