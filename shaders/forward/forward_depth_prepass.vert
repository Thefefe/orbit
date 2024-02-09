#version 460

#include "../include/common.glsl"
#include "../include/types.glsl"

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    uint draw_command_buffer;
    uint cull_info_buffer;
    uint vertex_buffer;
    uint meshlet_buffer;
    uint meshlet_data_buffer;
    uint entity_buffer;
    uint materials_buffer;
};

layout(location = 0) out VertexOutput {
    vec2 uv;
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
    uint vertex_index =
        GetBuffer(MeshletDrawCommandBuffer, draw_command_buffer).draws[gl_DrawID].meshlet_vertex_offset +
        GetBuffer(MeshletDataBuffer, meshlet_data_buffer).vertex_indices[gl_VertexIndex];
    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[vertex_index];
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;

    gl_Position = view_proj * model_matrix * vec4(vertex.position[0], vertex.position[1], vertex.position[2], 1.0);

    vout.uv = vertex.uv;

    uint meshlet_index = GetBuffer(MeshletDrawCommandBuffer, draw_command_buffer).draws[gl_DrawID].meshlet_index;
    vout.material_index = GetBuffer(MeshletBuffer, meshlet_buffer).meshlets[meshlet_index].material_index;
}