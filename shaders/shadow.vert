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
    float alpha_cutoff;
    flat uint texture_index;
} vout;

void main() {
    uint material_index = GetBuffer(DrawCommandsBuffer, draw_commands_buffer)
        .commands[gl_DrawID].material_index;

    MeshVertex vertex = GetBuffer(VertexBuffer, vertex_buffer).vertices[gl_VertexIndex];

    vout.uv = vec2(vertex.uv_coord[0], vertex.uv_coord[1]);
    vout.alpha_cutoff = GetBuffer(MaterialsBuffer, materials_buffer).materials[material_index].alpha_cutoff;
    vout.texture_index = GetBuffer(MaterialsBuffer, materials_buffer).materials[material_index].base_texture_index;
    
    vec3 pos = vec3(vertex.position[0], vertex.position[1], vertex.position[2]);
    mat4 model_matrix = GetBuffer(EntityBuffer, entity_buffer).entities[gl_InstanceIndex].model_matrix;
    
    gl_Position = view_proj * model_matrix * vec4(pos, 1.0);
}