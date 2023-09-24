#version 460

#include "../include/common.glsl"
#include "../include/types.glsl"

layout(push_constant, std430) uniform PushConstants {
    vec2 screen_size;
    uint input_texture;
    uint vertex_buffer;
    uint vertex_offset;
};

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, cutoff);
}

void main() {
    EguiVertex vertex = GetBuffer(EguiVertexBuffer, vertex_buffer).vertices[gl_VertexIndex + vertex_offset];
    vec2 normal_pos = vec2(vertex.position[0], vertex.position[1]) / screen_size;
    vec2 ndc_pos = 2.0 * normal_pos - 1.0;

    gl_Position = vec4(ndc_pos.x, -ndc_pos.y, 0.0, 1.0);

    vec4 color = vec4(vertex.color) / 255.0;

    outColor = vec4(srgb_to_linear(color.rgb), color.a);
    outUV = vec2(vertex.uv_coord[0], vertex.uv_coord[1]);
}