#version 460

#include "../common.glsl"
#extension GL_EXT_shader_explicit_arithmetic_types : require

struct Vertex {
    vec2     pos;
    vec2     uv;
    u8vec4   color;
};

RegisterBuffer(VertexBuffer, std430, readonly, {
    Vertex vertices[];
});

BindSlot(screen_width, 0); // float
BindSlot(screen_height, 1); // float
BindSlot(VertexBuffer, 3);

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, cutoff);
}

void main() {
    vec2 screen_size = vec2(GetFloat(screen_width),GetFloat(screen_height));
    
    Vertex vertex = GetBuffer(VertexBuffer).vertices[gl_VertexIndex];
    vec4 color = vec4(vertex.color.r, vertex.color.g, vertex.color.b, vertex.color.a) / 255.0;

    gl_Position = vec4(
        2.0 * vertex.pos.x / screen_size.x - 1.0,
        2.0 * vertex.pos.y / screen_size.y - 1.0,
        0.0, 1.0
    );

    outColor = vec4(srgb_to_linear(color.rgb), color.a);
    outUV = vertex.uv;
}