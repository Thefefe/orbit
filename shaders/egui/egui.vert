#version 460

#include "../include/common.glsl"

layout(push_constant, std430) uniform PushConstants {
    vec2 screen_size;
    uint input_texture;
};

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec2 vUv;
layout(location = 2) in vec4 vColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, cutoff);
}

void main() {
    vec2 normal_pos = vPos / screen_size;
    vec2 ndc_pos = 2.0 * normal_pos - 1.0;

    gl_Position = vec4(ndc_pos.x, -ndc_pos.y, 0.0, 1.0);

    outColor = vec4(srgb_to_linear(vColor.rgb), vColor.a);
    outUV = vUv;
}