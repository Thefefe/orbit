#version 460

#include "common.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;

layout(location = 0) out vec4 out_color;

RegisterBuffer(PerFrameData, std430, readonly, {
    mat4 view_proj;
    uint render_mode;
});

BindSlot(PerFrameData, 0);

void main() {
    
    uint render_mode = GetBuffer(PerFrameData).render_mode;
    if (render_mode == 0) {
        out_color = vec4(mod(in_uv, 1.0), 0.0, 1.0);
    } else if (render_mode == 1) {
        out_color = vec4((in_normal + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
    } else {
        out_color = vec4((in_tangent.xyz + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
    }
}