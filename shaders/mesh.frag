#version 460

#include "common.glsl"
#include "types.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;
layout(location = 3) flat in uint in_material_index;

layout(location = 0) out vec4 out_color;

RegisterBuffer(PerFrameData, std430, readonly, {
    mat4 view_proj;
    uint render_mode;
});

RegisterBuffer(Materials, std430, readonly, {
    MaterialData materials[];
});

BindSlot(PerFrameData, 0);
BindSlot(Materials, 4);

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, cutoff);
}

void main() {
    MaterialData material = GetBuffer(Materials).materials[in_material_index];
    uint render_mode = GetBuffer(PerFrameData).render_mode;    

    if (render_mode == 1) {
        out_color = vec4(mod(in_uv, 1.0), 0.0, 1.0);
    } else if (render_mode == 2) {
        out_color = vec4((in_normal + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
    } else if (render_mode == 3) {
        vec4 default_normal = vec4(0.5, 0.5, 1.0, 0.0);
        vec4 normal = sample_texture_index_default(material.normal_texture_index, in_uv, default_normal);
        out_color = vec4(normal.rgb, 1.0);
    } else if (render_mode == 4) {
        // uint ihash = hash(in_material_index);
        // vec3 icolor = vec3(float(ihash & 255), float((ihash >> 8) & 255), float((ihash >> 16) & 255)) / 255.0;
        out_color = vec4(vec3(float(in_material_index) / 255.0), 1.0);
    } else {
        vec4 base_tex_color = sample_texture_index_default(material.base_texture_index, in_uv, vec4(1.0));
        out_color = base_tex_color * material.base_color;
        
        if (out_color.a < 0.5) {
            discard;
        }
    }

    if (render_mode != 0) {
        out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
    }
}