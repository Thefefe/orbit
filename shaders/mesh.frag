#version 460

#include "common.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;
layout(location = 3) flat in uint in_material_index;

layout(location = 0) out vec4 out_color;

RegisterBuffer(PerFrameData, std430, readonly, {
    mat4 view_proj;
    uint render_mode;
});

struct MaterialData {
    vec4 base_color;
    uint base_texture_index;
    uint normal_texture_index;
    vec2 _padding;
};

RegisterBuffer(Materials, std430, readonly, {
    MaterialData materials[];
});

BindSlot(PerFrameData, 0);
BindSlot(Materials, 5);

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

vec3[8] colors = vec3[](
    vec3(1.0, 0.0, 0.0), // [0] Red
    vec3(0.0, 1.0, 0.0), // [1] Green
    vec3(0.0, 0.0, 1.0), // [2] Blue
    
    vec3(1.0, 1.0, 0.0), // [3] Yellow
    vec3(0.0, 1.0, 1.0), // [4] Cyan
    vec3(1.0, 0.0, 1.0), // [5] Magenta
    
    vec3(1.0, 1.0, 1.0), // [6] White
    vec3(0.1, 0.1, 0.1)  // [7] Gray
);

void main() {
    uint render_mode = GetBuffer(PerFrameData).render_mode;
    if (render_mode == 1) {
        out_color = vec4(mod(in_uv, 1.0), 0.0, 1.0);
    } else if (render_mode == 2) {
        out_color = vec4((in_normal + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
    } else if (render_mode == 3) {
        uint ihash = hash(in_material_index);
        vec3 icolor = vec3(float(ihash & 255), float((ihash >> 8) & 255), float((ihash >> 16) & 255)) / 255.0;
        // out_color = vec4(icolor, 1.0);
        // out_color = vec4(colors[in_material_index % 8], 1.0);
        out_color = vec4(vec3(float(in_material_index) / 255.0), 1.0);
    } else if (render_mode == 4) {
        out_color = vec4((in_tangent.xyz + vec3(1.0, 1.0, 1.0)) * 0.5, 1.0);
    } else {
        MaterialData material = GetBuffer(Materials).materials[in_material_index];
        vec4 base_tex_color = texture(sampler2D(GetTextureByIndex(material.base_texture_index), _uSamplers[0]), in_uv);
        out_color = base_tex_color * material.base_color;
        if (out_color.a < 0.1) {
            discard;
        }
        
        // out_color = material.base_color;
    }
}