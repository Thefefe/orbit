#version 460

#include "common.glsl"
#include "types.glsl"

RegisterBuffer(PerFrameBuffer, std430, readonly, {
    PerFrameData data;
});

RegisterBuffer(Materials, std430, readonly, {
    MaterialData materials[];
});

BindSlot(PerFrameBuffer, 0);
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

layout(location = 0) in VertexOutput {
    vec3 world_pos;
    vec2 uv;
    mat3 TBN;
    flat uint material_index;
} vout;

layout(location = 0) out vec4 out_color;

void main() {
    MaterialData material = GetBuffer(Materials).materials[vout.material_index];
    uint render_mode = GetBuffer(PerFrameBuffer).data.render_mode;    

    vec4  albedo    = material.base_color;
    vec3  normal    = vout.TBN[2];
    float metallic  = material.metallic_factor;
    float roughness = material.roughness_factor;
    vec3  emissive  = material.emissive_factor;
    float ao        = 0.0;

    if (material.base_texture_index != TEXTURE_NONE) {
        albedo *= texture(GetSampledTextureByIndex(material.base_texture_index), vout.uv);
    }

    if (material.normal_texture_index != TEXTURE_NONE) {
        vec4 normal_tex = texture(GetSampledTextureByIndex(material.normal_texture_index), vout.uv);
        vec3 tang_normal = normalize(normal_tex.xyz * 2.0 - 1.0);
        normal = vout.TBN * tang_normal;
    }
    
    if (material.metallic_roughness_texture_index != TEXTURE_NONE) {
        vec4 metallic_roughness =
            texture(GetSampledTextureByIndex(material.metallic_roughness_texture_index), vout.uv);
        metallic *= metallic_roughness.b;
        roughness *= metallic_roughness.g;
    }

    if (material.emissive_texture_index != TEXTURE_NONE) {
        emissive *= texture(GetSampledTextureByIndex(material.emissive_texture_index), vout.uv).rgb;
    }

    if (material.occulusion_texture_index != TEXTURE_NONE) {
        ao = texture(GetSampledTextureByIndex(material.occulusion_texture_index), vout.uv).r;
    }

    if (render_mode == 1) { // uv
        out_color = vec4(mod(vout.uv, 1.0), 0.0, 1.0);
    } else if (render_mode == 2) { // normal
        out_color = vec4(normal * 0.5 + 0.5, 1.0);
    } else if (render_mode == 3) { // metallic
        out_color = vec4(vec3(metallic), 1.0);
    } else if (render_mode == 4) { // roughness
        out_color = vec4(vec3(roughness), 1.0);
    } else if (render_mode == 5) { // emissive
        out_color = vec4(emissive, 1.0);
    } else if (render_mode == 6) { // occulusion
        out_color = vec4(vec3(1.0 - ao), 1.0);
    } else if (render_mode == 7) { // material index
        // uint ihash = hash(in_material_index);
        // vec3 icolor = vec3(float(ihash & 255), float((ihash >> 8) & 255), float((ihash >> 16) & 255)) / 255.0;
        out_color = vec4(vec3(float(vout.material_index) / 255.0), 1.0);
    } else {
        vec4 base_tex_color = sample_texture_index_default(material.base_texture_index, vout.uv, vec4(1.0));
        out_color = base_tex_color * material.base_color;
        
        if (out_color.a < 0.5) {
            discard;
        }
    }

    if (render_mode != 0) {
        out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
    }
}