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

const float PI = 3.14159265359;
const float EPSILON = 0.0000001; // just some small number that isn't 0

float distribution_ggx(float n_dot_h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return a2 / max(denom, EPSILON);
}

float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k); // schlick ggx
    float ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float h_dot_v, vec3 base_reflectivity) {
    return base_reflectivity + (1.0 - base_reflectivity) * pow(1.0 - h_dot_v, 5.0);
}

vec3 calculate_light(
    vec3 view_dir,

    vec3 light_dir,
    vec3 light_color,

    vec3 base_reflectivity,

    vec3  albedo,
    vec3  normal,
    float metallic,
    float roughness
) {
    vec3 N = normalize(normal);
    vec3 V = normalize(view_dir);
    vec3 L = normalize(light_dir);
    vec3 H = normalize(V + L);

    float distance = 1.0;
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = light_color * attenuation;

    // Cook-Torrance BRDF
    float n_dot_v = max(dot(N, V), EPSILON);
    float n_dot_l = max(dot(N, L), EPSILON);
    float h_dot_v = max(dot(H, V), 0.0);
    float n_dot_h = max(dot(N, H), 0.0);

    float D = distribution_ggx(n_dot_h, roughness);
    float G = geometry_smith(n_dot_v, n_dot_l, roughness);
    vec3  F = fresnel_schlick(h_dot_v, base_reflectivity);

    vec3 specular = D * G * F;

    specular /= 4.0 * n_dot_v * n_dot_l;

    vec3 kD = vec3(1.0) - F;
    kD *= 1.0 - metallic;

    return (kD * albedo / PI + specular) * radiance * n_dot_l;;
}

void main() {
    MaterialData material = GetBuffer(Materials).materials[vout.material_index];
    uint render_mode = GetBuffer(PerFrameBuffer).data.render_mode;

    vec4  base_color = material.base_color;
    vec3  albedo     = base_color.rgb;
    vec3  normal     = vout.TBN[2];
    float metallic   = material.metallic_factor;
    float roughness  = material.roughness_factor;
    vec3  emissive   = material.emissive_factor;
    float ao         = 1.0;

    if (material.base_texture_index != TEXTURE_NONE) {
        base_color *= texture(GetSampledTextureByIndex(material.base_texture_index), vout.uv);
        albedo = base_color.rgb;
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
        ao = texture(GetSampledTextureByIndex(material.occulusion_texture_index), vout.uv).r * material.occulusion_factor;
    }

    vec3 base_reflectivity = mix(vec3(0.04), albedo, metallic);

    vec3 N = normalize(normal);
    vec3 V = normalize(GetBuffer(PerFrameBuffer).data.view_pos - vout.world_pos);

    out_color.a = base_color.a;
    if (out_color.a < 0.5) {
        discard;
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
        out_color = vec4(vec3(ao), 1.0);
    } else if (render_mode == 7) { // material index
        // uint ihash = hash(in_material_index);
        // vec3 icolor = vec3(float(ihash & 255), float((ihash >> 8) & 255), float((ihash >> 16) & 255)) / 255.0;
        out_color = vec4(vec3(float(vout.material_index) / 255.0), 1.0);
    } else {
        vec3 light_direction = normalize(GetBuffer(PerFrameBuffer).data.light_direction);
        vec3 light_color = vec3(4.5);
        vec3 Lo = calculate_light(
            V,
            light_direction,
            light_color,
            base_reflectivity,
            albedo,
            normal,
            metallic,
            roughness
        );

        vec3 ambient = vec3(0.03) * albedo.rgb * ao;
        out_color.rgb = ambient + Lo + emissive;
    }

    if (render_mode != 0) {
        out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
    }
}