#version 460

#include "common1.glsl"
#include "types.glsl"

RegisterBuffer(PerFrameBuffer, std430, readonly, {
    PerFrameData data;
});

RegisterBuffer(DirectionalLightBuffer, std430, readonly, {
    DirectionalLightData data;
});

RegisterBuffer(Materials, std430, readonly, {
    MaterialData materials[];
});

layout(push_constant, std430) uniform PushConstants {
    uint per_frame_buffer;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands;
    uint materials;
    uint directional_light_buffer;
};

layout(location = 0) in VertexOutput {
    vec4 world_pos;
    vec4 view_pos;
    vec2 uv;
    mat3 TBN;
    flat uint material_index;
} vout;

layout(location = 0) out vec4 out_color;

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

float compute_shadow(vec4 light_space_frag_pos, uint shadow_map) {
    vec3 proj_coords = light_space_frag_pos.xyz / light_space_frag_pos.w;
    proj_coords.y *= -1.0;
    // float closest_depth = texture(GetSampledTexture(light_shadow_map), (proj_coords.xy + 1.0 ) / 2.0).r;
    float current_depth = proj_coords.z;

    vec2 tex_coord = (proj_coords.xy + 1.0 ) / 2.0;

    if (tex_coord.x < 0.0 || tex_coord.x > 1.0 || tex_coord.y < 0.0 || tex_coord.y > 1.0) return 1.0;

    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcf_depth = texture(GetSampledTexture2D(shadow_map), tex_coord + vec2(x, y) * texel_size).r;
            shadow += current_depth  > pcf_depth ? 1.0 : 0.0;
        }    
    }
    shadow /= 9.0;

    return shadow;
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

    return (kD * albedo / PI + specular) * radiance * n_dot_l;
}

float uniform_frustum_split(uint index, float near, float far, float cascade_count){
    return near + (far - near) * (float(index) / float(cascade_count));
}

float logarithmic_frustum_split(uint index, float near, float far, float cascade_count) {
    return near * pow((far / near), float(index) / float(cascade_count));
}

float practical_frustum_split(uint index, float near, float far, float cascade_count, float lambda) {
    return logarithmic_frustum_split(index, near, far, cascade_count) * lambda +  
           uniform_frustum_split(index, near, far, cascade_count) * (1.0 - lambda);
}

const int CASCADE_COUNT = 4;
const float MAX_SHADOW_DISTANCE = 200.0;

const vec3 CASCADE_COLORS[6] = vec3[](
    vec3(1.0, 0.25, 0.25),
    vec3(0.25, 1.0, 0.25),
    vec3(0.25, 0.25, 1.0),
    vec3(1.0, 1.0, 0.25),
    vec3(0.25, 1.0, 1.0),
    vec3(1.0, 0.25, 1.0)
);

void main() {
    MaterialData material = GetBuffer(Materials, materials).materials[vout.material_index];
    uint render_mode = GetBuffer(PerFrameBuffer, per_frame_buffer).data.render_mode;

    vec4  base_color = material.base_color;
    vec3  albedo     = base_color.rgb;
    vec3  normal     = vout.TBN[2];
    float metallic   = material.metallic_factor;
    float roughness  = material.roughness_factor;
    vec3  emissive   = material.emissive_factor;
    float ao         = 1.0;

    if (material.base_texture_index != TEXTURE_NONE) {
        base_color *= texture(GetSampledTexture2D(material.base_texture_index), vout.uv);
        albedo = base_color.rgb;
    }

    if (material.normal_texture_index != TEXTURE_NONE) {
        // support for 2 component normals, for now I do this for all normal maps
        vec4 normal_tex = texture(GetSampledTexture2D(material.normal_texture_index), vout.uv);
        vec3 normal_tang = normal_tex.xyz * 2.0 - 1.0;
        normal_tang.z = sqrt(abs(1 - normal_tang.x*normal_tang.x - normal_tang.y * normal_tang.y));
        normal = vout.TBN * normal_tang;
    }
    
    if (material.metallic_roughness_texture_index != TEXTURE_NONE) {
        vec4 metallic_roughness =
            texture(GetSampledTexture2D(material.metallic_roughness_texture_index), vout.uv);
        metallic *= metallic_roughness.b;
        roughness *= metallic_roughness.g;
    }

    if (material.emissive_texture_index != TEXTURE_NONE) {
        emissive *= texture(GetSampledTexture2D(material.emissive_texture_index), vout.uv).rgb;
    }

    if (material.occulusion_texture_index != TEXTURE_NONE) {
        ao = texture(GetSampledTexture2D(material.occulusion_texture_index), vout.uv).r * material.occulusion_factor;
    }

    vec3 base_reflectivity = mix(vec3(0.04), albedo, metallic);

    vec3 N = normalize(normal);
    vec3 V = normalize(GetBuffer(PerFrameBuffer, per_frame_buffer).data.view_pos - vout.world_pos.xyz);

    out_color.a = base_color.a;
    if (out_color.a < 0.5) {
        discard;
    }

    uint cascade_index = CASCADE_COUNT;
    for(uint i = 1; i <= CASCADE_COUNT; ++i) {
        if(vout.view_pos.z < GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.cascades[CASCADE_COUNT - i].far_view_distance) {
            cascade_index -= 1;
        }
    }

    vec4 light_space_frag_pos = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.cascades[cascade_index].light_projection * vout.world_pos;
    uint shadow_map = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.cascades[cascade_index].shadow_map_index;
    // uint shadow_map = cascade_shadow_maps[cascade_index];

    float shadow = 1.0;
    if (cascade_index < CASCADE_COUNT) shadow = compute_shadow(light_space_frag_pos, shadow_map);

    switch (render_mode) {
        case 0:
            vec3 light_direction =  GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.direction;
            vec3 light_color = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.color;
            light_color *= GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.intensitiy;
            vec3 Lo = calculate_light(
                V,
                light_direction,
                light_color,
                base_reflectivity,
                albedo,
                normal,
                metallic,
                roughness
            ) * shadow;

            vec3 ambient = vec3(0.03) * albedo.rgb * ao;
            out_color.rgb = ambient + Lo + emissive;
            break;
        case 1: 
            // out_color = vec4(mod(vout.uv, 1.0), 0.0, 1.0);
            float shadow = max(shadow, 0.3);
            vec3 cascade_color = vec3(0.25);
            if (cascade_index < CASCADE_COUNT) cascade_color = CASCADE_COLORS[cascade_index];
            out_color = vec4(cascade_color * albedo * shadow, 1.0);
            break;
        case 2: 
            out_color = vec4(normal * 0.5 + 0.5, 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
            break;
        case 3: 
            out_color = vec4(vec3(metallic), 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
            break;
        case 4: 
            out_color = vec4(vec3(roughness), 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
            break;
        case 5: 
            out_color = vec4(emissive, 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
            break;
        case 6: 
            out_color = vec4(vec3(ao), 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
            break;
        case 7: 
            // uint ihash = hash(in_material_index);
            // vec3 icolor = vec3(float(ihash & 255), float((ihash >> 8) & 255), float((ihash >> 16) & 255)) / 255.0;
            out_color = vec4(vec3(float(vout.material_index) / 255.0), 1.0);
            break;
    }

    if (render_mode != 0) {
        out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
    }
}