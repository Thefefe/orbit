#version 460

#include "include/common.glsl"
#include "include/types.glsl"
#include "include/functions.glsl"

layout(push_constant, std430) uniform PushConstants {
    PerFrameBuffer per_frame_buffer;
    VertexBuffer vertex_buffer;
    EntityBuffer entity_buffer;
    DrawCommandsBuffer draw_commands;
    MaterialsBuffer materials_buffer;
    DirectionalLightBuffer directional_light_buffer;
    uint irradiance_image_index;
    uint prefiltered_env_map_index;
    uint brdf_integration_map_index;
};

layout(location = 0) in VertexOutput {
    vec4 world_pos;
    vec4 view_pos;
    vec2 uv;
    mat3 TBN;
    flat uint material_index;
    vec4 cascade_map_coords[MAX_SHADOW_CASCADE_COUNT];
} vout;

layout(location = 0) out vec4 out_color;

float compute_shadow(vec4 light_space_frag_pos, uint shadow_map) {
    vec3 proj_coords = light_space_frag_pos.xyz / light_space_frag_pos.w;
    proj_coords.y *= -1.0;
    float current_depth = proj_coords.z;

    vec2 tex_coord = (proj_coords.xy + 1.0 ) / 2.0;

    if (tex_coord.x < 0.0 || tex_coord.x > 1.0 || tex_coord.y < 0.0 || tex_coord.y > 1.0) return 1.0;

    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0);
    const int SAMPLES = 2;
    for(int x = -SAMPLES; x <= SAMPLES; ++x)
    {
        for(int y = -SAMPLES; y <= SAMPLES; ++y)
        {
            shadow += texture(
                sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
                vec3(tex_coord + vec2(x, y) * texel_size, current_depth)
            );
        }    
    }
    shadow /= (SAMPLES * 2 + 1) * (SAMPLES * 2 + 1);

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

bool check_ndc_bounds(vec2 v) {
    return all(bvec4(lessThanEqual(vec2(-1.0), v), lessThanEqual(v, vec2(1.0))));
}

const vec3 CASCADE_COLORS[6] = vec3[](
    vec3(1.0, 0.25, 0.25),
    vec3(0.25, 1.0, 0.25),
    vec3(0.25, 0.25, 1.0),
    vec3(1.0, 1.0, 0.25),
    vec3(0.25, 1.0, 1.0),
    vec3(1.0, 0.25, 1.0)
);

void main() {
    MaterialData material = materials_buffer.materials[vout.material_index];
    uint render_mode = per_frame_buffer.data.render_mode;

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
        normal = normalize(vout.TBN * normal_tang);
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

    vec3 view_direction = normalize(per_frame_buffer.data.view_pos - vout.world_pos.xyz);

    out_color.a = base_color.a;
    if (out_color.a < 0.5) {
        discard;
    }

    uint cascade_index = 4 ;
    for (int i = 3; i >= 0; i -= 1) {
        if (check_ndc_bounds(vout.cascade_map_coords[i].xy)) {
            cascade_index = i;
        }
    }

    vec4 light_space_frag_pos = vout.cascade_map_coords[cascade_index];
    uint shadow_map = directional_light_buffer.data.shadow_maps[cascade_index];

    float shadow = 1.0;
    if (cascade_index < MAX_SHADOW_CASCADE_COUNT) shadow = compute_shadow(light_space_frag_pos, shadow_map);

    switch (render_mode) {
        case 0:
            vec3 light_direction =  directional_light_buffer.data.direction;
            vec3 light_color = directional_light_buffer.data.color;
            light_color *= directional_light_buffer.data.intensitiy;
            vec3 Lo = calculate_light(
                view_direction,
                light_direction,
                light_color,
                base_reflectivity,
                albedo,
                normal,
                metallic,
                roughness
            ) * shadow;

            vec3 R = reflect(view_direction, normal);
            R.y *= -1.0;
            vec3 F = fresnel_schlick_roughness(
                max(dot(normal, view_direction), 0.0),
                base_reflectivity,
                roughness
            );

            vec3 kS = F;
            vec3 kD = 1.0 - kS;
            kD *= 1.0 - metallic;	
            
            vec3 irradiance = texture(GetSampledTextureCube(irradiance_image_index), normal).rgb;
            vec3 diffuse    = irradiance * albedo;

            float max_reflection_lod = textureQueryLevels(GetSampledTextureCube(prefiltered_env_map_index)) - 1;
            float reflection_lod = roughness * max_reflection_lod;
            vec3 reflection_color = textureLod(GetSampledTextureCube(prefiltered_env_map_index), R, reflection_lod).rgb;
            vec2 env_brdf = texture(
                GetSampledTexture2D(brdf_integration_map_index),
                vec2(max(dot(normal, view_direction), 0.0), roughness)
            ).rg;
            vec3 specular = reflection_color * (kS * env_brdf.x + env_brdf.y);

            vec3 ambient    = (kD * diffuse + specular) * ao;

            out_color.rgb = ambient + Lo + emissive * 4.0;
            break;
        case 1: 
            float shadow = max(shadow, 0.7);
            vec3 cascade_color = vec3(0.25);
            if (cascade_index < MAX_SHADOW_CASCADE_COUNT)
                cascade_color = CASCADE_COLORS[cascade_index];
            out_color = vec4(cascade_color * albedo * shadow, 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
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
            out_color = vec4(vec3(float(vout.material_index) / 255.0), 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
            break;
    }
}