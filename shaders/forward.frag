#version 460

#include "include/common.glsl"
#include "include/types.glsl"
#include "include/functions.glsl"

layout(push_constant, std430) uniform PushConstants {
    uint per_frame_buffer;
    uint vertex_buffer;
    uint entity_buffer;
    uint draw_commands;
    uint materials_buffer;
    uint directional_light_buffer;
    uint light_count;
    uint light_data_buffer;
    uint irradiance_image_index;
    uint prefiltered_env_map_index;
    uint brdf_integration_map_index;
    uint jitter_texture_index;
};

layout(location = 0) in VertexOutput {
    vec4 world_pos;
    vec2 uv;
    vec3 normal;
    vec4 tangent;
    flat uint material_index;
} vout;

layout(location = 0) out vec4 out_color;

const vec2 poisson_offsets[64] = vec2[](
    vec2(0.0617981, 0.07294159),
	vec2(0.6470215, 0.7474022),
	vec2(-0.5987766, -0.7512833),
	vec2(-0.693034, 0.6913887),
	vec2(0.6987045, -0.6843052),
	vec2(-0.9402866, 0.04474335),
	vec2(0.8934509, 0.07369385),
	vec2(0.1592735, -0.9686295),
	vec2(-0.05664673, 0.995282),
	vec2(-0.1203411, -0.1301079),
	vec2(0.1741608, -0.1682285),
	vec2(-0.09369049, 0.3196758),
	vec2(0.185363, 0.3213367),
	vec2(-0.1493771, -0.3147511),
	vec2(0.4452095, 0.2580113),
	vec2(-0.1080467, -0.5329178),
	vec2(0.1604507, 0.5460774),
	vec2(-0.4037193, -0.2611179),
	vec2(0.5947998, -0.2146744),
	vec2(0.3276062, 0.9244621),
	vec2(-0.6518704, -0.2503952),
	vec2(-0.3580975, 0.2806469),
	vec2(0.8587891, 0.4838005),
	vec2(-0.1596546, -0.8791054),
	vec2(-0.3096867, 0.5588146),
	vec2(-0.5128918, 0.1448544),
	vec2(0.8581337, -0.424046),
	vec2(0.1562584, -0.5610626),
	vec2(-0.7647934, 0.2709858),
	vec2(-0.3090832, 0.9020988),
	vec2(0.3935608, 0.4609676),
	vec2(0.3929337, -0.5010948),
	vec2(-0.8682281, -0.1990303),
	vec2(-0.01973724, 0.6478714),
	vec2(-0.3897587, -0.4665619),
	vec2(-0.7416366, -0.4377831),
	vec2(-0.5523247, 0.4272514),
	vec2(-0.5325066, 0.8410385),
	vec2(0.3085465, -0.7842533),
	vec2(0.8400612, -0.200119),
	vec2(0.6632416, 0.3067062),
	vec2(-0.4462856, -0.04265022),
	vec2(0.06892014, 0.812484),
	vec2(0.5149567, -0.7502338),
	vec2(0.6464897, -0.4666451),
	vec2(-0.159861, 0.1038342),
	vec2(0.6455986, 0.04419327),
	vec2(-0.7445076, 0.5035095),
	vec2(0.9430245, 0.3139912),
	vec2(0.0349884, -0.7968109),
	vec2(-0.9517487, 0.2963554),
	vec2(-0.7304786, -0.01006928),
	vec2(-0.5862702, -0.5531025),
	vec2(0.3029106, 0.09497032),
	vec2(0.09025345, -0.3503742),
	vec2(0.4356628, -0.0710125),
	vec2(0.4112572, 0.7500054),
	vec2(0.3401214, -0.3047142),
	vec2(-0.2192158, -0.6911137),
	vec2(-0.4676369, 0.6570358),
	vec2(0.6295372, 0.5629555),
	vec2(0.1253822, 0.9892166),
	vec2(-0.1154335, 0.8248222),
	vec2(-0.4230408, -0.7129914)
);

// https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/
float interleaved_gradient_noise(vec2 seed) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(seed, magic.xy)));
}

float avg_blockers_depth_to_penumbra(float z_shadow_map_view, float avg_blockers_depth) {
    float penumbra = (z_shadow_map_view - avg_blockers_depth) / avg_blockers_depth;
    penumbra *= penumbra;
    return clamp(80.0f * penumbra, 0.0, 1.0);
}

#define PENUMBRA_SAMPLE_COUNT 8
#define SHADOW_SAMPLE_COUNT 32

vec2 vogel_disk_sample(int sampleIndex, int samplesCount, float phi) {
    float GoldenAngle = 2.4;

    float r = sqrt(float(sampleIndex) + 0.5) / sqrt(float(samplesCount));
    float theta = sampleIndex * GoldenAngle + phi;

    return vec2(r * cos(theta), r * sin(theta));
}

void penumbra_vogel(uint shadow_map, float vogel_theta, vec3 light_space_pos, out uint blockers_count, out float avg_blockers_depth) {
    avg_blockers_depth = 0.0f;
    blockers_count = 0;

    vec2 texel_size = 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0);
    float penumbra_filter_max_size = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.penumbra_filter_max_size;

    for(int i = 0; i < PENUMBRA_SAMPLE_COUNT; i ++) {
        vec2 sample_uv = vogel_disk_sample(i, PENUMBRA_SAMPLE_COUNT, vogel_theta);
        sample_uv = light_space_pos.xy + sample_uv * penumbra_filter_max_size * texel_size;

        float sample_depth = texture(
            sampler2D(GetTexture2D(shadow_map), GetSampler(SHADOW_DEPTH_SAMPLER)),
            sample_uv
        ).x;

        if(sample_depth > light_space_pos.z) {
            avg_blockers_depth += 1.0 - sample_depth;
            blockers_count += 1;
        }
    }

    avg_blockers_depth /= float(blockers_count);
}

float pcf_vogel(uint shadow_map, vec4 clip_pos) {
    clip_pos.xyz /= clip_pos.w;
    clip_pos.y *= -1.0;
    clip_pos.xy = (clip_pos.xy + 1.0) * 0.5;

    float sum = 0.0;
    vec2 texel_size = 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0);
    float random_theta = interleaved_gradient_noise(gl_FragCoord.xy) * 2 * PI;

    uint blockers_count;
    float avg_blockers_depth;
    penumbra_vogel(shadow_map, random_theta, clip_pos.xyz, blockers_count, avg_blockers_depth);
    
    if (blockers_count == 0 && blockers_count == PENUMBRA_SAMPLE_COUNT) return blockers_count / PENUMBRA_SAMPLE_COUNT;

    float penumbra_scale = avg_blockers_depth_to_penumbra(1.0 - clip_pos.z, avg_blockers_depth);

    float max_filter_radius = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.max_filter_radius;
    float min_filter_radius = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.min_filter_radius;
    
    float filter_radius = mix(min_filter_radius, max_filter_radius, penumbra_scale);

    for (int i = 0; i < SHADOW_SAMPLE_COUNT; i += 1) {
        vec2 offset = vogel_disk_sample(i, SHADOW_SAMPLE_COUNT, random_theta) * filter_radius * texel_size;
        vec2 sample_pos = clip_pos.xy + offset;

        vec4 gathered_samples = textureGather(
            sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
            sample_pos,
            clip_pos.z
        );

        sum += dot(gathered_samples, vec4(1.0));
    }

    sum /= SHADOW_SAMPLE_COUNT * 4;

    return sum;
}

void penumbra_poisson(uint shadow_map, float random_theta, vec3 light_space_pos, out uint blockers_count, out float avg_blockers_depth) {
    avg_blockers_depth = 0.0f;
    blockers_count = 0;

    vec2 texel_size = 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0);
    float penumbra_filter_max_size = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.penumbra_filter_max_size;

    float s = sin(random_theta);
    float c = cos(random_theta);
    mat2 rot = mat2(c, s, -s, c);

    for(int i = 0; i < PENUMBRA_SAMPLE_COUNT; i ++) {
        vec2 sample_uv = rot * poisson_offsets[i];
        sample_uv = light_space_pos.xy + sample_uv * penumbra_filter_max_size * texel_size;

        float sample_depth = texture(
            sampler2D(GetTexture2D(shadow_map), GetSampler(SHADOW_DEPTH_SAMPLER)),
            sample_uv
        ).x;

        if(sample_depth > light_space_pos.z) {
            avg_blockers_depth += 1.0 - sample_depth;
            blockers_count += 1;
        }
    }

    avg_blockers_depth /= float(blockers_count);
}

float pcf_poisson(uint shadow_map, vec4 clip_pos) {
    clip_pos.xyz /= clip_pos.w;
    clip_pos.y *= -1.0;
    clip_pos.xy = (clip_pos.xy + 1.0) * 0.5;

    float sum = 0.0;
    vec2 texel_size = 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0);
    float random_theta = interleaved_gradient_noise(gl_FragCoord.xy) * 2 * PI;
    float s = sin(random_theta);
    float c = cos(random_theta);
    mat2 rot = mat2(c, s, -s, c);

    uint blockers_count;
    float avg_blockers_depth;
    penumbra_poisson(shadow_map, random_theta, clip_pos.xyz, blockers_count, avg_blockers_depth);
    
    if (blockers_count == 0 && blockers_count == PENUMBRA_SAMPLE_COUNT) return blockers_count / PENUMBRA_SAMPLE_COUNT;

    float penumbra_scale = avg_blockers_depth_to_penumbra(1.0 - clip_pos.z, avg_blockers_depth);

    float max_filter_radius = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.max_filter_radius;
    float min_filter_radius = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.min_filter_radius;
    
    float filter_radius = mix(min_filter_radius, max_filter_radius, penumbra_scale);

    for (int i = 0; i < SHADOW_SAMPLE_COUNT; i += 1) {
        vec2 offset = rot * poisson_offsets[i];
        vec2 sample_pos = clip_pos.xy + offset * filter_radius * texel_size;

        vec4 gathered_samples = textureGather(
            sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
            sample_pos,
            clip_pos.z
        );

        sum += dot(gathered_samples, vec4(1.0));
    }

    sum /= SHADOW_SAMPLE_COUNT * 4;

    return sum;
}

float pcf_branch(uint shadow_map, vec4 clip_pos) {
    clip_pos.xyz /= clip_pos.w;
    clip_pos.y *= -1.0;
    clip_pos.xy = (clip_pos.xy + 1.0) * 0.5;

    ivec3 offset_coord;
    vec3 jitter_size = textureSize(GetSampledTexture3D(jitter_texture_index), 0);
    offset_coord.yz = ivec2(mod(gl_FragCoord.xy, jitter_size.yz));

    float sum = 0.0;

    vec2 texel_size = 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0);

    float max_filter_radius = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.max_filter_radius;
    float min_filter_radius = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.min_filter_radius;
    
    float filter_radius = min_filter_radius;

    for (int i = 0; i < 4; i++) {
        offset_coord.x = i;
        vec4 offsets = texelFetch(GetSampledTexture3D(jitter_texture_index), offset_coord, 0) * filter_radius;
        
        sum += texture(
            sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
            vec3(clip_pos.xy + offsets.xy * texel_size, clip_pos.z)
        );

        sum += texture(
            sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
            vec3(clip_pos.xy + offsets.zw * texel_size, clip_pos.z)
        );
    }

    if (sum == 0.0 || sum == 8.0) {
        float shadow = sum / 8.0;
        float center = texture(
            sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
            vec3(clip_pos.xy, clip_pos.z)
        );
        
        if (shadow == center) return shadow;
    }

    for (int i = 4; i < 32; i++) {
        offset_coord.x = i;
        vec4 offsets = texelFetch(GetSampledTexture3D(jitter_texture_index), offset_coord, 0) * filter_radius;
        
        sum += texture(
            sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
            vec3(clip_pos.xy + offsets.xy * texel_size,clip_pos.z)
        );

        sum += texture(
            sampler2DShadow(GetTexture2D(shadow_map), GetCompSampler(SHADOW_SAMPLER)),
            vec3(clip_pos.xy + offsets.zw * texel_size,clip_pos.z)
        );
    }

    return sum / 64.0;
}

vec3 calculate_light(
    vec3 view_dir,

    vec3 light_dir,
    vec3 light_color,
    float light_distance,

    vec3  albedo,
    vec3  normal,
    float metallic,
    float roughness
) {
    vec3 H = normalize(view_dir + light_dir);

    float attenuation = 1.0 / max(light_distance * light_distance, EPSILON);
    vec3 radiance = light_color * attenuation;

    float n_dot_v = max(dot(normal, view_dir), EPSILON);
    float n_dot_l = max(dot(normal, light_dir), EPSILON);

    float D = distribution_ggx(max(dot(normal, H), 0.0), roughness);
    float G = geometry_smith(n_dot_v, n_dot_l, roughness);
    vec3  F = fresnel_schlick(max(dot(H, view_dir), 0.0), mix(vec3(0.04), albedo, metallic));

    vec3 specular = D * G * F;
    specular /= 4.0 * n_dot_v * n_dot_l;

    vec3 kD = vec3(1.0) - F;
    kD *= 1.0 - metallic;

    return (kD * albedo / PI + specular) * radiance * n_dot_l;
}

bool check_ndc_bounds(vec4 p) {
    p /= p.w;
    return all(greaterThanEqual(p, vec4(-1.0, -1.0, 0.0, 0.0))) && all(lessThanEqual(p, vec4(1.0)));
}

uint select_cascade_by_interval(vec4 distances, float d) {
    vec4 cmp = mix(vec4(0.0), vec4(1.0), lessThanEqual(vec4(d), distances));
    return 4 - uint(dot(cmp, vec4(1.0)));
}

const vec3 CASCADE_COLORS[6] = vec3[](
    vec3(1.0, 0.25, 0.25),
    vec3(0.25, 1.0, 0.25),
    vec3(0.25, 0.25, 1.0),
    vec3(1.0, 1.0, 0.25),
    vec3(0.25, 1.0, 1.0),
    vec3(1.0, 0.25, 1.0)
);

#define MIP_SCALE 0.25

void main() {
    uint render_mode = GetBuffer(PerFrameBuffer, per_frame_buffer).data.render_mode;

    out_color = vec4(1.0);

    vec4  base_color;
    vec3  normal;
    float metallic;
    float roughness;
    vec3  emissive;
    float ao;
    float alpha_cutoff;
    {
        MaterialData material = GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index];
        base_color = material.base_color;
        normal     = vout.normal;
        metallic   = material.metallic_factor;
        roughness  = material.roughness_factor;
        emissive   = material.emissive_factor;
        ao         = 1.0;
        alpha_cutoff = material.alpha_cutoff;

        if (material.base_texture_index != TEXTURE_NONE) {
            base_color *= texture(GetSampledTexture2D(material.base_texture_index), vout.uv);

            vec2 lods = textureQueryLod(GetSampledTexture2D(material.base_texture_index), vout.uv);
            float alpha_mip_level = max(lods.x, lods.y);
            
            out_color.a = (base_color.a - alpha_cutoff) / max(fwidth(base_color.a), EPSILON) + 0.5;
            out_color.a *= 1 + alpha_mip_level * MIP_SCALE;
        }

        if (material.normal_texture_index != TEXTURE_NONE) {
            mat3 TBN = mat3(
                vout.tangent.xyz,
                cross(vout.normal, vout.tangent.xyz) * sign(vout.tangent.w),
                vout.normal
            );

            vec4 normal_tex = texture(GetSampledTexture2D(material.normal_texture_index), vout.uv);
            vec3 normal_tang = normal_tex.xyz * 2.0 - 1.0;
            normal_tang.z = sqrt(abs(1 - normal_tang.x*normal_tang.x - normal_tang.y * normal_tang.y));
            normal_tang = normalize(normal_tang);

            normal = normalize(TBN * normal_tang);
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

        if (material.occlusion_texture_index != TEXTURE_NONE) {
            ao = texture(GetSampledTexture2D(material.occlusion_texture_index), vout.uv).r * material.occlusion_factor;
        }
    }

    uint cascade_index = 4;
    vec4 cascade_map_coord = vec4(0.0);

    for (uint i = 0; i < MAX_SHADOW_CASCADE_COUNT; i++) {
        vec4 map_coords = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.projection_matrices[i] * vout.world_pos;
        if (check_ndc_bounds(map_coords)) {
            cascade_index = i;
            cascade_map_coord = map_coords;
            break;
        }
    }
    
    float shadow = 1.0;
    if (cascade_index < MAX_SHADOW_CASCADE_COUNT) {
        uint shadow_map = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.shadow_maps[cascade_index];
        // shadow = pcf_vogel(shadow_map, cascade_map_coord);
        shadow = pcf_poisson(shadow_map, cascade_map_coord);
        // shadow = pcf_branch(shadow_map, cascade_map_coord);
    }

    switch (render_mode) {
        case 0:
            vec3 view_direction = normalize(GetBuffer(PerFrameBuffer, per_frame_buffer).data.view_pos - vout.world_pos.xyz);
            
            vec3 Lo = vec3(0.0);
            {
                vec3 light_direction = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.direction;
                vec3 light_color = GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.color;
                light_color *= GetBuffer(DirectionalLightBuffer, directional_light_buffer).data.intensitiy;

                Lo = calculate_light(
                    view_direction,
                    light_direction,
                    light_color,
                    1.0, // light_distance
                    base_color.rgb,
                    normal,
                    metallic,
                    roughness
                ) * shadow;
            }


            for (int i = 0; i < light_count; i++) {
                LightData light = GetBuffer(LightBuffer, light_data_buffer).lights[i];
                vec3 light_direction = light.position - vout.world_pos.xyz;
                vec3 light_color = light.color * light.intensity;

                float light_distance = length(light_direction);
                light_direction /= light_distance;

                Lo += calculate_light(
                    view_direction,
                    light_direction,
                    light_color,
                    light_distance,
                    base_color.rgb,
                    normal,
                    metallic,
                    roughness
                );
            }

            vec3 R = reflect(view_direction, normal);
            R.y *= -1.0;
            vec3 F_roughness = fresnel_schlick_roughness(
                max(dot(normal, view_direction), 0.0),
                mix(vec3(0.04), base_color.rgb, metallic),
                roughness
            );

            vec3 kS = F_roughness;
            vec3 kD = 1.0 - kS;
            kD *= 1.0 - metallic;	
            
            vec3 ambient = vec3(0.0);
            if (prefiltered_env_map_index != TEXTURE_NONE && irradiance_image_index != TEXTURE_NONE) {
                vec3 irradiance = texture(GetSampledTextureCube(irradiance_image_index), normal).rgb;
                vec3 diffuse    = irradiance * base_color.rgb;

                float max_reflection_lod = textureQueryLevels(GetSampledTextureCube(prefiltered_env_map_index)) - 1;
                float reflection_lod = roughness * max_reflection_lod;
                vec3 reflection_color = 
                    textureLod(GetSampledTextureCube(prefiltered_env_map_index), R, reflection_lod).rgb;
                vec2 env_brdf = texture(
                    GetSampledTexture2D(brdf_integration_map_index),
                    vec2(max(dot(normal, view_direction), 0.0), roughness)
                ).rg;
                vec3 specular = reflection_color * (kS * env_brdf.x + env_brdf.y);

                ambient = (kD * diffuse + specular) * ao;
            }

            out_color.rgb = ambient + Lo + emissive;
            break;
        case 1: {
            float shadow = max(shadow, 0.1);
            vec3 cascade_color = vec3(0.25);

            if (cascade_index < MAX_SHADOW_CASCADE_COUNT)
                cascade_color = CASCADE_COLORS[cascade_index];
            
            out_color = vec4(cascade_color * base_color.rgb * shadow, 1.0);
        } break;
        case 2:
            out_color = vec4(normal * 0.5 + 0.5, 1.0);
            out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);

            out_color = vec4(vec3(shadow), 1.0);
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