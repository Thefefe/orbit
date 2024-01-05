#version 460

#include "../include/common.glsl"
#include "../include/types.glsl"
#include "../include/functions.glsl"
#include "../light_cluster/cluster_common.glsl"

#include "forward_common.glsl"

RegisterUintImageFormat(rg32ui);

layout(location = 0) in VERTEX_OUTPUT vout;

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

float penumbra_size(float reciever_depth, float avg_blockers_depth) {
    return (reciever_depth - avg_blockers_depth) / avg_blockers_depth;
}

#define PENUMBRA_SAMPLE_COUNT 16
#define SHADOW_SAMPLE_COUNT 16

vec2 vogel_disk_sample(int sampleIndex, int samplesCount, float phi) {
    float GoldenAngle = 2.4;

    float r = sqrt(float(sampleIndex) + 0.5) / sqrt(float(samplesCount));
    float theta = sampleIndex * GoldenAngle + phi;

    return vec2(r * cos(theta), r * sin(theta));
}

void penumbra_poisson(
    uint shadow_map,
    float random_theta,
    vec3 light_space_pos,
    out uint blockers_count,
    out float avg_blockers_depth,
    float inv_world_size
) {
    avg_blockers_depth = 0.0f;
    blockers_count = 0;

    float blocker_search_radius =
        GetBuffer(ShadowSettingsBuffer, shadow_settings_buffer).data.blocker_search_radius *
        inv_world_size;

    float s = sin(random_theta);
    float c = cos(random_theta);
    mat2 rot = mat2(c, s, -s, c);

    for(int i = 0; i < PENUMBRA_SAMPLE_COUNT; i ++) {
        vec2 sample_uv = rot * poisson_offsets[i];
        sample_uv = light_space_pos.xy + sample_uv * blocker_search_radius * inv_world_size;

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

float pcf_poisson(uint shadow_map, vec4 clip_pos, float inv_world_size, float light_size_uv) {
    clip_pos.xyz /= clip_pos.w;
    clip_pos.y *= -1.0;
    clip_pos.xy = (clip_pos.xy + 1.0) * 0.5;

    float sum = 0.0;
    float random_theta = interleaved_gradient_noise(gl_FragCoord.xy) * 2 * PI;
    float s = sin(random_theta);
    float c = cos(random_theta);
    mat2 rot = mat2(c, s, -s, c);

    // float light_size_uv = GetBuffer(ShadowSettingsBuffer, shadow_settings_buffer).data.light_size * world_size_inv;

    uint blockers_count;
    float avg_blockers_depth;
    penumbra_poisson(shadow_map, random_theta, clip_pos.xyz, blockers_count, avg_blockers_depth, inv_world_size);
    
    if (blockers_count == 0 && blockers_count == PENUMBRA_SAMPLE_COUNT) return blockers_count / PENUMBRA_SAMPLE_COUNT;

    float penumbra_scale = penumbra_size(1.0 - clip_pos.z, avg_blockers_depth) * light_size_uv;
    float filter_radius = max(penumbra_scale * inv_world_size, 1.0 / textureSize(GetSampledTexture2D(shadow_map), 0).x);

    for (int i = 0; i < SHADOW_SAMPLE_COUNT; i += 1) {
        vec2 offset = rot * poisson_offsets[i];
        vec2 sample_pos = clip_pos.xy + offset * filter_radius;

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

vec3 shadow_normal_offset(float n_dot_l, vec3 normal, uint shadow_map, float scale) {
    float texel_size = 1.0f / textureSize(GetSampledTexture2D(shadow_map), 0).x;
    float normal_offset_scale = clamp(1.0f - n_dot_l, 0.0, 1.0);
    return texel_size * scale * normal_offset_scale * normal;
}

// http://www.jp.square-enix.com/tech/library/pdf/2023_FFXVIShadowTechPaper.pdf
float get_oriented_bias(vec3 face_normal, vec3 light_direction, float oriented_bias, bool is_sss) {
    bool is_facing_light = dot(face_normal, light_direction) > 0.0;
    bool move_toward_light = is_sss || is_facing_light ;
    return move_toward_light ? - oriented_bias : oriented_bias;
}

vec3 calculate_light(
    vec3 view_dir,

    vec3 light_dir,
    vec3 light_color,
    float attenuation,

    vec3  albedo,
    vec3  normal,
    float metallic,
    float roughness
) {
    vec3 H = normalize(view_dir + light_dir);

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

#define DEBUG_COLOR_COUNT 8
const vec3 DEBUG_COLORS[DEBUG_COLOR_COUNT] = vec3[](
    vec3(1.0, 0.25, 0.25),
    vec3(0.25, 1.0, 0.25),
    vec3(0.25, 0.25, 1.0),
    vec3(1.0, 1.0, 0.25),
    vec3(0.25, 1.0, 1.0),
    vec3(1.0, 0.25, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(0.25, 0.25, 0.25)
);

// used for debugging
float debug_light(vec3 normal, vec3 light_dir, float shadow) {
    float ambient = 0.3;
    float diffuse = max(dot(normal, light_dir), 0.0) * shadow;
    return ambient + diffuse;
}

#define MIP_SCALE 0.25

float calc_mip_level(vec2 texture_coord) {
    vec2 dx = dFdx(texture_coord);
    vec2 dy = dFdy(texture_coord);
    float delta_max_sqr = max(dot(dx, dx), dot(dy, dy));
    
    return max(0.0, 0.5 * log2(delta_max_sqr));
}

void main() {
    if (GetBuffer(PerFrameBuffer, per_frame_buffer).render_mode == 9) {
        // reusing material_index as meshlet_index
        uint hash = hash(vout.material_index);
        vec3 mcolor = vec3(float(hash & 255), float((hash >> 8) & 255), float((hash >> 16) & 255)) / 255.0;
        out_color = vec4(mcolor, 1.0);
        
        out_color = vec4(srgb_to_linear(out_color.rgb), out_color.a);
        return;
    }

    out_color = vec4(1.0);
    uint alpha_mode;
    vec4  base_color;
    vec3  normal;
    float metallic;
    float roughness;
    vec3  emissive;
    float ao;
    float alpha_cutoff;
    {
        MaterialData material = GetBuffer(MaterialsBuffer, materials_buffer).materials[vout.material_index];
        alpha_mode = material.alpha_mode;
        base_color = material.base_color;
        normal     = vout.normal;
        metallic   = material.metallic_factor;
        roughness  = material.roughness_factor;
        emissive   = material.emissive_factor;
        ao         = 1.0;
        alpha_cutoff = material.alpha_cutoff;

        if (material.base_texture_index != TEXTURE_NONE) {
            base_color *= texture(GetSampledTexture2D(material.base_texture_index), vout.uv);
        }

        // masked
        if (alpha_mode == 1) {
            // avoid artifacts at edge cases
            if (alpha_cutoff >= 1.0 || (material.base_texture_index == TEXTURE_NONE && base_color.a <= alpha_cutoff)) {
                discard;
            }
            
            if (material.base_texture_index != TEXTURE_NONE) {
                vec2 texture_size = textureSize(GetSampledTexture2D(material.base_texture_index), 0);
                base_color.a *= 1 + max(0, calc_mip_level(vout.uv * texture_size)) * MIP_SCALE;
                out_color.a = (base_color.a - alpha_cutoff) / max(fwidth(base_color.a), 0.0001) + 0.5;
            }
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
    
    // overdraw visualization
    if (GetBuffer(PerFrameBuffer, per_frame_buffer).render_mode == 7) {
        if (base_color.a < alpha_cutoff) {
            out_color = vec4(0.0);
        } else {
            out_color = vec4(1.0);
        }
        return;
    }

    
    uvec2 tile_id = uvec2(gl_FragCoord.xy) / GetBuffer(ClusterBuffer, cluster_buffer).tile_px_size;
    float z_near =  GetBuffer(PerFrameBuffer, per_frame_buffer).z_near;
    float z_scale = GetBuffer(ClusterBuffer, cluster_buffer).z_scale;
    float z_bias = GetBuffer(ClusterBuffer, cluster_buffer).z_bias;
    uint depth_slice_count = GetBuffer(ClusterBuffer, cluster_buffer).z_slice_count;
    uint depth_slice = linear_z_to_depth_slice(z_scale, z_bias, z_near / gl_FragCoord.z);

    uvec3 cluster_id = uvec3(tile_id, depth_slice);
    uint image_index = GetBuffer(ClusterBuffer, cluster_buffer).light_offset_image;
    uvec2 cluster_slice = imageLoad(GetImage3D(rg32ui, image_index), ivec3(cluster_id)).xy;
    uint light_offset = cluster_slice.x;
    uint light_count = clamp(cluster_slice.y, 0, 256);

    switch (GetBuffer(PerFrameBuffer, per_frame_buffer).render_mode) {
        case 0:
            vec3 view_direction = normalize(GetBuffer(PerFrameBuffer, per_frame_buffer).view_pos - vout.world_pos.xyz);
            
            vec3 light_sum = emissive;

            for (int i = 0; i < light_count; i++) {
                uint light_index = GetBuffer(ClusterLightIndices, GetBuffer(ClusterBuffer, cluster_buffer).light_index_buffer).light_indices[light_offset + i];
                vec3 light_color =
                    GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].color *
                    GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].intensity;

                switch (GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].light_type) {
                    case LIGHT_TYPE_SKY: {
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

                        vec3 irradiance = texture(GetSampledTextureCube(GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].irradiance_map_index), normal).rgb;
                        vec3 diffuse    = irradiance * base_color.rgb;

                        float max_reflection_lod = textureQueryLevels(GetSampledTextureCube(GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].prefiltered_map_index)) - 1;
                        float reflection_lod = roughness * max_reflection_lod;
                        vec3 reflection_color = 
                            textureLod(GetSampledTextureCube(GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].prefiltered_map_index), R, reflection_lod).rgb;
                        vec2 env_brdf = texture(
                            GetSampledTexture2D(brdf_integration_map_index),
                            vec2(max(dot(normal, view_direction), 0.0), roughness)
                        ).rg;
                        vec3 specular = reflection_color * (kS * env_brdf.x + env_brdf.y);

                        light_sum += (kD * diffuse + specular) * light_color * ao;
                    } break;
                    case LIGHT_TYPE_DIRECTIONAL: {
                        uint shadow_index = GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].shadow_data_index;
                        float shadow = 1.0;
                        if (shadow_index != TEXTURE_NONE) {
                            uint cascade_index = 4;
                            vec4 cascade_map_coord = vec4(0.0);


                            for (uint i = 0; i < MAX_SHADOW_CASCADE_COUNT; i++) {
                                vec4 map_coords = GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].light_projection_matrices[i] * vout.world_pos;
                                if (check_ndc_bounds(map_coords)) {
                                    cascade_index = i;
                                    cascade_map_coord = map_coords;
                                    break;
                                }
                            }
                            
                            if (cascade_index < MAX_SHADOW_CASCADE_COUNT) {
                                uint shadow_map = GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].shadow_map_indices[cascade_index];

                                float n_dot_l = dot(normal, GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].direction);
                                float normal_bias_scale = GetBuffer(ShadowSettingsBuffer, shadow_settings_buffer).data.normal_bias_scale;
                                
                                vec3 shadow_pos_world_space = vout.world_pos.xyz;

                                shadow_pos_world_space += shadow_normal_offset(n_dot_l, normal, shadow_map, normal_bias_scale);
                                
                                float oriented_bias = GetBuffer(ShadowSettingsBuffer, shadow_settings_buffer).data.oriented_bias;
                                vec3 light_direction = GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].direction;
                                shadow_pos_world_space += get_oriented_bias(normal, light_direction, oriented_bias, false) * light_direction;
                                
                                vec4 cascade_shadow_map_coords =
                                    GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].light_projection_matrices[cascade_index] *
                                    vec4(shadow_pos_world_space, 1.0);

                                float inv_world_size = 1.0 / GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].shadow_map_world_sizes[cascade_index];
                                float uv_light_size = GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].inner_radius * inv_world_size;

                                shadow = pcf_poisson(shadow_map, cascade_shadow_map_coords, inv_world_size, uv_light_size);
                            }
                        }

                        vec3 light_direction = GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].direction;
                        light_sum += calculate_light(
                            view_direction,
                            light_direction,
                            light_color,
                            1.0, // attenuation
                            base_color.rgb,
                            normal,
                            metallic,
                            roughness
                        ) * shadow;
                    } break;
                    case LIGHT_TYPE_POINT: {
                        vec3 light_direction = 
                            GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].position
                            - vout.world_pos.xyz;

                        float light_distance = length(light_direction);
                        light_direction /= light_distance;
                        light_distance = max(light_distance, GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].inner_radius);
                        
                        float attenuation = attenuation(
                            max(light_distance, GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].inner_radius),
                            GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].intensity, 
                            GetBuffer(ClusterBuffer, cluster_buffer).luminance_cutoff, 
                            GetBuffer(LightDataBuffer, light_data_buffer).lights[light_index].outer_radius
                        );

                        light_sum += calculate_light(
                            view_direction,
                            light_direction,
                            light_color,
                            attenuation,
                            base_color.rgb,
                            normal,
                            metallic,
                            roughness
                        );
                    } break;
                }

            }

            out_color.rgb = light_sum;
            break;
        case 1: {
            if (selected_light == TEXTURE_NONE) {
                out_color = vec4(vec3(0.25), 1.0);
                return;
            }
            uint shadow_index = GetBuffer(LightDataBuffer, light_data_buffer).lights[selected_light].shadow_data_index;
            float shadow = 1.0;
            vec3 cascade_color = vec3(0.25);
            if (selected_light != TEXTURE_NONE) {
                uint cascade_index = 4;
                vec4 cascade_map_coord = vec4(0.0);

                for (uint i = 0; i < MAX_SHADOW_CASCADE_COUNT; i++) {
                    vec4 map_coords = GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].light_projection_matrices[i] * vout.world_pos;
                    if (check_ndc_bounds(map_coords)) {
                        cascade_index = i;
                        cascade_map_coord = map_coords;
                        break;
                    }
                }
                
                if (cascade_index < MAX_SHADOW_CASCADE_COUNT) {
                    uint shadow_map = GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].shadow_map_indices[cascade_index];

                    float n_dot_l = dot(normal, GetBuffer(LightDataBuffer, light_data_buffer).lights[selected_light].direction);
                    float normal_bias_scale = GetBuffer(ShadowSettingsBuffer, shadow_settings_buffer).data.normal_bias_scale;
                    
                    vec3 shadow_pos_world_space = vout.world_pos.xyz;

                    shadow_pos_world_space += shadow_normal_offset(n_dot_l, normal, shadow_map, normal_bias_scale);
                    
                    vec3 light_direction = GetBuffer(LightDataBuffer, light_data_buffer).lights[selected_light].direction;
                    float oriented_bias = GetBuffer(ShadowSettingsBuffer, shadow_settings_buffer).data.oriented_bias;
                    shadow_pos_world_space += get_oriented_bias(normal, light_direction, oriented_bias, false) * light_direction;
                    
                    vec4 cascade_shadow_map_coords =
                        GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].light_projection_matrices[cascade_index] *
                        vec4(shadow_pos_world_space, 1.0);

                    float inv_world_size = 1.0 / GetBuffer(ShadowDataBuffer, shadow_data_buffer).shadows[shadow_index].shadow_map_world_sizes[cascade_index];
                    float uv_light_size = GetBuffer(LightDataBuffer, light_data_buffer).lights[selected_light].inner_radius * inv_world_size;

                    shadow = pcf_poisson(shadow_map, cascade_shadow_map_coords, inv_world_size, uv_light_size);
                }

                if (cascade_index < MAX_SHADOW_CASCADE_COUNT) cascade_color = DEBUG_COLORS[cascade_index];
            }
            
            vec3 light_direction = GetBuffer(LightDataBuffer, light_data_buffer).lights[selected_light].direction;
            out_color.xyz = cascade_color * debug_light(vout.normal, light_direction, max(shadow, 0.2));
        } break;
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
        case 8: 
            float norm_light_count = clamp(light_count / 32.0, 0.0, 1.0);
            out_color.xyz = colormap(norm_light_count).xyz;
            break;
    }
}