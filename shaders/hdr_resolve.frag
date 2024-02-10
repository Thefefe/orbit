#include "include/common.glsl"
#include "include/types.glsl"
#include "include/functions.glsl"

float max_comp(vec3 v) {
    return max(v.x, max(v.y, v.z));
}

vec3 tonemap(vec3 c) {
    return c / (max_comp(c) + 1.0);
}

vec3 tonemap_with_weight(vec3 c, float w) {
    return c * (w / (max_comp(c) + 1.0));
}

vec3 tonemap_invert(vec3 c) {
    return c / (1.0 - max_comp(c));
}

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout (constant_id = 0) const uint SAMPLE_COUNT = 1;

layout(push_constant, std430) uniform PushConstants {
    uvec2 image_size;
    uint msaa_image;
};

void main() {
    ivec2 coords = ivec2(vec2(image_size) * in_uv);
    switch(SAMPLE_COUNT) {
        case 1: {
            vec3 sample0 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 0).xyz;

            out_color.xyz = tonemap_invert(
                tonemap_with_weight(sample0, 1.0)
            );
            out_color = vec4(1.0);
        } break;
        case 2: {
            vec3 sample0 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 0).xyz;
            vec3 sample1 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 1).xyz;

            out_color.xyz = tonemap_invert(
                tonemap_with_weight(sample0, 0.5) +
                tonemap_with_weight(sample1, 0.5)
            );
        } break;
        case 4: {
            vec3 sample0 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 0).xyz;
            vec3 sample1 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 1).xyz;
            vec3 sample2 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 2).xyz;
            vec3 sample3 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 3).xyz;

            out_color.xyz = tonemap_invert(
                tonemap_with_weight(sample0, 0.25) +
                tonemap_with_weight(sample1, 0.25) +
                tonemap_with_weight(sample2, 0.25) +
                tonemap_with_weight(sample3, 0.25)
            );
        } break;
        case 8: {
            vec3 sample0 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 0).xyz;
            vec3 sample1 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 1).xyz;
            vec3 sample2 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 2).xyz;
            vec3 sample3 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 3).xyz;
            vec3 sample4 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 4).xyz;
            vec3 sample5 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 5).xyz;
            vec3 sample6 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 6).xyz;
            vec3 sample7 = texelFetch(GetSampledTexture2DMS(msaa_image), coords, 7).xyz;

            out_color.xyz = tonemap_invert(
                tonemap_with_weight(sample0, 0.125) +
                tonemap_with_weight(sample1, 0.125) +
                tonemap_with_weight(sample2, 0.125) +
                tonemap_with_weight(sample3, 0.125) +
                tonemap_with_weight(sample4, 0.125) +
                tonemap_with_weight(sample5, 0.125) +
                tonemap_with_weight(sample6, 0.125) +
                tonemap_with_weight(sample7, 0.125)
            );
        } break;
    }
}