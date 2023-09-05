#include "include/common.glsl"
#include "include/types.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant, std430) uniform PushConstants {
    uint image_index;
    float exposure;
};

const mat3 ACES_INPUT_MATRIX = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);

const mat3 ACES_OUTPUT_MATRIX = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

vec3 aces_rrt_and_odt_fit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

//https://github.com/Shimmen/ArkoseRendererThesis/blob/master/shaders/aces.glsl
vec3 aces_hill(vec3 color) {
    color = ACES_INPUT_MATRIX * color;
    color = aces_rrt_and_odt_fit(color);
    color = ACES_OUTPUT_MATRIX * color;
    color = clamp(color, vec3(0.0), vec3(1.0));

    return color;
}

vec3 aces_narkowicz(vec3 col) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((col * (a * col + b)) / (col * (c * col + d) + e), 0.0, 1.0);
}

float luminance(vec3 radiance) {
    return 0.2125 * radiance.r + 0.7154 * radiance.g +0.0721 * radiance.b;
}

void main() {
    vec3 hdr_color = texture(GetSampledTexture2D(image_index), in_uv).rgb;
    // vec3 mapped = aces_narkowicz(hdr_color * exposure);
    vec3 mapped = aces_hill(hdr_color * exposure);
    out_color = vec4(mapped, 1.0);
}