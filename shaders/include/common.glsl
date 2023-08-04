#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

const float PI = 3.14159265359;
const float EPSILON = 0.0000001;

#define IMMUTABLE_SAMPLER_COUNT 6
#define SHADOW_SAMPLER 4
#define SHADOW_DEPTH_SAMPLER 5

layout(set = 1, binding = 0) uniform sampler _u_sampler_registry[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = 0) uniform samplerShadow _u_sampler_comparisson_registry[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2D _u_texture2d_registry[];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2DMS _u_texture2dms_registry[];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform textureCube _u_texture_cube_registry[];

#define GetBufferRegistryName(Name) _u_##Name##Registry

#define RegisterBuffer(Name, Struct) \
    layout(std430, set = 0, binding = 0) buffer Name Struct GetBufferRegistryName(Name)[]

#define GetBuffer(Name, Index) \
    GetBufferRegistryName(Name)[nonuniformEXT(Index)]

#define GetSampler(Index) \
    _u_sampler_registry[nonuniformEXT(Index)]

#define GetCompSampler(Index) \
    _u_sampler_comparisson_registry[nonuniformEXT(Index)]

#define GetTexture2D(Index) \
    _u_texture2d_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define GetTexture2DMS(Index) \
    _u_texture2dms_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define GetTextureCube(Index) \
    _u_texture_cube_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define TEXTURE_NONE 0xFFFFFFFF

#define TEXTURE_INDEX_TEXTURE_MASK 0x00FFFFFF
#define TEXTURE_INDEX_SAMPLER_MASK 0xFF000000
#define SAMPLER_BIT_COUNT 24

#define GetSamplerIndex(Index) \
    ((Index & TEXTURE_INDEX_SAMPLER_MASK) >> SAMPLER_BIT_COUNT)

#define GetSampledTexture2D(Index) \
    sampler2D(GetTexture2D(Index), GetSampler(GetSamplerIndex(Index)))

#define GetSampledTexture2DMS(Index) \
    sampler2DMS(GetTexture2DMS(Index), GetSampler(GetSamplerIndex(Index)))

#define GetSampledTextureCube(Index) \
    samplerCube(GetTextureCube(Index), GetSampler(GetSamplerIndex(Index)))

vec4 sample_texture_index_default(uint texture_index, vec2 tex_coord, vec4 _default) {
    if (texture_index == TEXTURE_NONE) return _default;
    return texture(GetSampledTexture2D(texture_index), tex_coord);
}
#endif