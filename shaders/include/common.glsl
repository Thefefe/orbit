#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_buffer_reference : require

const float PI = 3.14159265359;
const float EPSILON = 0.0000001;

#define Buffer(Alignment) \
  layout(buffer_reference, std430, buffer_reference_align = Alignment) buffer

#define IMMUTABLE_SAMPLER_COUNT 5
#define SHADOW_SAMPLER 4

layout(set = 1, binding = 0) uniform sampler _uSamplers[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = 0) uniform samplerShadow _uComparisonSamplers[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2D _uTextures2D[];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2DMS _uTextures2DMS[];

#define GetSampler(Index) \
    _uSamplers[nonuniformEXT(Index)]

#define GetTexture2D(Index) \
    _uTextures2D[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define GetTexture2DMS(Index) \
    _uTextures2DMS[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define TEXTURE_NONE 0xFFFFFFFF

#define TEXTURE_INDEX_TEXTURE_MASK 0x00FFFFFF
#define TEXTURE_INDEX_SAMPLER_MASK 0xFF000000
#define SAMPLER_BIT_COUNT 24

#define GET_SAMPLER_INDEX(Index) \
    ((Index & TEXTURE_INDEX_SAMPLER_MASK) >> SAMPLER_BIT_COUNT)

#define GetSampledTexture2D(Index) \
    sampler2D(GetTexture2D(Index), _uSamplers[nonuniformEXT(GET_SAMPLER_INDEX(Index))])

#define GetSampledTexture2DMS(Index) \
    sampler2DMS(GetTexture2DMS(Index), _uSamplers[nonuniformEXT(GET_SAMPLER_INDEX(Index))])

vec4 sample_texture_index_default(uint texture_index, vec2 tex_coord, vec4 _default) {
    if (texture_index == TEXTURE_NONE) return _default;
    return texture(GetSampledTexture2D(texture_index), tex_coord);
}
#endif