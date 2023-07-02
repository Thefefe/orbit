#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// layout(push_constant) uniform BindingIndexArray {
//     uint indices[32];
// } bindingIndexArray;

#define GetBufferRegistryName(Name) _u##Name##Registry
#define GetBindingIndexName(Name) _##Name##_BINDING_INDEX

#define IMMUTABLE_SAMPLER_COUNT 4

layout(set = 1, binding = 0) uniform sampler _uSamplers[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2D _uTextures[];

// Register storage buffer
#define RegisterBuffer(Name, Layout, BufferAccess, Struct) \
    layout(Layout, set = 0, binding = 0) \
    BufferAccess buffer Name Struct GetBufferRegistryName(Name)[]

#define BindSlot(Name, Index) \
    const uint GetBindingIndexName(Name) = Index

// Access a specific buffer
#define GetBuffer(Name, Index) \
    GetBufferRegistryName(Name)[nonuniformEXT(Index)]

#define GetSampler(Index) \
    _uSamplers[nonuniformEXT(Index)]

#define GetTexture(Index) \
    _uTextures[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define TEXTURE_NONE 0xFFFFFFFF

#define TEXTURE_INDEX_TEXTURE_MASK 0x3FFFFFFF
#define TEXTURE_INDEX_SAMPLER_MASK 0xC0000000

#define GET_SAMPLER_INDEX(Index) \
    ((Index & TEXTURE_INDEX_SAMPLER_MASK) >> 30)

#define GetSampledTexture(Index) \
    sampler2D(GetTexture(Index), _uSamplers[nonuniformEXT(GET_SAMPLER_INDEX(Index))])

vec4 sample_texture_index_default(uint texture_index, vec2 tex_coord, vec4 _default) {
    if (texture_index == TEXTURE_NONE) return _default;
    return texture(GetSampledTexture(texture_index), tex_coord);
}