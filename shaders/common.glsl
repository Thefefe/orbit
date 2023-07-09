#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(push_constant) uniform BindingIndexArray {
    uint indices[32];
} bindingIndexArray;

#define GetBufferRegistryName(Name) _u##Name##Registry
#define GetBindingIndexName(Name) _##Name##_BINDING_INDEX

#define IMMUTABLE_SAMPLER_COUNT 5

layout(set = 1, binding = 0) uniform sampler _uSamplers[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2D _uTextures[];

// Register storage buffer
#define RegisterBuffer(Name, Layout, BufferAccess, Struct) \
    layout(Layout, set = 0, binding = 0) \
    BufferAccess buffer Name Struct GetBufferRegistryName(Name)[]

#define BindSlot(Name, Index) \
    const uint GetBindingIndexName(Name) = Index

// Access a specific buffer
#define GetBuffer(Name) \
    GetBufferRegistryName(Name)[nonuniformEXT(bindingIndexArray.indices[GetBindingIndexName(Name)])]

#define GetFloat(Name) \
    uintBitsToFloat(bindingIndexArray.indices[GetBindingIndexName(Name)])

#define GetSampler(Name) \
    _uSamplers[nonuniformEXT(bindingIndexArray.indices[GetBindingIndexName(Name)])]

#define GetTexture(Name) \
    _uTextures[nonuniformEXT(bindingIndexArray.indices[GetBindingIndexName(Name)])]


#define GetBufferByIndex(Index) \
    GetBufferRegistryName(Name)[nonuniformEXT(Index)]

#define GetTextureByIndex(Index) \
    _uTextures[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define TEXTURE_NONE 0xFFFFFFFF

#define TEXTURE_INDEX_TEXTURE_MASK 0x00FFFFFF
#define TEXTURE_INDEX_SAMPLER_MASK 0xFF000000
#define SAMPLER_BIT_COUNT 24

#define GET_SAMPLER_INDEX(Index) \
    ((Index & TEXTURE_INDEX_SAMPLER_MASK) >> SAMPLER_BIT_COUNT)

#define GetSampledTextureByIndex(Index) \
    sampler2D(GetTextureByIndex(Index), _uSamplers[nonuniformEXT(GET_SAMPLER_INDEX(Index))])

vec4 sample_texture_index_default(uint texture_index, vec2 tex_coord, vec4 _default) {
    if (texture_index == TEXTURE_NONE) return _default;
    return texture(GetSampledTextureByIndex(texture_index), tex_coord);
}