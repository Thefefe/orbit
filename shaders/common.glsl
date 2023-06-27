#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(push_constant) uniform BindingIndexArray {
    uint indices[32];
} bindingIndexArray;

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

#define TEXTURE_INDEX_TEXTURE_MASK 0x3FFFFFFF
#define TEXTURE_INDEX_SAMPLER_MASK 0xC0000000

uint get_sampler_index(uint image_index) {
    return (image_index & TEXTURE_INDEX_SAMPLER_MASK) >> 30;
}

#define GetSampledTextureByIndex(Index) \
    sampler2D(GetTextureByIndex(Index), _uSamplers[nonuniformEXT(get_sampler_index(Index))])