#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(push_constant) uniform BindingIndexArray {
    uint indices[32];
} bindingIndexArray;

#define GetBufferRegistryName(Name) _u##Name##Registry
#define GetBindingIndexName(Name) _##Name##_BINDING_INDEX

layout(set = 1, binding = 0) uniform sampler _uSamplers[1];
layout(set = 1, binding = 1) uniform texture2D _uTextures[];

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

#define GetSamplerByIndex(Index) \
    _uSamplers[nonuniformEXT(Index)]

#define GetTextureByIndex(Index) \
    _uTextures[nonuniformEXT(Index)]

#define TEXTURE_NONE 0xFFFFFFFF