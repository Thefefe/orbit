#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

const float PI = 3.14159265359;
const float EPSILON = 0.0001;

#define IMMUTABLE_SAMPLER_COUNT 7
#define SHADOW_SAMPLER 4
#define SHADOW_DEPTH_SAMPLER 5
#define REDUCE_MIN_SAMPLER 6

layout(set = 1, binding = 0) uniform sampler _u_sampler_registry[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = 0) uniform samplerShadow _u_sampler_comparisson_registry[IMMUTABLE_SAMPLER_COUNT];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture1D _u_texture1d_registry[];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2D _u_texture2d_registry[];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture3D _u_texture3d_registry[];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform texture2DMS _u_texture2dms_registry[];
layout(set = 1, binding = IMMUTABLE_SAMPLER_COUNT) uniform textureCube _u_texture_cube_registry[];

#define RegisterImageFormat(Type, Format) \
    layout(set = 2, binding = 0, Format) uniform Type##2D _u_##Type##2D##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##3D _u_##Type##3D##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##Cube _u_##Type##Cube##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##2DArray _u_##Type##2DArray##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##CubeArray _u_##Type##CubeArray##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##1D _u_##Type##1D##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##1DArray _u_##Type##1DArray##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##2DMS _u_##Type##2DMS##_##Format##_registry[]; \
    layout(set = 2, binding = 0, Format) uniform Type##2DMSArray _u_##Type##2DMSArray##_##Format##_registry[]

// float
RegisterImageFormat(image, rgba32f);
RegisterImageFormat(image, rgba16f);
RegisterImageFormat(image, rg32f);
RegisterImageFormat(image, rg16f);
RegisterImageFormat(image, r11f_g11f_b10f);
RegisterImageFormat(image, r32f);
RegisterImageFormat(image, r16f);
RegisterImageFormat(image, rgba16);
RegisterImageFormat(image, rgb10_a2);
RegisterImageFormat(image, rgba8);
RegisterImageFormat(image, rg16);
RegisterImageFormat(image, rg8);
RegisterImageFormat(image, r16);
RegisterImageFormat(image, r8);
RegisterImageFormat(image, rgba16_snorm);
RegisterImageFormat(image, rgba8_snorm);
RegisterImageFormat(image, rg16_snorm);
RegisterImageFormat(image, rg8_snorm);
RegisterImageFormat(image, r16_snorm);
RegisterImageFormat(image, r8_snorm);

// signed int
RegisterImageFormat(iimage, rgba32i);
RegisterImageFormat(iimage, rgba16i);
RegisterImageFormat(iimage, rgba8i);
RegisterImageFormat(iimage, rg32i);
RegisterImageFormat(iimage, rg16i);
RegisterImageFormat(iimage, rg8i);
RegisterImageFormat(iimage, r32i);
RegisterImageFormat(iimage, r16i);
RegisterImageFormat(iimage, r8i);

//unsigned image format
RegisterImageFormat(uimage, rgba32ui);
RegisterImageFormat(uimage, rgba16ui);
RegisterImageFormat(uimage, rgb10_a2ui);
RegisterImageFormat(uimage, rgba8ui);
RegisterImageFormat(uimage, rg32ui);
RegisterImageFormat(uimage, rg16ui);
RegisterImageFormat(uimage, rg8ui);
RegisterImageFormat(uimage, r32ui);
RegisterImageFormat(uimage, r16ui);
RegisterImageFormat(uimage, r8ui);

#define GetImage(TypeWithDim, Format, Index) \
    _u_##TypeWithDim##_##Format##_registry[nonuniformEXT(Index)]

#define GetBufferRegistryName(Name) _u_##Name##Registry

#define RegisterBuffer(Name, Struct) \
    layout(std430, set = 0, binding = 0) buffer Name Struct GetBufferRegistryName(Name)[]

#define GetBuffer(Name, Index) \
    GetBufferRegistryName(Name)[nonuniformEXT(Index)]

#define GetSampler(Index) \
    _u_sampler_registry[nonuniformEXT(Index)]

#define GetCompSampler(Index) \
    _u_sampler_comparisson_registry[nonuniformEXT(Index)]


#define GetTexture1D(Index) \
    _u_texture1d_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define GetTexture2D(Index) \
    _u_texture2d_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define GetTexture3D(Index) \
    _u_texture3d_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define GetTexture2DMS(Index) \
    _u_texture2dms_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define GetTextureCube(Index) \
    _u_texture_cube_registry[nonuniformEXT(Index & TEXTURE_INDEX_TEXTURE_MASK)]

#define TEXTURE_NONE 0xFFFFFFFF
#define TEXTURE_INDEX_TEXTURE_MASK 0x00FFFFFF
#define TEXTURE_INDEX_SAMPLER_MASK 0xFF000000
#define SAMPLER_BIT_COUNT 24

#define GetTextureIndex(Index) (Index << 8 >> 8)
#define GetSamplerIndex(Index) \
    ((Index & TEXTURE_INDEX_SAMPLER_MASK) >> SAMPLER_BIT_COUNT)

#define GetSampledTexture1D(Index) \
    sampler1D(GetTexture1D(Index), GetSampler(GetSamplerIndex(Index)))

#define GetSampledTexture2D(Index) \
    sampler2D(GetTexture2D(Index), GetSampler(GetSamplerIndex(Index)))

#define GetSampledTexture3D(Index) \
    sampler3D(GetTexture3D(Index), GetSampler(GetSamplerIndex(Index)))

#define GetSampledTexture2DMS(Index) \
    sampler2DMS(GetTexture2DMS(Index), GetSampler(GetSamplerIndex(Index)))

#define GetSampledTextureCube(Index) \
    samplerCube(GetTextureCube(Index), GetSampler(GetSamplerIndex(Index)))

#endif