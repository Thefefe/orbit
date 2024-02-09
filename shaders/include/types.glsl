#include "common.glsl"

#define MAX_SHADOW_CASCADE_COUNT 4
#define MAX_MESH_LODS 8

// keep in sync with the ones in src/assets/mesh.rs
#define MESHLET_VERTEX_COUNT 64
#define MESHLET_TRIANGLE_COUNT 64

struct GlobalData {
    ivec2 screen_size;
    uint elapsed_frames;
    float elapsed_time;
};

struct LightData {
    uint  light_type;
    uint  shadow_data_index;
    uint  irradiance_map_index;
    uint  prefiltered_map_index;
    vec3  color;
    float intensity;
    vec3  position;
    float inner_radius;
    vec3  direction;
    float outer_radius;
};

struct ShadowData {
    mat4 light_projection_matrices[MAX_SHADOW_CASCADE_COUNT];
    vec4 shadow_map_world_sizes;
    uint shadow_map_indices[MAX_SHADOW_CASCADE_COUNT];
};

struct ShadowSettings {
    float blocker_search_radius;
    float normal_bias_scale;
    float oriented_bias;
    uint _padding;
};

RegisterBuffer(ClusterBuffer, {
    uvec3 cluster_count;
    uint  tile_px_size;
    uvec2 screen_size;
    uint  z_slice_count;
    float z_scale;
    float z_bias;
    float luminance_cutoff;
    uint  light_offset_image;
    uint  light_index_buffer;
    uint  tile_depth_slice_mask;
});

RegisterBuffer(PerFrameBuffer, {
    mat4  view_projection;
    mat4  view;
    vec3  view_pos;
    uint  render_mode;
    uvec2 screen_size;
    float z_near;
});

struct MeshVertex {
    // float position[3];
    // i8vec4 packed_normals;
    // float uv_coord[2];
    vec3 position;
    i8vec4 normal;
    vec2 uv;
    i8vec4 tangent;
    uint _padding;
};

struct EntityData {
    mat4 model_matrix;
    mat4 normal_matrix;
};

struct DrawCommand {
    // DrawIndiexedIndirectCommand
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;

    // other per-draw data
    uint material_index;
};

struct MaterialData {
    vec4  base_color;

    vec3  emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float occlusion_factor;
    
    float alpha_cutoff;

    uint base_texture_index;
    uint normal_texture_index;
    uint metallic_roughness_texture_index;
    uint occlusion_texture_index;
    uint emissive_texture_index;

    uint alpha_mode;
    uint _padding[3];
};

struct EntityDraw {
    uint entity_index;
    uint mesh_index;
    uint visibility_offset;
};

RegisterBuffer(EntityDrawBuffer, {
    uint count;
    EntityDraw draws[];
});

struct Aabb {
    vec4 min_pos;
    vec4 max_pos;  
};

struct MeshLod {
    uint meshlet_offset;
    uint meshlet_count;
};

struct MeshInfo {
    vec4 bounding_sphere;
    Aabb aabb;
    uint vertex_offset;
    uint meshlet_data_offset;
    uint lod_count;
    uint _padding;
    MeshLod mesh_lods[MAX_MESH_LODS];
};

struct Meshlet {
    vec4     bounding_sphere;
    i8vec3   cone_axis;
    int8_t   cone_cutoff;
    uint32_t vertex_offset;
    uint32_t data_offset;
    uint16_t material_index;
    uint8_t  vertex_count;
    uint8_t  triangle_count;
};

RegisterBuffer(MeshletBuffer, {
    Meshlet meshlets[];
});

RegisterBuffer(MeshletDataBuffer, {
    uint vertex_indices[];
});

RegisterBuffer(MeshletTriangleDataBuffer, {
    uint8_t indices[];
});

struct MeshletDispatch {
    uint entity_index;
    uint meshlet_offset;
    uint meshlet_count;
    uint visibility_offset;
};

RegisterBuffer(MeshletDispatchBuffer, {
    uint workgroup_count_x;
    uint workgroup_count_y;
    uint workgroup_count_z;
    MeshletDispatch dispatches[];
});

struct MeshletDrawCommand {
    uint cmd_index_count;
    uint cmd_instance_count;
    uint cmd_first_index;
    int  cmd_vertex_offset;
    uint cmd_first_instance;

    uint meshlet_vertex_offset;
    uint meshlet_index;
};

RegisterBuffer(MeshletDrawCommandBuffer, {
    uint count;
    MeshletDrawCommand draws[];
});
    
struct MeshTaskPayload {
    uint entity_index;
    uint meshlet_offset;
    uint8_t meshlet_indices[32];
};

struct CullInfo {
    mat4 view_matrix;
    mat4 reprojection_matrix;
    vec4 cull_planes[12];
    uint cull_plane_count;

    uint alpha_mode_flag;
    uint noskip_alphamode;

    uint  occlusion_pass;
    uint  visibility_buffer;
    uint  meshlet_visibility_buffer;
    uint  depth_pyramid;
    uint  secondary_depth_pyramid;

    uint  projection_type;
    float p00_or_width_recipx2;
    float p11_or_height_recipx2;
    float z_near;
    float z_far;
    
    float lod_base;
    float lod_step;
    uint  min_mesh_lod;
    vec3  lod_target_pos_view_space;
    uint  max_mesh_lod;
};

struct EguiVertex {
    float position[2];
    float uv_coord[2];
    u8vec4 color;
};

struct DebugLineVertex {
    vec3   position;
    u8vec4 color;
};

struct DebugMeshInstance {
    mat4 matrix;
    vec4 color;
};

RegisterBuffer(ClusterLightIndices, {
    uint light_count;
    uint light_indices[];
});

struct ClusterDepthBounds {
    // floats
    uint min_depth;
    uint max_depth;
};

RegisterBuffer(ClusterDepthBoundsBuffer, {
    ClusterDepthBounds clusters[];
});

RegisterBuffer(ClusterVolumeBuffer, {
    Aabb clusters[];
});

RegisterBuffer(TileDepthSliceMask, {
    uint masks[];
});

// also doubles as indirect dispatch arguments
RegisterBuffer(CompactedClusterIndexList, {
    uint workgroup_count_x;
    uint workgroup_count_y;
    uint workgroup_count_z;
    uint cluster_count;
    uint cluster_indices[];
});

RegisterBuffer(DebugMeshInstanceBuffer, {
    DebugMeshInstance instances[];
});

RegisterBuffer(VisibilityBuffer, {
    uint32_t entries[];
});

RegisterBuffer(CullInfoBuffer, {
    CullInfo cull_info;
});

RegisterBuffer(EguiVertexBuffer, {
    EguiVertex vertices[];
});

RegisterBuffer(DebugLineVertexBuffer, {
    DebugLineVertex vertices[];
});

#define LIGHT_TYPE_SKY 0
#define LIGHT_TYPE_DIRECTIONAL 1
#define LIGHT_TYPE_POINT 2

RegisterBuffer(LightDataBuffer, {
	LightData lights[];
});

RegisterBuffer(ShadowDataBuffer, {
	ShadowData shadows[];
});

RegisterBuffer(ShadowSettingsBuffer, {
	ShadowSettings data;
});

RegisterBuffer(MeshInfoBuffer, {
	MeshInfo mesh_infos[];
});

RegisterBuffer(DrawCommandsBuffer, {
	uint count;
	DrawCommand draws[];
});

RegisterBuffer(VertexBuffer, {
    MeshVertex vertices[];
});

RegisterBuffer(EntityBuffer, {
    EntityData entities[];
});

RegisterBuffer(MaterialsBuffer, {
    MaterialData materials[];
});