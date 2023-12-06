#include "common.glsl"

#define MAX_SHADOW_CASCADE_COUNT 4

struct GlobalData {
    ivec2 screen_size;
    uint elapsed_frames;
    float elapsed_time;
};

// struct DirectionalLightData {
//     mat4 projection_matrices[MAX_SHADOW_CASCADE_COUNT];
//     uint shadow_maps[MAX_SHADOW_CASCADE_COUNT];
//     float cascade_world_sizes[MAX_SHADOW_CASCADE_COUNT];
//     vec4 cascade_distances;
//     vec3 color;
//     float intensitiy;
//     vec3 direction;
//     float light_size;
//     float blocker_search_radius;
//     float normal_bias_scale;
//     float oriented_bias;
//     uint _padding;
// };

struct LightData {
    vec3  color;
    float intensity;
    vec3  direction_or_position;
    float size;
    uint  light_type;
    uint  shadow_data_index;
    uint  irradiance_map_index;
    uint  prefiltered_map_index;
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

struct ClusterGridInfo {
    float z_scale;
    float z_bias;
    uint  z_slice_count;
};

RegisterBuffer(PerFrameBuffer, {
    mat4  view_projection;
    mat4  view;
    vec3  view_pos;
    uint  render_mode;
    float z_near;
    ClusterGridInfo cluster_info;
});

struct MeshVertex {
    float position[3];
    i8vec4 packed_normals;
    float uv_coord[2];
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
    
    uint _debug_index; // for debuging only
    
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

struct Submesh {
    uint entity_index;
    uint mesh_index;
    uint material_index;
    uint alpha_mode;
};

struct Aabb {
    float min_pos[3];
    float max_pos[3];  
};

struct MeshInfo {
    uint index_offset;
    uint index_count;
    int  vertex_offset;
    Aabb aabb;
    float bounding_sphere[4];
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
    uint  depth_pyramid;
    uint  secondary_depth_pyramid;

    uint  projection_type;
    float p00_or_width_recipx2;
    float p11_or_height_recipx2;
    float z_near;
    float z_far;
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

struct ClusterVolume {
    vec4 min_pos;
    vec4 max_pos;
};

RegisterBuffer(ClusterVolumeBuffer, {
    ClusterVolume clusters[];
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
    uint32_t submeshes[];
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

RegisterBuffer(SubmeshBuffer, {
	uint count;
	Submesh submeshes[];
});

RegisterBuffer(MeshInfoBuffer, {
	MeshInfo mesh_infos[];
});

RegisterBuffer(DrawCommandsBuffer, {
	uint count;
	DrawCommand commands[];
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