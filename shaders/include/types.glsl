#include "common.glsl"

#define MAX_SHADOW_CASCADE_COUNT 4

struct GlobalData {
    ivec2 screen_size;
    uint elapsed_frames;
    float elapsed_time;
};

struct DirectionalLightData {
    mat4 projection_matrices[MAX_SHADOW_CASCADE_COUNT];
    uint shadow_maps[MAX_SHADOW_CASCADE_COUNT];
    vec4 cascade_distances;
    vec3 color;
    float intensitiy;
    vec3 direction;
    float split_blend_ratio;

    float penumbra_filter_max_size;
    float min_filter_radius;
    float max_filter_radius;
    float normal_bias_scale;
};

struct PerFrameData {
    mat4 view_projection;
    mat4 view;
    vec3 view_pos;
    uint render_mode;
};

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
};

struct Submesh {
    uint entity_index;
    uint mesh_index;
    uint material_index;
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
    vec4 cull_planes[12];
    uint cull_plane_count;

    uint  occlusion_pass;
    uint  visibility_buffer;
    uint  depth_pyramid;
    float p00;
    float p11;
    float z_near;
    uint  _padding;
};

struct LightData {
    vec3 color;
    float intensity;
    vec3 position;
    float size;
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

RegisterBuffer(VisibilityBuffer, {
    uint submeshes[];
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

RegisterBuffer(LightBuffer, {
	LightData lights[];
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

RegisterBuffer(PerFrameBuffer, {
    PerFrameData data;
});

RegisterBuffer(DirectionalLightBuffer, {
    DirectionalLightData data;
});

RegisterBuffer(MaterialsBuffer, {
    MaterialData materials[];
});