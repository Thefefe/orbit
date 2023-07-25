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
    vec3 color;
    float intensitiy;
    vec3 direction;
    uint _padding;
};

struct PerFrameData {
    mat4 view_projection;
    mat4 view;
    vec3 view_pos;
    uint render_mode;
};

struct MeshVertex {
    vec3 pos;
    uint _padding0;
    vec2 uv;
    uint _padding1;
    uint _padding2;
    vec3 norm;
    uint _padding3;
    vec4 tang;
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
    uint _padding0;
    uint _padding1;
};

struct MaterialData {
    vec4  base_color;

    vec3  emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float occulusion_factor;
    
    uint base_texture_index;
    uint normal_texture_index;
    uint metallic_roughness_texture_index;
    uint occulusion_texture_index;
    uint emissive_texture_index;
    
    uint _padding;
};

struct Submesh {
    uint entity_index;
    uint mesh_index;
    uint material_index;
    uint _padding;
};

struct Aabb {
    vec4 min_pos;
    vec4 max_pos;  
};

struct MeshInfo {
    uint index_offset;
    uint index_count;
    int  vertex_offset;
    float sphere_radius;
    Aabb aabb;
};

struct CullData {
    mat4 view_matrix;
    vec4 frustum_sides[4];
};

Buffer(16) CullDataBuffer {
    CullData data;  
};

Buffer(4) SubmeshBuffer {
	uint count;
	Submesh submeshes[];
};

Buffer(16) MeshInfoBuffer {
	MeshInfo mesh_infos[];
};

Buffer(4) DrawCommandsBuffer {
	uint count;
	DrawCommand commands[];
};

Buffer(16) VertexBuffer {
    MeshVertex vertices[];
};

Buffer(16) EntityBuffer {
    EntityData entities[];
};

Buffer(16) PerFrameBuffer {
    PerFrameData data;
};

Buffer(16) DirectionalLightBuffer {
    DirectionalLightData data;
};

Buffer(16) MaterialsBuffer {
    MaterialData materials[];
};