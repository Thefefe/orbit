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
    vec4 base_color;
    uint base_texture_index;
    uint normal_texture_index;
    vec2 _padding;
};

struct Submesh {
    uint entity_index;
    uint mesh_index;
    uint material_index;
    uint _padding;   
};

struct MeshInfo {
    uint index_offset;
    uint index_count;
    int  vertex_offset;
    uint _padding;
};