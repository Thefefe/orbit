use crate::render::DescriptorHandle;
use crate::{render, collections::arena};
use crate::collections::freelist_alloc::*;

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4};
use gpu_allocator::MemoryLocation;
use ash::vk;

//    4    8   12   16
// PPPP PPPP PPPP
// UUUU UUUU
// NNNN NNNN NNNN
// TTTT TTTT TTTT TTTT
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshVertex {
    position: Vec3,
    _padding0: u32,
    uv_coord: Vec2,
    _padding1: [u32; 2],
    normal: Vec3,
    _padding2: u32,
    tangent: Vec4,
}

impl GpuMeshVertex {
    pub fn new(
        position: Vec3,
        uv_coord: Vec2,
        normal: Vec3,
        tangent: Vec4
    ) -> Self {
        Self {
            position,
            _padding0: 0,
            uv_coord,
            _padding1: [0; 2],
            normal,
            _padding2: 0,
            tangent,
        }
    }

    pub fn from_arrays(
        position: [f32; 3],
        uv_coord: [f32; 2],
        normal: [f32; 3],
        tangent: [f32; 4],
    ) -> Self {
        Self {
            position: Vec3::from_array(position),
            uv_coord: Vec2::from_array(uv_coord),
            normal: Vec3::from_array(normal),
            tangent: Vec4::from_array(tangent),
            _padding0: 0,
            _padding1: [0; 2],
            _padding2: 0,
        }
    }
}


#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshInfo {
    index_offset: u32,
    index_count: u32,
    vertex_offset: u32,
    _padding: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshBlock {
    pub vertex_range: BlockRange,
    pub index_range: BlockRange,
    pub vertex_alloc_index: arena::Index,
    pub index_alloc_index: arena::Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Submesh {
    pub mesh_handle: MeshHandle,
    pub material_index: MaterialHandle,
}

pub struct ModelData {
    pub submeshes: Vec<Submesh>,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialData {
    pub base_color: Vec4,
    pub base_texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMaterialData {
    base_color: Vec4,
    base_texture_index: u32,
    normal_texture_index: u32,
    _padding: [u32; 2],
}

pub type MeshHandle = arena::Index;
pub type ModelHandle = arena::Index;
pub type TextureHandle = arena::Index;
pub type MaterialHandle = arena::Index;

const MAX_VERTEX_COUNT: usize = 4_000_000;
const MAX_INDEX_COUNT: usize = 12_000_000;
const MAX_MATERIAL_COUNT: usize = 1_000;

pub struct GpuAssetStore {
    pub vertex_buffer: render::Buffer,
    pub index_buffer: render::Buffer,

    // INDICES!!! not bytes
    vertex_allocator: FreeListAllocator,
    index_allocator: FreeListAllocator,

    pub mesh_info_buffer: render::Buffer,
    pub mesh_blocks: arena::Arena<MeshBlock>,
    pub models: arena::Arena<ModelData>,

    pub textures: arena::Arena<render::Image>,

    pub material_buffer: render::Buffer,
    pub material_indices: arena::Arena<MaterialData>,
}

impl GpuAssetStore {
    pub fn new(context: &render::Context) -> Self {
        let vertex_buffer = context.create_buffer("mesh_vertex_buffer", &render::BufferDesc {
            size: MAX_VERTEX_COUNT * std::mem::size_of::<GpuMeshVertex>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        let index_buffer = context.create_buffer("mesh_index_buffer", &render::BufferDesc {
            size: MAX_INDEX_COUNT * std::mem::size_of::<u32>(),
            usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        let mesh_info_buffer = context.create_buffer("mesh_info_buffer", &render::BufferDesc {
            size: MAX_INDEX_COUNT * std::mem::size_of::<GpuMeshInfo>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        let material_buffer = context.create_buffer("material_buffer", &render::BufferDesc {
            size: MAX_MATERIAL_COUNT * std::mem::size_of::<GpuMaterialData>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        Self {
            vertex_buffer,
            index_buffer,
            
            vertex_allocator: FreeListAllocator::new(MAX_VERTEX_COUNT),
            index_allocator: FreeListAllocator::new(MAX_INDEX_COUNT),

            mesh_info_buffer,
            mesh_blocks: arena::Arena::new(),
            models: arena::Arena::new(),

            textures: arena::Arena::new(),

            material_buffer,
            material_indices: arena::Arena::new(),
        }
    }

    pub fn add_mesh(&mut self, context: &render::Context, mesh: &MeshData) -> MeshHandle {
        let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);

        let (vertex_alloc_index, vertex_range) = self.vertex_allocator.allocate(mesh.vertices.len()).unwrap();
        let (index_alloc_index, index_range) = self.index_allocator.allocate(mesh.indices.len()).unwrap();

        context.immediate_write_buffer(&self.vertex_buffer, vertex_bytes, vertex_range.start * std::mem::size_of::<GpuMeshVertex>());
        context.immediate_write_buffer(&self.index_buffer, index_bytes, index_range.start * std::mem::size_of::<u32>());

        let mesh_block = MeshBlock {
            vertex_range,
            index_range,
            vertex_alloc_index,
            index_alloc_index,
        };
        let mesh_index = self.mesh_blocks.insert(mesh_block);


        let mesh_info = GpuMeshInfo {
            index_offset: index_range.start as u32,
            index_count: index_range.size() as u32,
            vertex_offset: vertex_range.start as u32,
            _padding: 0,
        };
        context.immediate_write_buffer(
            &self.mesh_info_buffer,
            bytemuck::bytes_of(&mesh_info), 
            std::mem::size_of::<GpuMeshInfo>() * mesh_index.slot() as usize
        );

        mesh_index
    }

    pub fn add_model(&mut self, submeshes: &[Submesh]) -> ModelHandle {
        self.models.insert(ModelData {
            submeshes: submeshes.to_vec(),
        })
    }

    pub fn import_texture(&mut self, image: render::Image)-> TextureHandle {
        self.textures.insert(image)
    }

    fn get_texture_desc_index(&self, handle: TextureHandle) -> u32 {
        self.textures[handle].descriptor_index.unwrap().to_raw()
    }

    pub fn add_material(&mut self, context: &render::Context, material_data: MaterialData) -> MaterialHandle {
        let index = self.material_indices.insert(material_data);
        let base_texture_index = material_data.base_texture
            .map(|handle| self.get_texture_desc_index(handle))
            .unwrap_or(u32::MAX);
        let normal_texture_index = material_data.normal_texture
            .map(|handle| self.get_texture_desc_index(handle))
            .unwrap_or(u32::MAX);


        let gpu_data = GpuMaterialData {
            base_color: material_data.base_color,
            base_texture_index,
            normal_texture_index,
            _padding: [0; 2],
        };

        context.immediate_write_buffer(
            &self.material_buffer,
            bytemuck::bytes_of(&gpu_data),
            index.slot_index() * std::mem::size_of::<GpuMaterialData>()
        );

        index
    }

    pub fn submesh_blocks(&self, model: ModelHandle) -> impl Iterator<Item = &MeshBlock> {
        self.models[model].submeshes.iter().map(|submesh| &self.mesh_blocks[submesh.mesh_handle])
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_buffer(&self.vertex_buffer);
        context.destroy_buffer(&self.index_buffer);
        context.destroy_buffer(&self.mesh_info_buffer);
        context.destroy_buffer(&self.material_buffer);
        for (_, image) in self.textures.iter() {
            context.destroy_image(&image);
        }
    }
}

struct SepVertexIter<'a> {
    positions: std::slice::ChunksExact<'a, f32>,
    uvs: std::slice::ChunksExact<'a, f32>,
    normals: std::slice::ChunksExact<'a, f32>,
    tangents: std::slice::ChunksExact<'a, f32>,
}

impl<'a> SepVertexIter<'a> {
    fn new(
        positions: &'a [f32],
        uvs: &'a [f32],
        normals: &'a [f32],
        tangents: &'a [f32],
    ) -> Self {
        Self {
            positions: positions.chunks_exact(3),
            uvs: uvs.chunks_exact(2),
            normals: normals.chunks_exact(3),
            tangents: tangents.chunks_exact(4),
        }
    }

    fn next_pos(&mut self) -> Option<Vec3> {
        self.positions.next().map(|slice| Vec3::from_array(slice.try_into().unwrap()))
    }

    fn next_uv(&mut self) -> Option<Vec2> {
        self.uvs.next().map(|slice| Vec2::from_array(slice.try_into().unwrap()))
    }
    
    fn next_normal(&mut self) -> Option<Vec3> {
        self.normals.next().map(|slice| Vec3::from_array(slice.try_into().unwrap()))
    }

    fn next_tangent(&mut self) -> Option<Vec4> {
        self.tangents.next().map(|slice| Vec4::from_array(slice.try_into().unwrap()))
    }
}

impl Iterator for SepVertexIter<'_> {
    type Item = GpuMeshVertex;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.next_pos()?;
        let uv = self.next_uv().unwrap_or_default();
        let norm = self.next_normal().unwrap_or_default();
        let tan = self.next_tangent().unwrap_or_default();

        Some(GpuMeshVertex {
            position: pos,
            _padding0: 0,
            uv_coord: uv,
            _padding1: [0; 2],
            normal: norm,
            _padding2: 0,
            tangent: tan,
        })
    }
}

pub struct MeshData {
    pub vertices: Vec<GpuMeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn compute_normals(&mut self) {
        compute_normals(&mut self.vertices, &self.indices);
    }

    pub fn load_obj(path: &str) -> Result<Self, tobj::LoadError> {
        let (mut models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
        if models.len() > 1 {
            log::warn!("{path}: multiple models found, loading only the first");
        }
        let mut mesh_data = Self::new();

        if let Some(model) = models.first_mut() {
            let name = &model.name;
            let mesh = &mut model.mesh;

            let have_uvs = !mesh.texcoords.is_empty();
            let have_normals = !mesh.normals.is_empty();

            if !have_uvs {
                log::warn!("model '{name}' has no uv coordinates");
            }

            if !have_normals {
                log::warn!("model '{name}' has no normals");
            }

            std::mem::swap(&mut mesh_data.indices, &mut mesh.indices);
            mesh_data.vertices = SepVertexIter::new(&mesh.positions, &mesh.texcoords, &mesh.normals, &[]).collect();
            
            if !have_normals {
                log::info!("computing normals for '{name}'...");
                mesh_data.compute_normals();
            }
        }

        Ok(mesh_data)
    }
}

/// Calculates the normal vectors for each vertex.
///
/// The normals have to be zero before calling.
pub fn compute_normals(vertices: &mut [GpuMeshVertex], indices: &[u32]) {
    for chunk in indices.chunks_exact(3) {
        let v0 = vertices[chunk[0] as usize].position;
        let v1 = vertices[chunk[1] as usize].position;
        let v2 = vertices[chunk[2] as usize].position;

        let edge_a = v1 - v0;
        let edge_b = v2 - v0;

        let face_normal = Vec3::cross(edge_a, edge_b);

        vertices[chunk[0] as usize].normal += face_normal;
        vertices[chunk[1] as usize].normal += face_normal;
        vertices[chunk[2] as usize].normal += face_normal;
    }

    for vertex in vertices {
        vertex.normal = vertex.normal.normalize();
    }
}

pub fn sep_vertex_merge(
    positions: impl IntoIterator<Item = [f32; 3]>,
    uvs: impl IntoIterator<Item = [f32; 2]>,
    normals: impl IntoIterator<Item = [f32; 3]>,
    tangents: impl IntoIterator<Item = [f32; 4]>,
) -> impl Iterator<Item = GpuMeshVertex> {
    let mut positions = positions.into_iter();
    let mut uv_coords = uvs.into_iter();
    let mut normals = normals.into_iter();
    let mut tangents = tangents.into_iter();
    
    std::iter::from_fn(move || {
        let position = positions.next()?;
        let uv_coord = uv_coords.next().unwrap_or_default();
        let normal = normals.next().unwrap_or_default();
        let tangent = tangents.next().unwrap_or_default();

        Some(GpuMeshVertex::from_arrays(position, uv_coord, normal, tangent))
    })
}