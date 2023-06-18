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

#[derive(Debug, Clone, Copy)]
pub struct MeshBlock {
    pub vertex_range: BlockRange,
    pub index_range: BlockRange,
    pub vertex_alloc_index: arena::Index,
    pub index_alloc_index: arena::Index,
}

impl MeshBlock {
}

struct ModelData {
    submesh_indices: Vec<arena::Index>,
}

pub type MeshHandle = arena::Index;
pub type ModelHandle = arena::Index;

pub struct GpuAssetStore {
    pub vertex_buffer: render::Buffer,
    pub index_buffer: render::Buffer,

    // COUNTS!!! not bytes
    vertex_allocator: FreeListAllocator,
    index_allocator: FreeListAllocator,

    mesh_blocks: arena::Arena<MeshBlock>,
    models: arena::Arena<ModelData>,
}

const MAX_VERTEX_COUNT: usize = 1_000_000;
const MAX_INDEX_COUNT: usize = 1_000_000;

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

        Self {
            vertex_buffer,
            index_buffer,
            
            vertex_allocator: FreeListAllocator::new(MAX_VERTEX_COUNT),
            index_allocator: FreeListAllocator::new(MAX_INDEX_COUNT),

            mesh_blocks: arena::Arena::new(),
            models: arena::Arena::new(),
        }
    }

    pub fn add_mesh(&mut self, context: &render::Context, mesh: &MeshData) -> MeshHandle {
        let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);

        let (vertex_alloc_index, vertex_range) = self.vertex_allocator.allocate(mesh.vertices.len()).unwrap();
        let (index_alloc_index, index_range) = self.index_allocator.allocate(mesh.indices.len()).unwrap();

        context.immediate_write_buffer(&self.vertex_buffer, vertex_bytes, vertex_range.start * std::mem::size_of::<GpuMeshVertex>());
        context.immediate_write_buffer(&self.index_buffer, index_bytes, index_range.start * std::mem::size_of::<u32>());

        let mesh_index = self.mesh_blocks.insert(MeshBlock {
            vertex_range,
            index_range,
            vertex_alloc_index,
            index_alloc_index,
        });

        mesh_index
    }

    pub fn add_model(&mut self, submeshes: &[arena::Index]) -> ModelHandle {
        self.models.insert(ModelData {
            submesh_indices: submeshes.to_vec(),
        })
    }

    pub fn submeshes(&self, model: ModelHandle) -> impl Iterator<Item = &MeshBlock> {
        self.models[model].submesh_indices.iter().map(|submesh| &self.mesh_blocks[*submesh])
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_buffer(&self.vertex_buffer);
        context.destroy_buffer(&self.index_buffer);
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