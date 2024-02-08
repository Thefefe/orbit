use crate::collections::freelist_alloc::*;
use crate::math::Aabb;
use crate::{collections::arena, graphics};

pub mod mesh;
use bytemuck::Zeroable;
use mesh::*;

use ash::vk;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4};
use gpu_allocator::MemoryLocation;
use parking_lot::{RwLock, RwLockReadGuard};
use smallvec::SmallVec;

pub const MAX_MESH_LODS: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshInfo {
    bounding_sphere: Vec4,
    aabb: GpuAabb,
    vertex_offset: u32,
    meshlet_data_offset: u32,
    mesh_lod_count: u32,
    _padding: u32,
    mesh_lods: [GpuMeshLod; MAX_MESH_LODS],
}

#[derive(Debug, Clone, Copy)]
pub struct SubmeshInfo {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshLod {
    pub meshlet_offset: u32,
    pub meshlet_count: u32,
}

#[derive(Debug, Clone)]
pub struct MeshInfo {
    pub vertex_range: BlockRange,
    pub index_range: BlockRange,
    pub meshlet_data_range: BlockRange,
    pub meshlet_range: BlockRange,

    pub vertex_alloc_index: arena::Index,
    pub index_alloc_index: arena::Index,
    pub meshlet_data_alloc_index: arena::Index,
    pub meshlet_alloc_index: arena::Index,
    pub mesh_lod_count: usize,
    pub mesh_lods: [GpuMeshLod; MAX_MESH_LODS],

    pub aabb: Aabb,
    pub bounding_sphere: Vec4,
    pub submesh_infos: SmallVec<[SubmeshInfo; 4]>,
}

impl MeshInfo {
    pub fn vertex_offset(&self) -> u32 {
        self.vertex_range.start as u32
    }

    pub fn meshlet_offset(&self) -> u32 {
        self.meshlet_range.start as u32
    }

    pub fn meshlet_count(&self) -> u32 {
        self.meshlet_range.size() as u32
    }

    pub fn meshlet_data_offset(&self) -> u32 {
        self.meshlet_data_range.start as u32
    }

    pub fn to_gpu(&self) -> GpuMeshInfo {
        let mut mesh_lods = self.mesh_lods;
        for lod in mesh_lods.iter_mut() {
            lod.meshlet_offset += self.meshlet_offset();
        }
        GpuMeshInfo {
            bounding_sphere: self.bounding_sphere,
            aabb: GpuAabb::from(self.aabb),
            vertex_offset: self.vertex_offset(),
            meshlet_data_offset: self.meshlet_data_offset(),
            mesh_lod_count: self.mesh_lod_count as u32,
            _padding: 0,
            mesh_lods,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshletDrawCommand {
    pub cmd_index_count: u32,
    pub cmd_instance_count: u32,
    pub cmd_first_index: u32,
    pub cmd_vertex_offset: i32,
    pub cmd_first_instance: u32,

    pub meshlet_vertex_offset: u32,
    pub meshlet_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshlet {
    bounding_sphere: Vec4,
    cone_axis: [i8; 3],
    cone_cutoff: i8,
    vertex_offset: u32,
    data_offset: u32,
    material_index: u16,
    vertex_count: u8,
    triangle_count: u8,
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque = 0,
    Masked = 1,
    Transparent = 2,
}

impl AlphaMode {
    pub fn raw_index(self) -> u32 {
        match self {
            AlphaMode::Opaque => 0,
            AlphaMode::Masked => 1,
            AlphaMode::Transparent => 2,
        }
    }
}

impl From<gltf::material::AlphaMode> for AlphaMode {
    fn from(value: gltf::material::AlphaMode) -> Self {
        match value {
            gltf::material::AlphaMode::Opaque => Self::Opaque,
            gltf::material::AlphaMode::Mask => Self::Masked,
            gltf::material::AlphaMode::Blend => Self::Transparent,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialData {
    pub alpha_mode: AlphaMode,

    pub base_color: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub occlusion_factor: f32,
    pub emissive_factor: Vec3,

    pub alpha_cutoff: f32,

    pub base_texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub occlusion_texture: Option<TextureHandle>,
    pub emissive_texture: Option<TextureHandle>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMaterialData {
    base_color: Vec4,

    emissive_factor: Vec3,
    metallic_factor: f32,
    roughness_factor: f32,
    occulusion_factor: f32,

    alpha_cutoff: f32,

    base_texture_index: u32,
    normal_texture_index: u32,
    metallic_roughness_texture_index: u32,
    occulusion_texture_index: u32,
    emissive_texture_index: u32,

    alpha_mode: u32,
    _padding: [u32; 3],
}

pub type MeshHandle = arena::Index;
pub type TextureHandle = arena::Index;
pub type MaterialHandle = arena::Index;

const MAX_MESH_COUNT: usize = 10_000;
const MAX_VERTEX_COUNT: usize = 4_000_000;
const MAX_INDEX_COUNT: usize = 12_000_000;
const MAX_MATERIAL_COUNT: usize = 1_000;
const MAX_MESHLET_COUNT: usize = MAX_MESH_COUNT * 64;
const MAX_SUBMESH_COUNT: usize = 10_000 * 8;
// 64 (meshlet vertex index), 24 (approx. micro index)
const MAX_MESHLET_DATA_COUNT: usize = MAX_MESHLET_COUNT * (64 + 24);

#[derive(Clone, Copy)]
pub struct AssetGraphData {
    pub vertex_buffer: graphics::GraphBufferHandle,
    pub index_buffer: graphics::GraphBufferHandle,
    pub meshlet_data_buffer: graphics::GraphBufferHandle,
    pub meshlet_buffer: graphics::GraphBufferHandle,
    pub mesh_info_buffer: graphics::GraphBufferHandle,
    pub materials_buffer: graphics::GraphBufferHandle,
}

pub struct AssetsSharedStuff {
    // indices not bytes
    vertex_allocator: FreeListAllocator,
    index_allocator: FreeListAllocator,
    meshlet_data_allocator: FreeListAllocator,
    meshlet_allocator: FreeListAllocator,
    submesh_allocator: FreeListAllocator,

    pub mesh_infos: arena::Arena<MeshInfo>,
    pub textures: arena::Arena<graphics::Image>,
    pub materials: arena::Arena<MaterialData>,
}

pub struct GpuAssets {
    pub vertex_buffer: graphics::Buffer,
    pub index_buffer: graphics::Buffer,
    pub meshlet_data_buffer: graphics::Buffer,
    pub meshlet_buffer: graphics::Buffer,
    pub mesh_info_buffer: graphics::Buffer,
    pub material_buffer: graphics::Buffer,

    pub shared_stuff: RwLock<AssetsSharedStuff>,
}

impl GpuAssets {
    pub fn new(context: &graphics::Context) -> Self {
        let vertex_buffer = context.create_buffer(
            "mesh_vertex_buffer",
            &graphics::BufferDesc {
                size: MAX_VERTEX_COUNT * std::mem::size_of::<GpuMeshVertex>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let index_buffer = context.create_buffer(
            "mesh_index_buffer",
            &graphics::BufferDesc {
                size: MAX_INDEX_COUNT * 4,
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let meshlet_data_buffer = context.create_buffer(
            "meshlet_data_buffer",
            &graphics::BufferDesc {
                size: MAX_MESHLET_DATA_COUNT * 4,
                usage: vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let meshlet_buffer = context.create_buffer(
            "meshlet_buffer",
            &graphics::BufferDesc {
                size: MAX_MESHLET_COUNT * std::mem::size_of::<GpuMeshlet>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let mesh_info_buffer = context.create_buffer(
            "mesh_info_buffer",
            &graphics::BufferDesc {
                size: MAX_MESH_COUNT * std::mem::size_of::<GpuMeshInfo>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let material_buffer = context.create_buffer(
            "material_buffer",
            &graphics::BufferDesc {
                size: MAX_MATERIAL_COUNT * std::mem::size_of::<GpuMaterialData>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let shared_stuff = RwLock::new(AssetsSharedStuff {
            vertex_allocator: FreeListAllocator::new(MAX_VERTEX_COUNT),
            index_allocator: FreeListAllocator::new(MAX_INDEX_COUNT),
            meshlet_data_allocator: FreeListAllocator::new(MAX_MESHLET_DATA_COUNT),
            meshlet_allocator: FreeListAllocator::new(MAX_MESHLET_COUNT),

            submesh_allocator: FreeListAllocator::new(MAX_SUBMESH_COUNT),

            mesh_infos: arena::Arena::new(),

            textures: arena::Arena::new(),
            materials: arena::Arena::new(),
        });

        Self {
            vertex_buffer,
            index_buffer,
            meshlet_data_buffer,
            meshlet_buffer,
            mesh_info_buffer,
            material_buffer,

            shared_stuff,
        }
    }

    pub fn add_mesh(&self, context: &graphics::Context, mesh: &MeshData) -> MeshHandle {
        let mut meshlet_data = Vec::new();
        let mut meshlets = Vec::new();
        let mut mesh_lod_count = 0;
        let mut mesh_lods = [GpuMeshLod::zeroed(); MAX_MESH_LODS];

        let mut submesh_infos: SmallVec<[SubmeshInfo; 4]> = SmallVec::new();

        let mut lod_indices = Vec::new();
        let mut index_count_scale: f64 = 1.0;

        for lod_index in 0..MAX_MESH_LODS {
            let meshlet_offset = meshlets.len();

            let mut finish_lod = false;

            for submesh in mesh.submeshes.iter().copied() {
                let submesh_vertices = &mesh.vertices[submesh.vertex_range()];
                let submesh_indices = &mesh.indices[submesh.index_range()];
                lod_indices.clear();

                if lod_index == 0 {
                    lod_indices.extend_from_slice(&mesh.indices[submesh.index_range()]);
                } else {
                    let target_index_count = (submesh_indices.len() as f64 * index_count_scale) as usize;
                    let _error = build_mesh_lod(
                        submesh_vertices,
                        submesh_indices,
                        target_index_count,
                        mesh.submeshes.len() > 1,
                        &mut lod_indices,
                    );

                    if target_index_count.div_ceil(3) * 3 < lod_indices.len() {
                        finish_lod = true;
                    }

                    // log::info!(
                    //     "lod_{lod_index}: source_index_count: {}, target_index_count: {target_index_count}, result_index_count: {}, result_error: {_error}",
                    //     submesh_indices.len(),
                    //     lod_indices.len(),
                    // );
                }

                if !lod_indices.is_empty() {
                    compute_meshlets(
                        submesh_vertices,
                        &lod_indices,
                        submesh.material,
                        submesh.vertex_offset as u32,
                        &mut meshlet_data,
                        &mut meshlets,
                    );
                }

                // for now submeshes are only used for wireframe drawing, first lod is enough
                if lod_index == 0 {
                    submesh_infos.push(SubmeshInfo {
                        vertex_offset: submesh.vertex_offset as u32,
                        vertex_count: submesh.vertex_count as u32,
                        index_offset: submesh.index_offset as u32,
                        index_count: submesh.index_count as u32,
                    });
                }

                index_count_scale *= 0.8;
            }

            mesh_lod_count += 1;

            let meshlet_count = meshlets.len() - meshlet_offset;
            mesh_lods[lod_index].meshlet_offset = meshlet_offset as u32;
            mesh_lods[lod_index].meshlet_count = meshlet_count as u32;

            if finish_lod {
                break;
            }
        }

        let mut shared = self.shared_stuff.write();

        let (vertex_alloc_index, vertex_range) = shared.vertex_allocator.allocate(mesh.vertices.len()).unwrap();
        let (index_alloc_index, index_range) = shared.index_allocator.allocate(mesh.indices.len()).unwrap();
        let (meshlet_data_alloc_index, meshlet_data_range) =
            shared.meshlet_data_allocator.allocate(meshlet_data.len()).unwrap();
        let (meshlet_alloc_index, meshlet_range) = shared.meshlet_allocator.allocate(meshlets.len()).unwrap();

        drop(shared);

        for meshlet in meshlets.iter_mut() {
            meshlet.vertex_offset += vertex_range.start as u32;
            meshlet.data_offset += meshlet_data_range.start as u32;
            // log::info!("{meshlet:?}");
        }

        for submesh in submesh_infos.iter_mut() {
            submesh.vertex_offset += vertex_range.start as u32;
            submesh.index_offset += index_range.start as u32;
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);
        let meshlet_data_bytes: &[u8] = bytemuck::cast_slice(&meshlet_data);
        let meshlet_bytes: &[u8] = bytemuck::cast_slice(&meshlets);

        context.queue_write_buffer(
            &self.vertex_buffer,
            vertex_range.start * std::mem::size_of::<GpuMeshVertex>(),
            vertex_bytes,
        );
        context.queue_write_buffer(&self.index_buffer, index_range.start * 4, index_bytes);
        context.queue_write_buffer(
            &self.meshlet_data_buffer,
            meshlet_data_range.start * 4,
            meshlet_data_bytes,
        );
        context.queue_write_buffer(
            &self.meshlet_buffer,
            meshlet_range.start * std::mem::size_of::<GpuMeshlet>(),
            meshlet_bytes,
        );

        let mesh_info = MeshInfo {
            vertex_range,
            index_range,
            meshlet_data_range,
            meshlet_range,

            vertex_alloc_index,
            index_alloc_index,
            meshlet_data_alloc_index,
            meshlet_alloc_index,

            aabb: mesh.aabb,
            bounding_sphere: mesh.bounding_sphere,

            submesh_infos,
            mesh_lod_count,
            mesh_lods,
        };

        let gpu_mesh_info = mesh_info.to_gpu();
        let mesh_index = self.shared_stuff.write().mesh_infos.insert(mesh_info);

        context.queue_write_buffer(
            &self.mesh_info_buffer,
            std::mem::size_of::<GpuMeshInfo>() * mesh_index.slot() as usize,
            bytemuck::bytes_of(&gpu_mesh_info),
        );

        mesh_index
    }

    pub fn import_texture(&self, image: graphics::Image) -> TextureHandle {
        assert!(image.sampled_index().is_some());
        self.shared_stuff.write().textures.insert(image)
    }

    fn get_texture_desc_index(&self, handle: TextureHandle) -> u32 {
        self.shared_stuff.read().textures[handle]._descriptor_index
    }

    pub fn add_material(&mut self, context: &graphics::Context, material_data: MaterialData) -> MaterialHandle {
        let index = self.shared_stuff.write().materials.insert(material_data);

        let base_texture_index =
            material_data.base_texture.map(|handle| self.get_texture_desc_index(handle)).unwrap_or(u32::MAX);
        let normal_texture_index =
            material_data.normal_texture.map(|handle| self.get_texture_desc_index(handle)).unwrap_or(u32::MAX);
        let metallic_roughness_texture_index = material_data
            .metallic_roughness_texture
            .map(|handle| self.get_texture_desc_index(handle))
            .unwrap_or(u32::MAX);
        let occulusion_texture_index = material_data
            .occlusion_texture
            .map(|handle| self.get_texture_desc_index(handle))
            .unwrap_or(u32::MAX);
        let emissive_texture_index =
            material_data.emissive_texture.map(|handle| self.get_texture_desc_index(handle)).unwrap_or(u32::MAX);

        let gpu_data = GpuMaterialData {
            base_color: material_data.base_color,
            emissive_factor: material_data.emissive_factor,
            metallic_factor: material_data.metallic_factor,
            roughness_factor: material_data.roughness_factor,
            occulusion_factor: material_data.occlusion_factor,

            alpha_cutoff: material_data.alpha_cutoff,

            base_texture_index,
            normal_texture_index,
            metallic_roughness_texture_index,
            occulusion_texture_index,
            emissive_texture_index,

            alpha_mode: material_data.alpha_mode as u32,
            _padding: [0; 3],
        };

        context.queue_write_buffer(
            &self.material_buffer,
            index.slot_index() * std::mem::size_of::<GpuMaterialData>(),
            bytemuck::bytes_of(&gpu_data),
        );

        index
    }

    pub fn import_to_graph(&self, context: &mut graphics::Context) -> AssetGraphData {
        AssetGraphData {
            vertex_buffer: context.import(&self.vertex_buffer),
            index_buffer: context.import(&self.index_buffer),
            meshlet_data_buffer: context.import(&self.meshlet_data_buffer),
            meshlet_buffer: context.import(&self.meshlet_buffer),
            mesh_info_buffer: context.import(&self.mesh_info_buffer),
            materials_buffer: context.import(&self.material_buffer),
        }
    }

    pub fn get_mesh_info(&self, index: arena::Index) -> parking_lot::MappedRwLockReadGuard<MeshInfo> {
        RwLockReadGuard::map(self.shared_stuff.read(), |shared| &shared.mesh_infos[index])
    }
}
