use crate::collections::freelist_alloc::*;
use crate::math::Aabb;
use crate::{collections::arena, graphics};

pub mod mesh;
use mesh::*;

use ash::vk;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4};
use gpu_allocator::MemoryLocation;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshInfo {
    bounding_sphere: Vec4,
    aabb: GpuAabb,
    vertex_offset: u32,
    meshlet_data_offset: u32,
    meshlet_offset: u32,
    meshlet_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshInfo {
    pub vertex_range: BlockRange,
    pub meshlet_data_range: BlockRange,
    pub meshlet_range: BlockRange,

    pub vertex_alloc_index: arena::Index,
    pub meshlet_data_alloc_index: arena::Index,
    pub meshlet_alloc_index: arena::Index,

    pub aabb: Aabb,
    pub bounding_sphere: Vec4,
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
        GpuMeshInfo {
            bounding_sphere: self.bounding_sphere,
            aabb: GpuAabb::from(self.aabb),
            vertex_offset: self.vertex_offset(),
            meshlet_data_offset: self.meshlet_data_offset(),
            meshlet_offset: self.meshlet_offset(),
            meshlet_count: self.meshlet_count(),
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
const MAX_MATERIAL_COUNT: usize = 1_000;
const MAX_MESHLET_COUNT: usize = 64_000;
// 64 (meshlet vertex index), 24 (approx. micro index)
const MAX_MESHLET_DATA_COUNT: usize = MAX_MESHLET_COUNT * (64 + 24);

pub const MAX_MESHLET_VERTICES: usize = 64;
pub const MAX_MESHLET_TRIANGLES: usize = 64;
pub const MESHLET_CONE_WEIGHT: f32 = 0.5;

#[derive(Clone, Copy)]
pub struct AssetGraphData {
    pub vertex_buffer: graphics::GraphBufferHandle,
    pub meshlet_data_buffer: graphics::GraphBufferHandle,
    pub meshlet_buffer: graphics::GraphBufferHandle,
    pub mesh_info_buffer: graphics::GraphBufferHandle,
    pub materials_buffer: graphics::GraphBufferHandle,
}

pub struct GpuAssets {
    pub vertex_buffer: graphics::Buffer,
    pub meshlet_data_buffer: graphics::Buffer,
    pub meshlet_buffer: graphics::Buffer,

    // indices not bytes
    vertex_allocator: FreeListAllocator,
    meshlet_data_allocator: FreeListAllocator,
    meshlet_allocator: FreeListAllocator,

    pub mesh_info_buffer: graphics::Buffer,
    pub mesh_infos: arena::Arena<MeshInfo>,

    pub textures: arena::Arena<graphics::Image>,

    pub material_buffer: graphics::Buffer,
    pub material_indices: arena::Arena<MaterialData>,
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

        Self {
            vertex_buffer,
            meshlet_data_buffer,
            meshlet_buffer,

            vertex_allocator: FreeListAllocator::new(MAX_VERTEX_COUNT),
            meshlet_data_allocator: FreeListAllocator::new(MAX_MESHLET_DATA_COUNT),
            meshlet_allocator: FreeListAllocator::new(MAX_MESHLET_COUNT),

            mesh_info_buffer,
            mesh_infos: arena::Arena::new(),

            textures: arena::Arena::new(),

            material_buffer,
            material_indices: arena::Arena::new(),
        }
    }

    pub fn add_mesh(&mut self, context: &graphics::Context, mesh: &MeshData) -> MeshHandle {
        let mut meshlet_data = Vec::new();
        let mut meshlets = Vec::new();
        for (vertices, indices, vertex_offset, material) in mesh.submeshes() {
            compute_meshlets(vertices, indices, material, vertex_offset, &mut meshlet_data, &mut meshlets);
        }

        let (vertex_alloc_index, vertex_range) = self.vertex_allocator.allocate(mesh.vertices.len()).unwrap();
        let (meshlet_data_alloc_index, meshlet_data_range) =
            self.meshlet_data_allocator.allocate(meshlet_data.len()).unwrap();
        let (meshlet_alloc_index, meshlet_range) = self.meshlet_allocator.allocate(meshlets.len()).unwrap();

        for meshlet in meshlets.iter_mut() {
            meshlet.vertex_offset += vertex_range.start as u32;
            meshlet.data_offset += meshlet_data_range.start as u32;
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let meshlet_data_bytes: &[u8] = bytemuck::cast_slice(&meshlet_data);
        let meshlet_bytes: &[u8] = bytemuck::cast_slice(&meshlets);

        context.queue_write_buffer(
            &self.vertex_buffer,
            vertex_range.start * std::mem::size_of::<GpuMeshVertex>(),
            vertex_bytes,
        );
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
            meshlet_data_range,
            meshlet_range,

            vertex_alloc_index,
            meshlet_data_alloc_index,
            meshlet_alloc_index,

            aabb: mesh.aabb,
            bounding_sphere: mesh.bounding_sphere,
        };
        let mesh_index = self.mesh_infos.insert(mesh_info);

        let gpu_mesh_info = mesh_info.to_gpu();
        context.queue_write_buffer(
            &self.mesh_info_buffer,
            std::mem::size_of::<GpuMeshInfo>() * mesh_index.slot() as usize,
            bytemuck::bytes_of(&gpu_mesh_info),
        );

        mesh_index
    }

    pub fn import_texture(&mut self, image: graphics::Image) -> TextureHandle {
        assert!(image.sampled_index().is_some());
        self.textures.insert(image)
    }

    fn get_texture_desc_index(&self, handle: TextureHandle) -> u32 {
        self.textures[handle]._descriptor_index
    }

    pub fn add_material(&mut self, context: &graphics::Context, material_data: MaterialData) -> MaterialHandle {
        let index = self.material_indices.insert(material_data);

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
            meshlet_data_buffer: context.import(&self.meshlet_data_buffer),
            meshlet_buffer: context.import(&self.meshlet_buffer),
            mesh_info_buffer: context.import(&self.mesh_info_buffer),
            materials_buffer: context.import(&self.material_buffer),
        }
    }
}
