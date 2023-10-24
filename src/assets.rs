use crate::collections::freelist_alloc::*;
use crate::math;
use crate::{collections::arena, graphics};

use ash::vk;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4};
use gpu_allocator::MemoryLocation;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshVertex {
    pub position: [f32; 3],
    pub packed_normals: [i8; 4],
    pub uv_coord: [f32; 2],
}

impl GpuMeshVertex {
    pub fn new(position: Vec3, uv_coord: Vec2, normal: Vec3, tangent: Vec4) -> Self {
        Self {
            position: position.to_array(),
            packed_normals: math::pack_normal_tangent_bitangent(normal, tangent),
            uv_coord: uv_coord.to_array(),
        }
    }

    pub fn from_arrays(position: [f32; 3], uv_coord: [f32; 2], normal: [f32; 3], tangent: [f32; 4]) -> Self {
        Self {
            position,
            packed_normals: math::pack_normal_tangent_bitangent(Vec3::from_array(normal), Vec4::from_array(tangent)),
            uv_coord,
        }
    }

    pub fn position_a(&self) -> Vec3A {
        Vec3A::from_array(self.position)
    }

    pub fn uv(&self) -> Vec2 {
        Vec2::from_array(self.uv_coord)
    }

    pub fn pack_normals(&mut self, normal: Vec3, tangent: Vec4) {
        self.packed_normals = math::pack_normal_tangent_bitangent(normal, tangent);
    }

    pub fn unpack_normals(&self) -> (Vec3, Vec4) {
        math::unpack_normal_tangent_bitangent(self.packed_normals)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshInfo {
    index_offset: u32,
    index_count: u32,
    vertex_offset: u32,
    aabb: GpuAabb,
    bounding_sphere: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct MeshInfo {
    pub vertex_range: BlockRange,
    pub index_range: BlockRange,
    pub vertex_alloc_index: arena::Index,
    pub index_alloc_index: arena::Index,
    pub aabb: Aabb,
    pub bounding_sphere: Vec4,
}

impl MeshInfo {
    pub fn vertex_offset(&self) -> i32 {
        self.vertex_range.start as i32
    }

    pub fn index_offset(&self) -> u32 {
        self.index_range.start as u32
    }

    pub fn index_count(&self) -> u32 {
        self.index_range.size() as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Submesh {
    pub mesh_handle: MeshHandle,
    pub material_index: MaterialHandle,
}

pub struct ModelData {
    pub submeshes: Vec<Submesh>,
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
pub type ModelHandle = arena::Index;
pub type TextureHandle = arena::Index;
pub type MaterialHandle = arena::Index;

const MAX_VERTEX_COUNT: usize = 4_000_000;
const MAX_INDEX_COUNT: usize = 12_000_000;
const MAX_MATERIAL_COUNT: usize = 1_000;

#[derive(Clone, Copy)]
pub struct AssetGraphData {
    pub vertex_buffer: graphics::GraphBufferHandle,
    pub index_buffer: graphics::GraphBufferHandle,
    pub mesh_info_buffer: graphics::GraphBufferHandle,
    pub materials_buffer: graphics::GraphBufferHandle,
}

pub struct GpuAssetStore {
    pub vertex_buffer: graphics::Buffer,
    pub index_buffer: graphics::Buffer,

    // INDICES!!! not bytes
    vertex_allocator: FreeListAllocator,
    index_allocator: FreeListAllocator,

    pub mesh_info_buffer: graphics::Buffer,
    pub mesh_infos: arena::Arena<MeshInfo>,
    pub models: arena::Arena<ModelData>,

    pub textures: arena::Arena<graphics::Image>,

    pub material_buffer: graphics::Buffer,
    pub material_indices: arena::Arena<MaterialData>,
}

impl GpuAssetStore {
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
                size: MAX_INDEX_COUNT * std::mem::size_of::<u32>(),
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let mesh_info_buffer = context.create_buffer(
            "mesh_info_buffer",
            &graphics::BufferDesc {
                size: MAX_INDEX_COUNT * std::mem::size_of::<GpuMeshInfo>(),
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
            index_buffer,

            vertex_allocator: FreeListAllocator::new(MAX_VERTEX_COUNT),
            index_allocator: FreeListAllocator::new(MAX_INDEX_COUNT),

            mesh_info_buffer,
            mesh_infos: arena::Arena::new(),
            models: arena::Arena::new(),

            textures: arena::Arena::new(),

            material_buffer,
            material_indices: arena::Arena::new(),
        }
    }

    pub fn add_mesh(&mut self, context: &graphics::Context, mesh: &MeshData) -> MeshHandle {
        let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);

        let (vertex_alloc_index, vertex_range) = self.vertex_allocator.allocate(mesh.vertices.len()).unwrap();
        let (index_alloc_index, index_range) = self.index_allocator.allocate(mesh.indices.len()).unwrap();

        context.immediate_write_buffer(
            &self.vertex_buffer,
            vertex_bytes,
            vertex_range.start * std::mem::size_of::<GpuMeshVertex>(),
        );
        context.immediate_write_buffer(
            &self.index_buffer,
            index_bytes,
            index_range.start * std::mem::size_of::<u32>(),
        );

        let mesh_block = MeshInfo {
            vertex_range,
            index_range,
            vertex_alloc_index,
            index_alloc_index,
            aabb: mesh.aabb,
            bounding_sphere: mesh.bounding_sphere,
        };
        let mesh_index = self.mesh_infos.insert(mesh_block);

        let mesh_info = GpuMeshInfo {
            index_offset: index_range.start as u32,
            index_count: index_range.size() as u32,
            vertex_offset: vertex_range.start as u32,
            aabb: mesh.aabb.into(),
            bounding_sphere: mesh.bounding_sphere.to_array(),
        };
        context.immediate_write_buffer(
            &self.mesh_info_buffer,
            bytemuck::bytes_of(&mesh_info),
            std::mem::size_of::<GpuMeshInfo>() * mesh_index.slot() as usize,
        );

        mesh_index
    }

    pub fn add_model(&mut self, submeshes: &[Submesh]) -> ModelHandle {
        self.models.insert(ModelData {
            submeshes: submeshes.to_vec(),
        })
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

        context.immediate_write_buffer(
            &self.material_buffer,
            bytemuck::bytes_of(&gpu_data),
            index.slot_index() * std::mem::size_of::<GpuMaterialData>(),
        );

        index
    }

    pub fn submesh_blocks(&self, model: ModelHandle) -> impl Iterator<Item = &MeshInfo> {
        self.models[model].submeshes.iter().map(|submesh| &self.mesh_infos[submesh.mesh_handle])
    }

    pub fn import_to_graph(&self, context: &mut graphics::Context) -> AssetGraphData {
        AssetGraphData {
            vertex_buffer: context.import(&self.vertex_buffer),
            index_buffer: context.import(&self.index_buffer),
            mesh_info_buffer: context.import(&self.mesh_info_buffer),
            materials_buffer: context.import(&self.material_buffer),
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
    fn new(positions: &'a [f32], uvs: &'a [f32], normals: &'a [f32], tangents: &'a [f32]) -> Self {
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
        let position = self.next_pos()?;
        let uv_coord = self.next_uv().unwrap_or_default();
        let normal = self.next_normal().unwrap_or_default();
        let tangent = self.next_tangent().unwrap_or_default();

        Some(GpuMeshVertex::new(position, uv_coord, normal, tangent))
    }
}

pub struct MeshData {
    pub vertices: Vec<GpuMeshVertex>,
    pub indices: Vec<u32>,
    pub aabb: Aabb,
    pub bounding_sphere: Vec4,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            aabb: Aabb::default(),
            bounding_sphere: Vec4::ZERO,
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn compute_normals(&mut self) {
        compute_normals(&mut self.vertices, &self.indices);
    }

    pub fn compute_tangents(&mut self) {
        compute_tangents(&mut self.vertices, &self.indices);
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

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    pub const ZERO: Self = Self {
        min: Vec3A::ZERO,
        max: Vec3A::ZERO,
    };

    pub fn from_arrays(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            min: Vec3A::from_array(min),
            max: Vec3A::from_array(max),
        }
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::ZERO
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuAabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl From<Aabb> for GpuAabb {
    fn from(value: Aabb) -> Self {
        Self {
            min: value.min.to_array(),
            max: value.max.to_array(),
        }
    }
}

/// Calculates the normal vectors for each vertex.
///
/// The normals have to be zero before calling.
pub fn compute_normals(vertices: &mut [GpuMeshVertex], indices: &[u32]) {
    let mut normals = vec![Vec3A::ZERO; vertices.len()];

    for triangle in indices.chunks_exact(3) {
        let v0 = Vec3A::from(vertices[triangle[0] as usize].position);
        let v1 = Vec3A::from(vertices[triangle[1] as usize].position);
        let v2 = Vec3A::from(vertices[triangle[2] as usize].position);

        let edge_a = v1 - v0;
        let edge_b = v2 - v0;

        let face_normal = edge_a.cross(edge_b);

        normals[triangle[0] as usize] += face_normal;
        normals[triangle[1] as usize] += face_normal;
        normals[triangle[2] as usize] += face_normal;
    }

    for (vertex_index, vertex) in vertices.iter_mut().enumerate() {
        let (_, tangent) = vertex.unpack_normals();
        vertex.pack_normals(normals[vertex_index].normalize().into(), tangent);
    }
}

/// Calculates the tangents for each vertex.
///
/// Every vertex must have valid normals and uv coordinates
fn compute_tangents(vertices: &mut [GpuMeshVertex], indices: &[u32]) {
    let mut tan_vec = Vec::new();
    tan_vec.resize(vertices.len() * 2, Vec3::ZERO);
    let (tan1, tan2) = tan_vec.split_at_mut(vertices.len());

    for triangle in indices.chunks_exact(3) {
        let i1 = triangle[0] as usize;
        let i2 = triangle[1] as usize;
        let i3 = triangle[2] as usize;
        let v1 = vertices[i1].position_a();
        let v2 = vertices[i2].position_a();
        let v3 = vertices[i3].position_a();
        let w1 = vertices[i1].uv();
        let w2 = vertices[i2].uv();
        let w3 = vertices[i3].uv();
        let x1 = v2.x - v1.x;
        let x2 = v3.x - v1.x;
        let y1 = v2.y - v1.y;
        let y2 = v3.y - v1.y;
        let z1 = v2.z - v1.z;
        let z2 = v3.z - v1.z;
        let s1 = w2.x - w1.x;
        let s2 = w3.x - w1.x;
        let t1 = w2.y - w1.y;
        let t2 = w3.y - w1.y;
        let r = 1.0 / (s1 * t2 - s2 * t1);

        let sdir = vec3(
            (t2 * x1 - t1 * x2) * r,
            (t2 * y1 - t1 * y2) * r,
            (t2 * z1 - t1 * z2) * r,
        );
        let tdir = vec3(
            (s1 * x2 - s2 * x1) * r,
            (s1 * y2 - s2 * y1) * r,
            (s1 * z2 - s2 * z1) * r,
        );
        tan1[i1] += sdir;
        tan1[i2] += sdir;
        tan1[i3] += sdir;
        tan2[i1] += tdir;
        tan2[i2] += tdir;
        tan2[i3] += tdir;
    }

    for i in 0..vertices.len() {
        let (normal, _) = vertices[i].unpack_normals();
        let t = tan1[i];
        // Gram-Schmidt orthogonalize
        let tangent = (t - normal * normal.dot(t)).normalize();
        // Calculate handedness
        let bitangent = if normal.cross(t).dot(tan2[i]).is_sign_negative() {
            -1.0
        } else {
            1.0
        };

        vertices[i].pack_normals(normal, tangent.extend(bitangent));
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
