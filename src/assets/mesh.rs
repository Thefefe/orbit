#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4};

use crate::math::{self, Aabb};

use super::{GpuMeshlet, MaterialHandle};

pub const MAX_MESHLET_VERTICES: usize = 64;
pub const MAX_MESHLET_TRIANGLES: usize = 64;
pub const MESHLET_CONE_WEIGHT: f32 = 0.0;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
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

#[derive(Debug, Clone, Copy)]
pub struct SubmeshData {
    pub vertex_offset: usize,
    pub vertex_count: usize,
    pub index_offset: usize,
    pub index_count: usize,
    pub material: MaterialHandle,
}

impl SubmeshData {
    pub fn vertex_range(&self) -> std::ops::Range<usize> {
        self.vertex_offset..self.vertex_offset + self.vertex_count
    }

    pub fn index_range(&self) -> std::ops::Range<usize> {
        self.index_offset..self.index_offset + self.index_count
    }
}

#[derive(Debug, Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<GpuMeshVertex>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<SubmeshData>,
    pub aabb: Aabb,
    pub bounding_sphere: Vec4,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            submeshes: Vec::new(),
            aabb: Aabb::default(),
            bounding_sphere: Vec4::ZERO,
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.submeshes.clear();
    }

    pub fn add_submesh(&mut self, vertices: &[GpuMeshVertex], indices: &[u32], material: MaterialHandle) -> usize {
        let submesh_index = self.submeshes.len();

        let vertex_offset = self.vertices.len();
        let index_offset = self.indices.len();
        self.vertices.extend_from_slice(vertices);
        self.indices.extend_from_slice(indices);

        self.submeshes.push(SubmeshData {
            vertex_offset,
            vertex_count: vertices.len(),
            index_offset,
            index_count: indices.len(),
            material,
        });

        submesh_index
    }

    pub fn compute_bounds(&mut self) {
        if self.vertices.is_empty() {
            return;
        }

        self.aabb = Aabb {
            min: Vec3A::from_array(self.vertices[0].position),
            max: Vec3A::from_array(self.vertices[0].position),
        };

        for vertex in self.vertices.iter() {
            let pos = Vec3A::from_array(vertex.position);
            self.aabb.min = self.aabb.min.min(pos);
            self.aabb.max = self.aabb.max.max(pos);
        }

        let sphere_center = (self.aabb.max + self.aabb.min) * 0.5;
        let mut sqr_radius = 0.0f32;
        for vertex in self.vertices.iter() {
            let pos = Vec3A::from_array(vertex.position);
            sqr_radius = sqr_radius.max(pos.distance_squared(sphere_center));
        }
        self.bounding_sphere = sphere_center.extend(sqr_radius.sqrt());
    }

    // pub fn load_obj(path: &str) -> Result<Self, tobj::LoadError> {
    //     let (mut models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
    //     if models.len() > 1 {
    //         log::warn!("{path}: multiple models found, loading only the first");
    //     }
    //     let mut mesh_data = Self::new();

    //     if let Some(model) = models.first_mut() {
    //         let name = &model.name;
    //         let mesh = &mut model.mesh;

    //         let have_uvs = !mesh.texcoords.is_empty();
    //         let have_normals = !mesh.normals.is_empty();

    //         if !have_uvs {
    //             log::warn!("model '{name}' has no uv coordinates");
    //         }

    //         if !have_normals {
    //             log::warn!("model '{name}' has no normals");
    //         }

    //         std::mem::swap(&mut mesh_data.indices, &mut mesh.indices);
    //         mesh_data.vertices = SepVertexIter::new(&mesh.positions, &mesh.texcoords, &mesh.normals, &[]).collect();

    //         if !have_normals {
    //             log::info!("computing normals for '{name}'...");
    //             mesh_data.compute_normals();
    //         }
    //     }

    //     Ok(mesh_data)
    // }
}

const MESH_LOD_TARGET_ERROR: f32 = 1e-2;

pub fn build_mesh_lod(
    vertices: &[GpuMeshVertex],
    indices: &[u32],
    target_index_count: usize,
    lock_border: bool,
    output_indices: &mut Vec<u32>,
) -> f32 {
    let output_indices_offset = output_indices.len();
    output_indices.reserve(output_indices.len() + indices.len());
    let mut error = 0.0;
    unsafe {
        let index_count = meshopt::ffi::meshopt_simplify(
            output_indices.as_mut_ptr().add(output_indices_offset),
            indices.as_ptr(),
            indices.len(),
            vertices.as_ptr().cast(),
            vertices.len(),
            std::mem::size_of::<GpuMeshVertex>(),
            target_index_count,
            MESH_LOD_TARGET_ERROR,
            if lock_border {
                meshopt::SimplifyOptions::LockBorder.bits()
            } else {
                0
            },
            &mut error as *mut f32,
        );

        output_indices.set_len(output_indices_offset + index_count);
        let result_index_slice = &mut output_indices[output_indices_offset..output_indices_offset + index_count];

        meshopt::optimize_vertex_cache_in_place(result_index_slice, vertices.len());
        meshopt::optimize_overdraw_in_place(result_index_slice, &vertex_adapter(vertices), 1.05);
    };

    error
}

pub fn compute_meshlets(
    vertices: &[GpuMeshVertex],
    indices: &[u32],
    material: MaterialHandle,
    vertex_offset: u32,

    meshlet_data: &mut Vec<u32>,
    meshlets: &mut Vec<GpuMeshlet>,
) {
    let raw_meshlets = meshopt::build_meshlets(
        indices,
        &vertex_adapter(&vertices),
        MAX_MESHLET_VERTICES,
        MAX_MESHLET_TRIANGLES,
        MESHLET_CONE_WEIGHT,
    );
    
    for meshlet in raw_meshlets.iter() {
        let data_offset = meshlet_data.len();
        meshlet_data.extend_from_slice(meshlet.vertices);

        let triangle_offset = meshlet_data.len() * 4;
        meshlet_data.resize(meshlet_data.len() + meshlet.triangles.len().div_ceil(4), 0);
        let triangle_slice: &mut [u8] = &mut bytemuck::cast_slice_mut::<_, u8>(meshlet_data)
            [triangle_offset..triangle_offset + meshlet.triangles.len()];
        triangle_slice.clone_from_slice(&meshlet.triangles);

        let triangle_count: u8 = (meshlet.triangles.len() / 3).try_into().unwrap();

        let meshlet_bounds = meshopt::compute_meshlet_bounds(meshlet, &vertex_adapter(vertices));
        meshlets.push(GpuMeshlet {
            bounding_sphere: Vec4::from_array([
                meshlet_bounds.center[0],
                meshlet_bounds.center[1],
                meshlet_bounds.center[2],
                meshlet_bounds.radius,
            ]),
            cone_axis: meshlet_bounds.cone_axis_s8,
            cone_cutoff: meshlet_bounds.cone_cutoff_s8,
            vertex_offset,
            data_offset: data_offset.try_into().unwrap(),
            material_index: material.slot() as u16,
            vertex_count: meshlet.vertices.len() as u8,
            triangle_count,
        });
    }
}

pub fn optimize_mesh(
    input_vertices: &[GpuMeshVertex],
    input_indices: &[u32],
    remap_buffer: &mut Vec<u32>,
    output_vertices: &mut Vec<GpuMeshVertex>,
    output_indices: &mut Vec<u32>,
) {
    const VERTEX_SIZE: usize = std::mem::size_of::<GpuMeshVertex>();

    remap_buffer.clear();
    output_vertices.clear();
    output_indices.clear();

    remap_buffer.reserve(input_vertices.len());

    let vertex_count = unsafe {
        remap_buffer.set_len(input_vertices.len());
        meshopt::ffi::meshopt_generateVertexRemap(
            remap_buffer.as_mut_ptr(),
            input_indices.as_ptr(),
            input_indices.len(),
            input_vertices.as_ptr().cast(),
            input_vertices.len(),
            VERTEX_SIZE,
        )
    };

    output_vertices.reserve(vertex_count);
    output_indices.reserve(input_indices.len());

    unsafe {
        output_vertices.set_len(vertex_count);
        output_indices.set_len(input_indices.len());

        meshopt::ffi::meshopt_remapVertexBuffer(
            output_vertices.as_mut_ptr().cast(),
            input_vertices.as_ptr().cast(),
            input_vertices.len(),
            VERTEX_SIZE,
            remap_buffer.as_ptr(),
        );

        meshopt::ffi::meshopt_remapIndexBuffer(
            output_indices.as_mut_ptr(),
            input_indices.as_ptr(),
            input_indices.len(),
            remap_buffer.as_ptr(),
        );
    };

    meshopt::optimize_vertex_cache_in_place(output_indices, vertex_count);
    meshopt::optimize_overdraw_in_place(output_indices, &vertex_adapter(output_vertices), 1.05);
    meshopt::optimize_vertex_fetch_in_place(output_indices, output_vertices);
}

pub fn vertex_adapter(vertices: &[GpuMeshVertex]) -> meshopt::VertexDataAdapter {
    let position_offset = bytemuck::offset_of!(GpuMeshVertex, position);
    let vertex_stride = std::mem::size_of::<GpuMeshVertex>();
    meshopt::VertexDataAdapter::new(bytemuck::cast_slice(vertices), vertex_stride, position_offset).unwrap()
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuAabb {
    min: Vec4,
    max: Vec4,
}

impl From<Aabb> for GpuAabb {
    fn from(value: Aabb) -> Self {
        Self {
            min: value.min.extend(0.0),
            max: value.max.extend(0.0),
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
pub fn compute_tangents(vertices: &mut [GpuMeshVertex], indices: &[u32]) {
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
