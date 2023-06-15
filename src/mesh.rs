use std::ops::Range;

use crate::render;

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
pub struct MeshVertex {
    position: Vec3,
    _padding0: u32,
    uv_coord: Vec2,
    _padding1: [u32; 2],
    normal: Vec3,
    _padding2: u32,
    tangent: Vec4,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshIndex {
    pub index_offset: usize,
    pub index_count: usize,
    pub vertex_offset: usize,
}

impl MeshIndex {
    pub fn index_range(&self) -> Range<u32> {
        self.index_offset as u32..(self.index_offset + self.index_count) as u32
    }   
}

pub struct MeshBuffer {
    pub vertex_buffer: render::Buffer,
    pub index_buffer: render::Buffer,
    // cursors in element count *NOT BYTES*
    vertex_cursor_count: usize,
    index_cursor_count: usize,
    mesh_indices: Vec<MeshIndex>,
}

impl MeshBuffer {
    pub fn new(context: &render::Context, vertex_count: usize, index_count: usize) -> Self {
        let vertex_buffer = context.create_buffer("mesh_vertex_buffer", &render::BufferDesc {
            size: vertex_count * std::mem::size_of::<MeshVertex>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        let index_buffer = context.create_buffer("mesh_index_buffer", &render::BufferDesc {
            size: index_count * std::mem::size_of::<u32>(),
            usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        Self {
            vertex_buffer,
            vertex_cursor_count: 0,
            index_buffer,
            index_cursor_count: 0,
            mesh_indices: Vec::new(),
        }
    }

    fn vertex_cursor_byte(&self) -> usize {
        self.vertex_cursor_count * std::mem::size_of::<MeshVertex>()
    }

    fn index_cursor_byte(&self) -> usize {
        self.index_cursor_count * std::mem::size_of::<u32>()
    }

    fn remaining_vertex_bytes(&self) -> usize {
        self.vertex_buffer.size as usize - self.vertex_cursor_byte()
    }

    fn remaining_index_bytes(&self) -> usize {
        self.index_buffer.size as usize - self.index_cursor_byte()
    }

    pub fn add_mesh(&mut self, context: &render::Context, mesh: &MeshData) -> MeshIndex {
        let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);

        assert!(vertex_bytes.len() <= self.remaining_vertex_bytes());
        assert!(index_bytes.len() <= self.remaining_index_bytes());
        context.immediate_write_buffer(&self.vertex_buffer, vertex_bytes, self.vertex_cursor_byte());
        context.immediate_write_buffer(&self.index_buffer, index_bytes, self.index_cursor_byte());

        let mesh_index = MeshIndex {
            index_offset: self.index_cursor_count,
            index_count: mesh.indices.len(),
            vertex_offset: self.vertex_cursor_count,
        };
        self.mesh_indices.push(mesh_index);

        self.vertex_cursor_count += mesh.vertices.len();
        self.index_cursor_count += mesh.indices.len();

        mesh_index
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
    type Item = MeshVertex;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.next_pos()?;
        let uv = self.next_uv().unwrap_or_default();
        let norm = self.next_normal().unwrap_or_default();
        let tan = self.next_tangent().unwrap_or_default();

        Some(MeshVertex {
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
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
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
pub fn compute_normals(vertices: &mut [MeshVertex], indices: &[u32]) {
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