use ash::vk;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4, Quat, Mat3, Mat4};
use gpu_allocator::MemoryLocation;

use crate::{assets::{ModelHandle, GpuAssetStore}, render};

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub orientation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn from_mat4(mat: Mat4) -> Self {
        let (scale, orientation, position) = mat.to_scale_rotation_translation();
        Self { position, orientation, scale }
    } 

    pub fn translate_relative(&mut self, translation: Vec3) {
        self.position += self.orientation * translation;
    }

    pub fn compute_matrix(&self) -> glam::Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.orientation, self.position)
    }
}

pub struct EntityData {
    pub name: Option<String>,
    pub transform: Transform,
    pub model: Option<ModelHandle>,
}

impl EntityData {
    fn compute_gpu_data(&self) -> GpuEntityData {
        let model_matrix = self.transform.compute_matrix();
        let normal_matrix = Mat4::from_mat3(Mat3::from_mat4(model_matrix.inverse().transpose()));
        GpuEntityData { model_matrix, normal_matrix }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuEntityData {
    model_matrix: Mat4,
    normal_matrix: Mat4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuSubmeshData {
    entity_index: u32,
    mesh_index: u32,
    material_index: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuDrawCommand {
    // DrawIndiexedIndirectCommand
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,

    // other per-draw data
    pub material_index: u32,
    _padding: [u32; 2],
}

pub struct SceneBuffer {
    pub entities: Vec<EntityData>,
    pub entity_data_buffer: render::Buffer, 
    
    pub submesh_data: Vec<GpuSubmeshData>,
    pub submesh_buffer: render::Buffer,
}

const MAX_INSTANCE_COUNT: usize = 1_000_000;

impl SceneBuffer {
    pub fn new(context: &render::Context) -> Self {
        let entity_data_buffer = context.create_buffer("scene_instance_buffer", &render::BufferDesc {
            size: MAX_INSTANCE_COUNT * std::mem::size_of::<GpuEntityData>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        let submesh_buffer = context.create_buffer("submesh_buffer", &render::BufferDesc {
            size: MAX_INSTANCE_COUNT * std::mem::size_of::<GpuSubmeshData>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        Self {
            entities: Vec::new(),
            entity_data_buffer,

            submesh_data: Vec::new(),
            submesh_buffer,
        }
    }

    pub fn add_entity(&mut self, data: EntityData) -> usize {
        let index = self.entities.len();
        self.entities.push(data);

        index
    }

    pub fn update_instances(&self, context: &render::Context) {
        for (index, entity) in self.entities.iter().enumerate() {
            let entity_data = entity.compute_gpu_data();
            context.immediate_write_buffer(
                &self.entity_data_buffer,
                bytemuck::bytes_of(&entity_data),
                index * std::mem::size_of::<GpuEntityData>()
            );
        }
    }

    pub fn update_submeshes(&mut self, context: &render::Context, assets: &GpuAssetStore) {
        self.submesh_data.clear();
        for (entity_index, entity) in self.entities.iter().enumerate() {
            if let Some(model) = entity.model {
                for submesh in assets.models[model].submeshes.iter() {
                    self.submesh_data.push(GpuSubmeshData {
                        entity_index: entity_index as u32,
                        mesh_index: submesh.mesh_handle.slot(),
                        material_index: submesh.material_index.slot(),
                        _padding: 0,
                    })
                }
            }
        }

        let count = self.submesh_data.len() as u32;

        context.immediate_write_buffer(
            &self.submesh_buffer,
            bytemuck::bytes_of(&count),
            0,
        );

        context.immediate_write_buffer(
            &self.submesh_buffer,
            bytemuck::cast_slice(&self.submesh_data),
            4,
        );
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_buffer(&self.entity_data_buffer);
        context.destroy_buffer(&self.submesh_buffer);
    }
}