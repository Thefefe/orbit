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

    pub fn transform(&self, transform: &mut Transform) {
        transform.position += self.position;
        transform.orientation *= self.orientation;
        transform.scale *= self.scale;
    }

    pub fn translate(&mut self, translation: Vec3) {
        self.position += self.orientation * translation;
    }

    pub fn compute_affine(&self) -> glam::Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.orientation, self.position)
    }

    pub fn compute_linear(&self) -> Mat3 {
        Mat3::from_mat4(self.compute_affine())
    }
}

pub struct EntityData {
    pub transform: Transform,
    pub model: Option<ModelHandle>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuInstanceData {
    model_matrix: Mat4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuDrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

pub struct SceneBuffer {
    entities: Vec<EntityData>,
    pub instance_buffer: render::Buffer, 
}

const MAX_INSTANCE_COUNT: usize = 1_000_000;

impl SceneBuffer {
    pub fn new(context: &render::Context) -> Self {
        let instance_buffer = context.create_buffer("scene_instance_buffer", &render::BufferDesc {
            size: MAX_INSTANCE_COUNT * std::mem::size_of::<GpuInstanceData>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        });

        Self {
            entities: Vec::new(),
            instance_buffer,
        }
    }

    pub fn update_instances(&self, context: &render::Context) {
        for (index, entity) in self.entities.iter().enumerate() {
            let instance = GpuInstanceData {
                model_matrix: entity.transform.compute_affine(),
            };
            context.immediate_write_buffer(
                &self.instance_buffer,
                bytemuck::bytes_of(&instance),
                index * std::mem::size_of::<GpuInstanceData>()
            );
        }
    }

    pub fn add_entity(&mut self, data: EntityData) -> usize {
        let index = self.entities.len();
        self.entities.push(data);

        index
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_buffer(&self.instance_buffer);
    }
}

// TODO: do this on the gpu
pub fn write_draw_commands_from_scene(
    scene: &SceneBuffer,
    assets: &GpuAssetStore,
    commands: &mut [GpuDrawIndexedIndirectCommand],
) -> usize {
    let mut cmd_cursor = 0;

    for (entity_index, entity) in scene.entities.iter().enumerate() {
        if let Some(model_handle) = entity.model {
            for mesh_block in assets.submeshes(model_handle) {
                commands[cmd_cursor] = GpuDrawIndexedIndirectCommand {
                    index_count: mesh_block.index_range.size() as u32,
                    instance_count: 1,
                    first_index: mesh_block.index_range.start as u32,
                    vertex_offset: mesh_block.vertex_range.start as i32,
                    first_instance: entity_index as u32,
                };
                cmd_cursor += 1;
            }
        }
    }
    
    cmd_cursor
}