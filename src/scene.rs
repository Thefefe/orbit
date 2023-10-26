use ash::vk;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Mat3, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};
use gpu_allocator::MemoryLocation;

use crate::{
    assets::{GpuAssetStore, ModelHandle},
    graphics::{self, FRAME_COUNT},
};

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
        Self {
            position,
            orientation,
            scale,
        }
    }

    pub fn translate_relative(&mut self, translation: Vec3) {
        self.position += self.orientation * translation;
    }

    pub fn compute_matrix(&self) -> glam::Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.orientation, self.position)
    }
}

#[derive(Debug, Clone)]
pub struct EntityData {
    pub name: Option<String>,
    pub transform: Transform,
    pub model: Option<ModelHandle>,
}

impl EntityData {
    fn compute_gpu_data(&self) -> GpuEntityData {
        let model_matrix = self.transform.compute_matrix();
        let normal_matrix = Mat4::from_mat3(Mat3::from_mat4(model_matrix.inverse().transpose()));
        GpuEntityData {
            model_matrix,
            normal_matrix,
        }
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
    alpha_mode: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuMeshDrawCommand {
    // DrawIndiexedIndirectCommand
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,

    pub _debug_index: u32, // only for debugging

    // other per-draw data
    pub material_index: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct LightData {
    pub color: Vec3,
    pub intensity: f32,
    pub position: Vec3,
    pub size: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuPointLight {
    color: Vec3,
    intensity: f32,
    position: Vec3,
    size: f32,
}

#[derive(Clone, Copy)]
pub struct SceneGraphData {
    pub submesh_count: usize,
    pub submesh_buffer: graphics::GraphBufferHandle,
    pub entity_buffer: graphics::GraphBufferHandle,
    pub light_count: usize, // temporary
    pub light_data_buffer: graphics::GraphHandle<graphics::BufferRaw>,
}

const MAX_INSTANCE_COUNT: usize = 1_000_000;
const MAX_LIGHT_COUNT: usize = 2_000;

struct Buffers {
    entity_data_buffer: graphics::Buffer,
    submesh_buffer: graphics::Buffer,
    light_data_buffer: graphics::Buffer,
}

impl Buffers {
    fn new(context: &graphics::Context) -> Self {
        let entity_data_buffer = context.create_buffer(
            "scene_instance_buffer",
            &graphics::BufferDesc {
                size: INSTANCE_BUFFER_SIZE * FRAME_COUNT,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let submesh_buffer = context.create_buffer(
            "submesh_buffer",
            &graphics::BufferDesc {
                size: SUBMESH_BUFFER_SIZE * FRAME_COUNT,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let light_data_buffer = context.create_buffer(
            "light_data_buffer",
            &graphics::BufferDesc {
                size: LIGHT_DATA_BUFFER_SIZE * FRAME_COUNT,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        Self {
            entity_data_buffer,
            submesh_buffer,
            light_data_buffer,
        }
    }
}

pub struct SceneData {
    pub entities: Vec<EntityData>,
    pub submesh_data: Vec<GpuSubmeshData>,
    pub lights: Vec<LightData>,
    frames: [Buffers; FRAME_COUNT],
}

const INSTANCE_BUFFER_SIZE: usize = MAX_INSTANCE_COUNT * std::mem::size_of::<GpuEntityData>();
const SUBMESH_BUFFER_SIZE: usize = MAX_INSTANCE_COUNT * std::mem::size_of::<GpuSubmeshData>() + 4;
const LIGHT_DATA_BUFFER_SIZE: usize = MAX_LIGHT_COUNT * std::mem::size_of::<GpuPointLight>();

impl SceneData {
    pub fn new(context: &graphics::Context) -> Self {
        let frames = std::array::from_fn(|_| Buffers::new(context));

        Self {
            entities: Vec::new(),
            submesh_data: Vec::new(),
            lights: Vec::new(),
            frames
        }
    }

    pub fn add_entity(&mut self, data: EntityData) -> usize {
        let index = self.entities.len();
        self.entities.push(data);
        index
    }

    pub fn add_light(&mut self, data: LightData) -> usize {
        let index = self.lights.len();
        self.lights.push(data);
        index
    }

    pub fn update_instances(&self, context: &graphics::Context) {
        puffin::profile_function!();
        let entity_datas: Vec<_> = self.entities.iter().map(EntityData::compute_gpu_data).collect();
        context.queue_write_buffer(
            &self.frames[context.frame_index()].entity_data_buffer,
            0,
            bytemuck::cast_slice(&entity_datas),
        );
    }

    pub fn update_submeshes(&mut self, context: &graphics::Context, assets: &GpuAssetStore) {
        puffin::profile_function!();
        self.submesh_data.clear();
        for (entity_index, entity) in self.entities.iter().enumerate() {
            if let Some(model) = entity.model {
                for submesh in assets.models[model].submeshes.iter() {
                    self.submesh_data.push(GpuSubmeshData {
                        entity_index: entity_index as u32,
                        mesh_index: submesh.mesh_handle.slot(),
                        material_index: submesh.material_index.slot(),
                        alpha_mode: assets.material_indices[submesh.material_index].alpha_mode.raw_index(),
                    })
                }
            }
        }

        let count = self.submesh_data.len() as u32;

        context.queue_write_buffer(
            &self.frames[context.frame_index()].submesh_buffer,
            0,
            bytemuck::bytes_of(&count),
        );
        context.queue_write_buffer(
            &self.frames[context.frame_index()].submesh_buffer,
            4,
            bytemuck::cast_slice(&self.submesh_data),
        );
    }

    pub fn update_lights(&self, context: &graphics::Context) {
        let light_datas: Vec<_> = self
            .lights
            .iter()
            .map(|light| GpuPointLight {
                color: light.color,
                intensity: light.intensity,
                position: light.position,
                size: light.size,
            })
            .collect();

        context.queue_write_buffer(
            &self.frames[context.frame_index()].light_data_buffer,
            0,
            bytemuck::cast_slice(light_datas.as_slice()),
        );
    }

    pub fn import_to_graph(&self, context: &mut graphics::Context) -> SceneGraphData {
        SceneGraphData {
            submesh_count: self.submesh_data.len(),
            submesh_buffer: context.import(&self.frames[context.frame_index()].submesh_buffer),
            entity_buffer: context.import(&self.frames[context.frame_index()].entity_data_buffer),
            light_count: self.lights.len(),
            light_data_buffer: context.import(&self.frames[context.frame_index()].light_data_buffer),
        }
    }
}
