use std::borrow::Cow;

use ash::vk;
use bytemuck::Zeroable;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Mat3, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};
use gpu_allocator::MemoryLocation;

use crate::{
    assets::{GpuAssets, MeshHandle},
    collections::{
        arena::Index,
        freelist_alloc::{BlockRange, FreeListAllocator},
    },
    graphics::{self, FRAME_COUNT},
    passes::shadow_renderer::{ShadowCommand, ShadowKind, ShadowRenderer},
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

#[derive(Debug, Clone, Copy)]
pub struct VisibilityBufferRange {
    alloc_index: Index,
    range: BlockRange,
}

#[derive(Debug, Clone, Default)]
pub struct EntityData {
    pub name: Option<Cow<'static, str>>,
    pub transform: Transform,
    pub mesh: Option<MeshHandle>,
    pub light: Option<Light>,
    pub visibility_buffer_range: Option<VisibilityBufferRange>,
}

impl EntityData {
    fn entity_gpu_data(&self) -> GpuEntityData {
        let model_matrix = self.transform.compute_matrix();
        let normal_matrix = Mat4::from_mat3(Mat3::from_mat4(model_matrix.inverse().transpose()));
        GpuEntityData {
            model_matrix,
            normal_matrix,
        }
    }

    fn light_gpu_data(&self, luminance_cutoff: f32) -> Option<GpuLightData> {
        let Some(light) = self.light.as_ref() else {
            return None;
        };

        let mut light_data = GpuLightData {
            color: light.color,
            intensity: light.intensity,
            light_type: light.params.kind() as u32,
            shadow_data_index: u32::MAX,
            ..GpuLightData::zeroed()
        };

        match &light.params {
            LightParams::Sky {
                irradiance,
                prefiltered,
            } => {
                light_data.irradiance_map_index = irradiance.descriptor_index().unwrap();
                light_data.prefiltered_map_index = prefiltered.descriptor_index().unwrap();
            }
            LightParams::Directional { angular_size: size } => {
                light_data.direction = -self.transform.orientation.mul_vec3(vec3(0.0, 0.0, -1.0));
                light_data.inner_radius = *size;
            }
            LightParams::Point { inner_radius: radius } => {
                light_data.position = self.transform.position;
                light_data.inner_radius = *radius;
                light_data.outer_radius = light.outer_radius(luminance_cutoff)
            }
        };

        Some(light_data)
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
pub struct GpuEntityDraw {
    instance_index: u32,
    mesh_index: u32,
    visibility_offset: u32,
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    Sky = 0,
    Directional = 1,
    Point = 2,
}

impl LightKind {
    pub fn name(&self) -> &str {
        match self {
            LightKind::Sky => "Sky",
            LightKind::Directional => "Directional",
            LightKind::Point => "Point",
        }
    }
}

#[derive(Debug, Clone)]
pub enum LightParams {
    Sky {
        irradiance: graphics::Image,
        prefiltered: graphics::Image,
    },
    Directional {
        angular_size: f32,
    },
    Point {
        inner_radius: f32,
    },
}

impl LightParams {
    fn default_from_kind(kind: LightKind) -> Self {
        match kind {
            LightKind::Sky => todo!(),
            LightKind::Directional => Self::Directional { angular_size: 0.6 },
            LightKind::Point => Self::Point { inner_radius: 0.1 },
        }
    }

    fn edit(&mut self, ui: &mut egui::Ui) {
        let mut kind = self.kind();

        ui.horizontal(|ui| {
            ui.label("type");
            egui::ComboBox::from_id_source("light_type").selected_text(kind.name()).show_ui(ui, |ui| {
                ui.selectable_value(&mut kind, LightKind::Directional, LightKind::Directional.name());
                ui.selectable_value(&mut kind, LightKind::Point, LightKind::Point.name());
            });
        });

        if kind != self.kind() {
            *self = Self::default_from_kind(kind);
        }

        match self {
            LightParams::Sky { .. } => {
                ui.label("editing not yet implemented");
            }
            LightParams::Directional { angular_size: size } => {
                ui.horizontal(|ui| {
                    ui.label("angular size");
                    ui.add(egui::DragValue::new(size).speed(0.01).clamp_range(0.0..=16.0));
                });
            }
            LightParams::Point { inner_radius: radius } => {
                ui.horizontal(|ui| {
                    ui.label("inner radius");
                    ui.add(egui::DragValue::new(radius).speed(0.01).clamp_range(0.0..=16.0));
                });
            }
        };
    }

    fn kind(&self) -> LightKind {
        match self {
            LightParams::Sky { .. } => LightKind::Sky,
            LightParams::Directional { .. } => LightKind::Directional,
            LightParams::Point { .. } => LightKind::Point,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Light {
    pub color: Vec3,
    pub intensity: f32,
    pub params: LightParams,
    pub cast_shadows: bool,
    pub _light_index: Option<usize>, // only used for debugging
}

impl Default for Light {
    fn default() -> Self {
        Self {
            color: Vec3::ONE,
            intensity: 1.0,
            params: LightParams::Point { inner_radius: 0.6 },
            cast_shadows: false,
            _light_index: None,
        }
    }
}

impl Light {
    pub fn is_skylight(&self) -> bool {
        matches!(&self.params, LightParams::Sky { .. })
    }

    pub fn is_directional(&self) -> bool {
        matches!(&self.params, LightParams::Directional { .. })
    }

    pub fn is_point(&self) -> bool {
        matches!(&self.params, LightParams::Point { .. })
    }

    pub fn edit(&mut self, ui: &mut egui::Ui) {
        let mut arr = self.color.to_array();
        ui.horizontal(|ui| {
            ui.label("color");
            ui.color_edit_button_rgb(&mut arr);
        });
        self.color = Vec3::from_array(arr);

        ui.horizontal(|ui| {
            ui.label("intensity");
            ui.add(egui::DragValue::new(&mut self.intensity).speed(0.01).clamp_range(0.0..=16.0));
        });

        self.params.edit(ui);
        ui.horizontal(|ui| {
            ui.label("cast shadows");
            ui.add(egui::Checkbox::without_text(&mut self.cast_shadows));
        });
    }

    pub fn outer_radius(&self, cutoff: f32) -> f32 {
        f32::sqrt(self.intensity / cutoff)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuLightData {
    pub light_type: u32,
    pub shadow_data_index: u32,
    pub irradiance_map_index: u32,
    pub prefiltered_map_index: u32,
    pub color: Vec3,
    pub intensity: f32,
    pub position: Vec3,
    pub inner_radius: f32,
    pub direction: Vec3,
    pub outer_radius: f32,
}

#[derive(Clone, Copy)]
pub struct SceneGraphData {
    pub entity_draw_count: usize,
    pub entity_draw_buffer: graphics::GraphBufferHandle,
    pub entity_buffer: graphics::GraphBufferHandle,
    pub light_count: usize, // temporary
    pub light_data_buffer: graphics::GraphHandle<graphics::BufferRaw>,
    pub meshlet_visibility_buffer: graphics::GraphBufferHandle,
}

const MAX_INSTANCE_COUNT: usize = 100_000;
const MAX_LIGHT_COUNT: usize = 2_000;

struct Buffers {
    entity_data_buffer: graphics::Buffer,
    entity_draw_buffer: graphics::Buffer,
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
                size: ENTITY_DRAW_BUFFER_SIZE * FRAME_COUNT,
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
            entity_draw_buffer: submesh_buffer,
            light_data_buffer,
        }
    }
}

const INSTANCE_BUFFER_SIZE: usize = MAX_INSTANCE_COUNT * std::mem::size_of::<GpuEntityData>();
const ENTITY_DRAW_BUFFER_SIZE: usize = 4 + MAX_INSTANCE_COUNT * std::mem::size_of::<GpuEntityDraw>();

// avg. 256 meshlet per instance
// 1 bit per meshlet, 4 byte per chunk
const MESHLET_VISIBILITY_BUFFER_CHUNK_COUNT: usize = MAX_INSTANCE_COUNT * 256 / 32;

const LIGHT_DATA_BUFFER_SIZE: usize = MAX_LIGHT_COUNT * std::mem::size_of::<GpuLightData>();

pub struct SceneData {
    pub entities: Vec<EntityData>,
    pub entity_draw_cache: Vec<GpuEntityDraw>,
    pub entity_data_cache: Vec<GpuEntityData>,
    pub light_data_cache: Vec<GpuLightData>,

    meshlet_visibility_buffer: graphics::Buffer,
    // chunks, not bytes
    meshlet_visibility_allocator: FreeListAllocator,

    frames: [Buffers; FRAME_COUNT],
}

impl SceneData {
    pub fn new(context: &graphics::Context) -> Self {
        let frames = std::array::from_fn(|_| Buffers::new(context));

        let meshlet_visibility_buffer = context.create_buffer(
            "meshlet_visibility_buffer",
            &graphics::BufferDesc {
                size: MESHLET_VISIBILITY_BUFFER_CHUNK_COUNT * 4,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        Self {
            entities: Vec::new(),

            entity_draw_cache: Vec::new(),
            entity_data_cache: Vec::new(),
            light_data_cache: Vec::new(),

            meshlet_visibility_buffer,
            meshlet_visibility_allocator: FreeListAllocator::new(MESHLET_VISIBILITY_BUFFER_CHUNK_COUNT),

            frames,
        }
    }

    pub fn add_entity(&mut self, data: EntityData) -> usize {
        let index = self.entities.len();
        self.entities.push(data);
        index
    }

    pub fn update_scene(
        &mut self,
        context: &graphics::Context,
        shadow_renderer: &mut ShadowRenderer,
        assets: &GpuAssets,
        luminance_cutoff: f32,
    ) {
        puffin::profile_function!();

        self.entity_draw_cache.clear();
        self.entity_data_cache.clear();
        self.light_data_cache.clear();
        shadow_renderer.clear_shadow_commands();

        for entity in self.entities.iter_mut() {
            if let Some(mesh) = entity.mesh {
                let instance_index = self.entity_data_cache.len() as u32;
                let visibility_offset = if let Some(v) = entity.visibility_buffer_range {
                    v.range.start
                } else {
                    let meshlet_count = assets.mesh_infos[mesh].mesh_lods[0].meshlet_count as usize;
                    let (alloc_index, range) =
                        self.meshlet_visibility_allocator.allocate(meshlet_count.div_ceil(32)).unwrap();
                    entity.visibility_buffer_range = Some(VisibilityBufferRange { alloc_index, range });

                    range.start
                };

                self.entity_data_cache.push(entity.entity_gpu_data());
                self.entity_draw_cache.push(GpuEntityDraw {
                    instance_index,
                    mesh_index: mesh.slot(),
                    visibility_offset: visibility_offset as u32,
                });
                // for submesh in assets.models[model].submeshes.iter() {
                //     self.entity_draw_cache.push(GpuEntityDraw {
                //         instance_index,
                //         mesh_index: submesh.mesh_handle.slot(),
                //         material_index: submesh.material_index.slot(),
                //         alpha_mode: assets.material_indices[submesh.material_index].alpha_mode.raw_index(),
                //     })
                // }
            }

            let light_data = entity.light_gpu_data(luminance_cutoff);
            if let (Some(light), Some(mut light_data)) = (entity.light.as_mut(), light_data) {
                light._light_index = Some(self.light_data_cache.len());
                if light.cast_shadows {
                    if let LightParams::Directional { .. } = &light.params {
                        let shadow_index = shadow_renderer.add_shadow(ShadowCommand {
                            name: entity.name.clone().unwrap_or("unnamed_light".into()),
                            kind: ShadowKind::Directional {
                                orientation: entity.transform.orientation,
                            },
                        });
                        light_data.shadow_data_index =
                            (shadow_index + ShadowRenderer::MAX_SHADOW_COMMANDS * context.frame_index()) as u32;
                    }
                }

                self.light_data_cache.push(light_data);
            }
        }

        context.queue_write_buffer(
            &self.frames[context.frame_index()].entity_data_buffer,
            0,
            bytemuck::cast_slice(&self.entity_data_cache),
        );

        let entity_draw_count = self.entity_draw_cache.len() as u32;
        context.queue_write_buffer(
            &self.frames[context.frame_index()].entity_draw_buffer,
            0,
            bytemuck::bytes_of(&entity_draw_count),
        );
        context.queue_write_buffer(
            &self.frames[context.frame_index()].entity_draw_buffer,
            4,
            bytemuck::cast_slice(&self.entity_draw_cache),
        );

        context.queue_write_buffer(
            &self.frames[context.frame_index()].light_data_buffer,
            0,
            bytemuck::cast_slice(&self.light_data_cache),
        );
    }

    pub fn import_to_graph(&self, context: &mut graphics::Context) -> SceneGraphData {
        SceneGraphData {
            entity_draw_count: self.entity_draw_cache.len(),
            entity_draw_buffer: context.import(&self.frames[context.frame_index()].entity_draw_buffer),
            entity_buffer: context.import(&self.frames[context.frame_index()].entity_data_buffer),
            light_count: self.light_data_cache.len(),
            light_data_buffer: context.import(&self.frames[context.frame_index()].light_data_buffer),
            meshlet_visibility_buffer: context.import(&self.meshlet_visibility_buffer),
        }
    }
}
