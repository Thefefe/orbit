use std::borrow::Cow;

use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;

use crate::{graphics, assets::AssetGraphData, scene::{SceneGraphData, GpuDrawCommand}, MAX_DRAW_COUNT};

pub fn create_draw_commands(
    context: &mut graphics::Context,
    draw_commands_name: Cow<'static, str>,
    view_projection_matrix: &Mat4,
    cull_plane_mask: FrustumPlaneMask,
    _depth_pyramid: Option<graphics::GraphImageHandle>,
    assets: AssetGraphData,
    scene: SceneGraphData,
) -> graphics::GraphBufferHandle {
    let pass_name = format!("{draw_commands_name}_generation");
    
    let draw_commands = context.create_transient(draw_commands_name, graphics::BufferDesc {
        size: MAX_DRAW_COUNT * std::mem::size_of::<GpuDrawCommand>(),
        usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
        memory_location: MemoryLocation::GpuOnly,
    });
    
    let pipeline = context.create_compute_pipeline(
        "scene_draw_gen_pipeline",
        graphics::ShaderSource::spv("shaders/scene_draw_gen.comp.spv")
    );

    let view_projection_matrix = view_projection_matrix.clone();

    context.add_pass(pass_name)
        // .with_dependency(scene_submeshes, AccessKind::ComputeShaderRead)
        // .with_dependency(mesh_infos, AccessKind::ComputeShaderRead)
        .with_dependency(draw_commands, graphics::AccessKind::ComputeShaderWrite)
        .render(move |cmd, graph| {
            let mesh_infos = graph.get_buffer(assets.mesh_info_buffer);
            let scene_submeshes = graph.get_buffer(scene.submesh_buffer);
            let entity_buffer = graph.get_buffer(scene.entity_buffer);
            let draw_commands = graph.get_buffer(draw_commands);
            
            cmd.bind_compute_pipeline(pipeline);

            cmd.build_constants()
                .mat4(&view_projection_matrix)
                .uint(cull_plane_mask.0)
                .buffer(&scene_submeshes)
                .buffer(&mesh_infos)
                .buffer(&draw_commands)
                .buffer(&entity_buffer);

            cmd.dispatch([scene.submesh_count as u32 / 256 + 1, 1, 1]);
        });
    
    draw_commands
}

pub struct DepthPyramid {
    pub pyramid: graphics::Image,
    usable: bool,
    current_access: graphics::AccessKind,
}

impl DepthPyramid {
    pub fn new(context: &graphics::Context, [width, height]: [u32; 2]) -> Self {
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(width, height));

        let pyramid = context.create_image("depth_pyramid", &graphics::ImageDesc {
            ty: graphics::ImageType::Single2D,
            format: vk::Format::R32_SFLOAT,
            dimensions: [width, height, 1],
            mip_levels,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: graphics::ImageSubresourceViewDesc {
                mip_count: u32::MAX,
                mip_descriptors: graphics::ImageDescriptorFlags::STORAGE,
                ..Default::default()
            },
            ..Default::default()
        });

        Self {
            pyramid,
            usable: false,
            current_access: graphics::AccessKind::None,
        }
    }

    pub fn resize(&mut self, [width, height]: [u32; 2]) {
        let dimensions = [width, height, 1];
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(width, height));
        if self.pyramid.recreate(&graphics::ImageDesc { dimensions, mip_levels, ..self.pyramid.desc }) {
            self.usable = false;
            self.current_access = graphics::AccessKind::None;
        }
    }

    pub fn get_current(&mut self, context: &mut graphics::Context) -> Option<graphics::GraphImageHandle> {
        if !self.usable {
            return None;
        }

        Some(context.import_with(self.pyramid.name.clone(), &self.pyramid, graphics::GraphResourceImportDesc {
            initial_access: graphics::AccessKind::ComputeShaderRead,
            target_access: graphics::AccessKind::ComputeShaderRead,
            ..Default::default()
        }))
    }

    pub fn update(&mut self, context: &mut graphics::Context) {
        let _pyramid = if self.usable {
            context.import(&self.pyramid)
        } else {
            self.usable = true;
            context.import_with(self.pyramid.name.clone(), &self.pyramid, graphics::GraphResourceImportDesc {
                initial_access: graphics::AccessKind::None,
                target_access: graphics::AccessKind::ComputeShaderRead,
                ..Default::default()
            })
        };
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct FrustumPlaneMask(u32);
ash::vk_bitflags_wrapped!(FrustumPlaneMask, u32);

impl FrustumPlaneMask {
    pub const RIGHT:   Self = Self(0b1);
    pub const LEFT:  Self = Self(0b01);
    pub const BOTTOM: Self = Self(0b001);
    pub const TOP:    Self = Self(0b0001);
    pub const NEAR:   Self = Self(0b00001);
    pub const FAR:    Self = Self(0b000001);

    pub const SIDES:  Self = Self(0b111100);
    pub const ALL:    Self = Self(0b111111);
}