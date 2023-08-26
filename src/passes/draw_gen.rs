use std::borrow::Cow;

use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;

use crate::{render, utils, assets::AssetGraphData, scene::{SceneGraphData, GpuDrawCommand}, MAX_DRAW_COUNT};

pub struct SceneDrawGen {
    pipeline: render::ComputePipeline,
    depth_pyramid: render::RecreatableImage,
}

impl SceneDrawGen {
    pub fn new(context: &render::Context) -> Self {
        let spv = utils::load_spv("shaders/scene_draw_gen.comp.spv").unwrap();
        let module = context.create_shader_module(&spv, "scene_draw_gen_module"); 

        let pipeline = context.create_compute_pipeline("scene_draw_gen_pipeline", &render::ShaderStage {
            module,
            entry: cstr::cstr!("main"),
        });

        context.destroy_shader_module(module);

        let screen_extent = context.swapchain.extent();
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(screen_extent.width, screen_extent.height));

        let depth_pyramid = render::RecreatableImage::new(context, "depth_pyramid".into(), render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: vk::Format::R32_SFLOAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: render::ImageSubresourceViewDesc::default(),
        });

        Self { pipeline, depth_pyramid }

    }

    pub fn create_draw_commands(
        &self,
        context: &mut render::Context,
        draw_commands_name: Cow<'static, str>,
        view_projection_matrix: &Mat4,
        cull_plane_mask: FrustumPlaneMask,
        assets: AssetGraphData,
        scene: SceneGraphData,
    ) -> render::GraphBufferHandle {
        let pass_name = format!("{draw_commands_name}_generation");
        let draw_commands = context.create_transient_buffer(draw_commands_name, render::BufferDesc {
            size: MAX_DRAW_COUNT * std::mem::size_of::<GpuDrawCommand>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        });
        let pipeline = self.pipeline;

        let view_projection_matrix = view_projection_matrix.clone();

        context.add_pass(pass_name)
            // .with_dependency(scene_submeshes, AccessKind::ComputeShaderRead)
            // .with_dependency(mesh_infos, AccessKind::ComputeShaderRead)
            .with_dependency(draw_commands, render::AccessKind::ComputeShaderWrite)
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

    pub fn update_depth_pyramid(&mut self, context: &mut render::Context, depth_image: &render::GraphImageHandle) {
        let screen_extent = context.swapchain_extent();
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(screen_extent.width, screen_extent.height));

        self.depth_pyramid.recreate(context, render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: vk::Format::R32_SFLOAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: render::ImageSubresourceViewDesc::default(),
        });
        let depth_pyramid = self.depth_pyramid.get_current(context);
        
        
    }

    pub fn destroy(&mut self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
        self.depth_pyramid.destroy(context);
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