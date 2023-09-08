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
    pub usable: bool,
}

impl DepthPyramid {
    pub fn new(context: &graphics::Context, [width, height]: [u32; 2]) -> Self {
        let [width, height] = [width.next_power_of_two() / 4, height.next_power_of_two() / 4];
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
            default_sampler: Some(graphics::SamplerKind::NearestClamp),
            ..Default::default()
        });

        Self {
            pyramid,
            usable: false,
        }
    }

    pub fn resize(&mut self, [width, height]: [u32; 2]) {
        let dimensions = [width.next_power_of_two() / 4, height.next_power_of_two() / 4, 1];
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(dimensions[0], dimensions[1]));
        if self.pyramid.recreate(&graphics::ImageDesc { dimensions, mip_levels, ..self.pyramid.desc }) {
            self.usable = false;
        }
    }

    pub fn get_current(&mut self, context: &mut graphics::Context) -> graphics::GraphImageHandle {
        let pyramid = context.import_with(self.pyramid.name.clone(), &self.pyramid, graphics::GraphResourceImportDesc {
            initial_access: if self.usable { graphics::AccessKind::FragmentShaderRead } else { graphics::AccessKind::None },
            target_access: graphics::AccessKind::None,
            ..Default::default()
        });
        
        pyramid
    }

    pub fn update(&mut self, context: &mut graphics::Context, depth_buffer: graphics::GraphImageHandle) {
        let pyramid = context.import_with(self.pyramid.name.clone(), &self.pyramid, graphics::GraphResourceImportDesc {
            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            ..Default::default()
        });

        let reduce_pipeline = context.create_compute_pipeline(
            "depth_reduce_pipeline",
            graphics::ShaderSource::spv("shaders/depth_reduce.comp.spv")
        );
    
        context.add_pass("depth_pyramid_reduce")
            .with_dependency(pyramid, graphics::AccessKind::ComputeShaderWrite)
            .with_dependency(depth_buffer, graphics::AccessKind::ComputeShaderRead)
            .render(move |cmd, graph| {
                let depth_buffer = graph.get_image(depth_buffer);
                let pyramid = graph.get_image(pyramid);
                
                cmd.bind_compute_pipeline(reduce_pipeline);
    
                for mip_level in 0..pyramid.mip_view_count() {
                    let src_view = if mip_level == 0 {
                        depth_buffer.full_view
                    } else {
                        cmd.barrier(&[], &[], &[
                            vk::MemoryBarrier2 {
                                src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                                dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                dst_access_mask: vk::AccessFlags2::SHADER_READ,
                                ..Default::default()
                            }
                        ]);

                        pyramid.mip_view(mip_level - 1).unwrap()
                    };
                    let dst_view = pyramid.mip_view(mip_level).unwrap();

                    cmd.build_constants()
                        .uint(dst_view.width())
                        .uint(dst_view.height())
                        .sampled_image(&src_view) 
                        .storage_image(&dst_view);
                    cmd.dispatch([dst_view.width() / 16 + 1, dst_view.height() / 16 + 1, 1]);
                }
            });
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