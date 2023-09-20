use std::borrow::Cow;

use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;

use crate::{graphics, assets::AssetGraphData, scene::{SceneGraphData, GpuDrawCommand}, MAX_DRAW_COUNT};

#[derive(Debug, Clone, Copy, Default)]
pub enum OcclusionCullInfo {
    #[default]
    None,
    FirstPass {
        visibility_buffer: graphics::GraphBufferHandle,
    },
    SecondPass {
        visibility_buffer: graphics::GraphBufferHandle,
        depth_pyramid: graphics::GraphImageHandle,
    },
}

impl OcclusionCullInfo {
    fn visibility_buffer(self) -> Option<graphics::GraphBufferHandle> {
        match self {
            OcclusionCullInfo::None                                 => None,
            OcclusionCullInfo::FirstPass { visibility_buffer, .. }  => Some(visibility_buffer),
            OcclusionCullInfo::SecondPass { visibility_buffer, .. } => Some(visibility_buffer),
        }
    }

    fn visibility_buffer_dependency(self) -> Option<(graphics::GraphBufferHandle, graphics::AccessKind)> {
        match self {
            OcclusionCullInfo::None => None,
            OcclusionCullInfo::FirstPass { visibility_buffer } => Some(
                (visibility_buffer, graphics::AccessKind::ComputeShaderRead)
            ),
            OcclusionCullInfo::SecondPass { visibility_buffer, .. } => Some(
                (visibility_buffer, graphics::AccessKind::ComputeShaderWrite)
            ),
        }
    }

    fn write_visibility_buffer(self) -> bool {
        match self {
            OcclusionCullInfo::None              => false,
            OcclusionCullInfo::FirstPass { .. }  => false,
            OcclusionCullInfo::SecondPass { .. } => true,
        }
    }

    fn depth_pyramid(self) -> Option<graphics::GraphImageHandle> {
        match self {
            OcclusionCullInfo::None                             => None,
            OcclusionCullInfo::FirstPass { .. }                 => None,
            OcclusionCullInfo::SecondPass { depth_pyramid, .. } => Some(depth_pyramid),
        }
    }

    fn depth_pyramid_dependency(self) -> Option<(graphics::GraphImageHandle, graphics::AccessKind)> {
        match self {
            OcclusionCullInfo::None                             => None,
            OcclusionCullInfo::FirstPass { .. }                 => None,
            OcclusionCullInfo::SecondPass { depth_pyramid, .. } => Some((depth_pyramid, graphics::AccessKind::ComputeShaderReadGeneral)),
        }
    }

    fn pass_index(&self) -> u32 {
        match self {
            OcclusionCullInfo::None              => 0,
            OcclusionCullInfo::FirstPass { .. }  => 1,
            OcclusionCullInfo::SecondPass { .. } => 2,
        }
    }
}

pub struct CullInfo {
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub cull_plane_mask: FrustumPlaneMask,
    pub occlusion_culling: OcclusionCullInfo,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuCullInfo {
    view_matrix: Mat4,
    projection_matrix: Mat4,
    cull_plane_mask: u32,

    occlusion_pass: u32,
    visibility_buffer: u32,
    depth_pyramid: u32,
}

pub fn create_draw_commands(
    context: &mut graphics::Context,
    draw_commands_name: Cow<'static, str>,
    assets: AssetGraphData,
    scene: SceneGraphData,
    cull_info: &CullInfo,
    reuse_buffer: Option<graphics::GraphBufferHandle>,
) -> graphics::GraphBufferHandle {
    let pass_name = format!("{draw_commands_name}_generation");
    
    let draw_commands = reuse_buffer.unwrap_or_else(|| context.create_transient(
        draw_commands_name,
        graphics::BufferDesc {
            size: MAX_DRAW_COUNT * std::mem::size_of::<GpuDrawCommand>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        }
    ));
    
    let pipeline = context.create_compute_pipeline(
        "scene_draw_gen_pipeline",
        graphics::ShaderSource::spv("shaders/scene_draw_gen.comp.spv")
    );

    let visibility_buffer_descriptor_index = cull_info.occlusion_culling
        .visibility_buffer()
        .map(|b| context.get_resource_descriptor_index(b))
        .flatten();

    let depth_pyramid_descriptor_index = cull_info.occlusion_culling
        .depth_pyramid()
        .map(|i| context.get_resource_descriptor_index(i))
        .flatten();

    let cull_data = context.transient_storage_data("cull_data", bytemuck::bytes_of(&GpuCullInfo {
        view_matrix: cull_info.view_matrix,
        projection_matrix: cull_info.projection_matrix,
        cull_plane_mask: cull_info.cull_plane_mask.0,

        occlusion_pass: cull_info.occlusion_culling.pass_index(),
        visibility_buffer: visibility_buffer_descriptor_index.unwrap_or(u32::MAX),
        depth_pyramid: depth_pyramid_descriptor_index.unwrap_or(u32::MAX),
    }));

    context.add_pass(pass_name)
        // .with_dependency(scene_submeshes, AccessKind::ComputeShaderRead)
        // .with_dependency(mesh_infos, AccessKind::ComputeShaderRead)
        .with_dependency(draw_commands, graphics::AccessKind::ComputeShaderWrite)
        .with_dependencies(cull_info.occlusion_culling.visibility_buffer_dependency())
        .with_dependencies(cull_info.occlusion_culling.depth_pyramid()
            .map(|i| (i, graphics::AccessKind::ComputeShaderReadGeneral)))
        .render(move |cmd, graph| {
            let mesh_infos = graph.get_buffer(assets.mesh_info_buffer);
            let scene_submeshes = graph.get_buffer(scene.submesh_buffer);
            let entity_buffer = graph.get_buffer(scene.entity_buffer);
            let draw_commands = graph.get_buffer(draw_commands);
            let cull_data = graph.get_buffer(cull_data);
            
            cmd.bind_compute_pipeline(pipeline);

            cmd.build_constants()
                .buffer(&scene_submeshes)
                .buffer(&mesh_infos)
                .buffer(&draw_commands)
                .buffer(&entity_buffer)
                .buffer(&cull_data);

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
        let [width, height] = [width.next_power_of_two() / 2, height.next_power_of_two() / 2];
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
        let dimensions = [width.next_power_of_two() / 2, height.next_power_of_two() / 2, 1];
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(dimensions[0], dimensions[1]));
        if self.pyramid.recreate(&graphics::ImageDesc { dimensions, mip_levels, ..self.pyramid.desc }) {
            self.usable = false;
        }
    }

    pub fn get_current(&mut self, context: &mut graphics::Context) -> graphics::GraphImageHandle {
        let pyramid = context.import_with(self.pyramid.name.clone(), &self.pyramid, graphics::GraphResourceImportDesc {
            initial_access: graphics::AccessKind::ComputeShaderReadGeneral,
            target_access: graphics::AccessKind::None,
            ..Default::default()
        });
        
        pyramid
    }

    pub fn update(&mut self, context: &mut graphics::Context, depth_buffer: graphics::GraphImageHandle) {
        self.usable = true;        
        
        let pyramid = context.import_with(self.pyramid.name.clone(), &self.pyramid, graphics::GraphResourceImportDesc {
            initial_access: graphics::AccessKind::ComputeShaderReadGeneral,
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