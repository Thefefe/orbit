use std::borrow::Cow;

use ash::vk;
use glam::{Mat4, Vec4};
use gpu_allocator::MemoryLocation;

use crate::{
    assets::{AlphaMode, AssetGraphData, GpuMeshletDrawCommand},
    graphics::{self, AccessKind},
    scene::SceneGraphData,
    Projection,
};

pub const MAX_DRAW_COUNT: usize = 1_000_000;
pub const MAX_MESHLET_DISPATCH_COUNT: usize = 1_000_000;

#[derive(Debug, Clone, Copy, Default)]
pub enum OcclusionCullInfo {
    #[default]
    None,
    VisibilityRead {
        visibility_buffer: graphics::GraphBufferHandle,
    },
    VisibilityWrite {
        visibility_buffer: graphics::GraphBufferHandle,
        depth_pyramid: graphics::GraphImageHandle,
        noskip_alphamode: AlphaModeFlags,
        aspect_ratio: f32,
    },
    ShadowMask {
        visibility_buffer: graphics::GraphBufferHandle,
        shadow_mask: graphics::GraphImageHandle,
        aspect_ratio: f32,
    },
}

impl OcclusionCullInfo {
    fn visibility_buffer(&self) -> Option<graphics::GraphBufferHandle> {
        match self {
            OcclusionCullInfo::None => None,
            OcclusionCullInfo::VisibilityRead { visibility_buffer, .. } => Some(*visibility_buffer),
            OcclusionCullInfo::VisibilityWrite { visibility_buffer, .. } => Some(*visibility_buffer),
            OcclusionCullInfo::ShadowMask { visibility_buffer, .. } => Some(*visibility_buffer),
        }
    }

    fn visibility_buffer_dependency(&self) -> Option<(graphics::GraphBufferHandle, AccessKind)> {
        match self {
            OcclusionCullInfo::None => None,
            OcclusionCullInfo::VisibilityRead { visibility_buffer } => {
                Some((*visibility_buffer, AccessKind::ComputeShaderRead))
            }
            OcclusionCullInfo::VisibilityWrite { visibility_buffer, .. } => {
                Some((*visibility_buffer, AccessKind::ComputeShaderWrite))
            }
            OcclusionCullInfo::ShadowMask { visibility_buffer, .. } => {
                Some((*visibility_buffer, AccessKind::ComputeShaderReadGeneral))
            }
        }
    }

    fn depth_pyramid(&self) -> Option<graphics::GraphImageHandle> {
        match self {
            OcclusionCullInfo::None => None,
            OcclusionCullInfo::VisibilityRead { .. } => None,
            OcclusionCullInfo::VisibilityWrite { depth_pyramid, .. } => Some(*depth_pyramid),
            OcclusionCullInfo::ShadowMask { shadow_mask, .. } => Some(*shadow_mask),
        }
    }

    fn depth_pyramid_dependency(&self) -> Option<(graphics::GraphImageHandle, AccessKind)> {
        self.depth_pyramid().map(|h| (h, AccessKind::ComputeShaderReadGeneral))
    }

    fn pass_index(&self) -> u32 {
        match self {
            OcclusionCullInfo::None => 0,
            OcclusionCullInfo::VisibilityRead { .. } => 1,
            OcclusionCullInfo::VisibilityWrite { .. } => 2,
            OcclusionCullInfo::ShadowMask { .. } => 4,
        }
    }
}

pub struct CullInfo<'a> {
    pub view_matrix: Mat4,
    pub view_space_cull_planes: &'a [Vec4],
    pub projection: Projection,
    pub occlusion_culling: OcclusionCullInfo,
    pub alpha_mode_filter: AlphaModeFlags,
    pub debug_print: bool,
}

const MAX_CULL_PLANES: usize = 12;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuCullInfo {
    view_matrix: Mat4,
    reprojection_matrix: Mat4,
    cull_planes: [Vec4; MAX_CULL_PLANES],
    cull_plane_count: u32,

    alpha_mode_flags: u32,
    noskip_alpha_mode: u32,

    occlusion_pass: u32,
    visibility_buffer: u32,
    depth_pyramid: u32,
    secondary_depth_pyramid: u32,

    projection_type: u32,
    p00_or_width_recip_x2: f32,
    p11_or_height_recipx2: f32,
    z_near: f32,
    z_far: f32,
}

pub fn create_draw_commands(
    context: &mut graphics::Context,
    draw_commands_name: Cow<'static, str>,
    assets: AssetGraphData,
    scene: SceneGraphData,
    cull_info: &CullInfo,
    reuse_buffer: Option<graphics::GraphBufferHandle>,
) -> graphics::GraphBufferHandle {
    assert!(cull_info.view_space_cull_planes.len() <= MAX_CULL_PLANES);

    let meshlet_dispatch_buffer = context.create_transient(
        format!("{draw_commands_name}_meshlet_dispatch_buffer"),
        graphics::BufferDesc {
            size: MAX_MESHLET_DISPATCH_COUNT * 16,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        },
    );

    let meshlet_draw_command_buffer = reuse_buffer.unwrap_or_else(|| {
        context.create_transient(
            draw_commands_name.clone(),
            graphics::BufferDesc {
                size: MAX_DRAW_COUNT * std::mem::size_of::<GpuMeshletDrawCommand>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        )
    });

    let depth_pyramid_descriptor_index = cull_info
        .occlusion_culling
        .depth_pyramid()
        .map(|i| context.get_resource_descriptor_index(i).unwrap());

    let visibility_buffer_descriptor_index = cull_info
        .occlusion_culling
        .visibility_buffer()
        .map(|b| context.get_resource_descriptor_index(b).unwrap());

    let mut gpu_cull_info_data = GpuCullInfo {
        view_matrix: cull_info.view_matrix,
        cull_plane_count: cull_info.view_space_cull_planes.len() as u32,

        alpha_mode_flags: cull_info.alpha_mode_filter.0,

        occlusion_pass: cull_info.occlusion_culling.pass_index(),
        visibility_buffer: visibility_buffer_descriptor_index.unwrap_or(u32::MAX),
        depth_pyramid: depth_pyramid_descriptor_index.unwrap_or(u32::MAX),

        ..Default::default()
    };

    if cull_info.view_space_cull_planes.len() > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                cull_info.view_space_cull_planes.as_ptr(),
                gpu_cull_info_data.cull_planes.as_mut_ptr(),
                cull_info.view_space_cull_planes.len(),
            );
        }
    }

    if let OcclusionCullInfo::ShadowMask {
        aspect_ratio,
        ..
    } = cull_info.occlusion_culling
    {
        match cull_info.projection {
            Projection::Perspective { fov, near_clip } => {
                let f = 1.0 / f32::tan(0.5 * fov);
                gpu_cull_info_data.projection_type = 0;
                gpu_cull_info_data.p00_or_width_recip_x2 = f / aspect_ratio;
                gpu_cull_info_data.p11_or_height_recipx2 = f;
                gpu_cull_info_data.z_near = near_clip;
            }
            Projection::Orthographic {
                half_width,
                near_clip,
                far_clip,
            } => {
                let width = half_width * 2.0;
                let height = width * aspect_ratio.recip();
                gpu_cull_info_data.projection_type = 1;
                gpu_cull_info_data.p00_or_width_recip_x2 = width.recip() * 2.0;
                gpu_cull_info_data.p11_or_height_recipx2 = height.recip() * 2.0;
                gpu_cull_info_data.z_near = near_clip;
                gpu_cull_info_data.z_far = far_clip;
            }
        }
    }

    match cull_info.projection {
        Projection::Perspective { .. } => gpu_cull_info_data.projection_type = 0,
        Projection::Orthographic {.. } => gpu_cull_info_data.projection_type = 1,
    }

    if let OcclusionCullInfo::VisibilityWrite {
        aspect_ratio,
        noskip_alphamode,
        ..
    } = cull_info.occlusion_culling
    {
        gpu_cull_info_data.noskip_alpha_mode = noskip_alphamode.0;

        match cull_info.projection {
            Projection::Perspective { fov, near_clip } => {
                let f = 1.0 / f32::tan(0.5 * fov);
                gpu_cull_info_data.projection_type = 0;
                gpu_cull_info_data.p00_or_width_recip_x2 = f / aspect_ratio;
                gpu_cull_info_data.p11_or_height_recipx2 = f;
                gpu_cull_info_data.z_near = near_clip;
            }
            Projection::Orthographic {
                half_width,
                near_clip,
                far_clip,
            } => {
                let width = half_width * 2.0;
                let height = width * aspect_ratio.recip();
                gpu_cull_info_data.projection_type = 1;
                gpu_cull_info_data.p00_or_width_recip_x2 = width.recip() * 2.0;
                gpu_cull_info_data.p11_or_height_recipx2 = height.recip() * 2.0;
                gpu_cull_info_data.z_near = near_clip;
                gpu_cull_info_data.z_far = far_clip;
            }
        }
    }

    let cull_info_buffer = context.transient_storage_data(
        format!("{draw_commands_name}_cull_data"),
        bytemuck::bytes_of(&gpu_cull_info_data),
    );

    let entity_cull_pipeline = context.create_compute_pipeline(
        "entity_cull_pipeline",
        graphics::ShaderSource::spv("shaders/entity_cull.comp.spv"),
    );

    let meshlet_cull_pipeline = context.create_compute_pipeline(
        "meshlet_cull_pipeline",
        graphics::ShaderSource::spv("shaders/meshlet_cull.comp.spv"),
    );

    // let debug_print: u32 = if cull_info.debug_print { 1 } else { 0 };

    context
        .add_pass(format!("clearing_{draw_commands_name}_culling_buffer"))
        .with_dependency(meshlet_dispatch_buffer, AccessKind::ComputeShaderWrite)
        .with_dependency(meshlet_draw_command_buffer, AccessKind::ComputeShaderWrite)
        .render(move |cmd, graph| {
            let meshlet_dispatch_buffer = graph.get_buffer(meshlet_dispatch_buffer);
            let meshlet_draw_command_buffer = graph.get_buffer(meshlet_draw_command_buffer);
            cmd.fill_buffer(meshlet_dispatch_buffer, 0, 4, 0);
            cmd.fill_buffer(meshlet_dispatch_buffer, 4, 8, 1);
            cmd.fill_buffer(meshlet_draw_command_buffer, 0, 4, 0);
        });

    context
        .add_pass(format!("{draw_commands_name}_entity_culling"))
        .with_dependency(scene.entity_draw_buffer, AccessKind::ComputeShaderRead)
        .with_dependency(meshlet_dispatch_buffer, AccessKind::ComputeShaderWrite)
        .with_dependency(cull_info_buffer, AccessKind::ComputeShaderRead)
        .with_dependencies(cull_info.occlusion_culling.visibility_buffer_dependency())
        .with_dependencies(cull_info.occlusion_culling.depth_pyramid_dependency())
        .render(move |cmd, graph| {
            let mesh_info_buffer = graph.get_buffer(assets.mesh_info_buffer);
            let entity_draw_buffer = graph.get_buffer(scene.entity_draw_buffer);
            let entity_buffer = graph.get_buffer(scene.entity_buffer);
            let meshlet_dispatch_buffer = graph.get_buffer(meshlet_dispatch_buffer);
            let cull_info_buffer = graph.get_buffer(cull_info_buffer);

            cmd.bind_compute_pipeline(entity_cull_pipeline);

            cmd.build_constants()
                .buffer(entity_draw_buffer)
                .buffer(mesh_info_buffer)
                .buffer(meshlet_dispatch_buffer)
                .buffer(entity_buffer)
                .buffer(cull_info_buffer);

            cmd.dispatch([scene.entity_draw_count.div_ceil(256) as u32, 1, 1]);
        });

    context
        .add_pass(format!("{draw_commands_name}_meshlet_culling"))
        .with_dependency(meshlet_draw_command_buffer, AccessKind::ComputeShaderWrite)
        .with_dependency(meshlet_dispatch_buffer, AccessKind::IndirectBuffer)
        .with_dependency(cull_info_buffer, AccessKind::ComputeShaderRead)
        .with_dependencies(cull_info.occlusion_culling.visibility_buffer_dependency())
        .with_dependencies(cull_info.occlusion_culling.depth_pyramid_dependency())
        .render(move |cmd, graph| {
            let mesh_info_buffer = graph.get_buffer(assets.mesh_info_buffer);
            let meshlet_draw_command_buffer = graph.get_buffer(meshlet_draw_command_buffer);
            let meshlet_buffer = graph.get_buffer(assets.meshlet_buffer);
            let entity_buffer = graph.get_buffer(scene.entity_buffer);
            let meshlet_dispatch_buffer = graph.get_buffer(meshlet_dispatch_buffer);
            let cull_info_buffer = graph.get_buffer(cull_info_buffer);

            cmd.bind_compute_pipeline(meshlet_cull_pipeline);

            cmd.build_constants()
                .buffer(meshlet_dispatch_buffer)
                .buffer(mesh_info_buffer)
                .buffer(meshlet_buffer)
                .buffer(meshlet_draw_command_buffer)
                .buffer(entity_buffer)
                .buffer(cull_info_buffer);

            cmd.dispatch_indirect(meshlet_dispatch_buffer, 0);
        });

    meshlet_draw_command_buffer
}

pub struct DepthPyramid {
    pub pyramid: graphics::Image,
    pub usable: bool,
}

impl DepthPyramid {
    pub fn new(context: &graphics::Context, name: Cow<'static, str>, [width, height]: [u32; 2]) -> Self {
        let [width, height] = [width.next_power_of_two() / 2, height.next_power_of_two() / 2];
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(width, height));

        let pyramid = context.create_image(
            name,
            &graphics::ImageDesc {
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
            },
        );

        Self { pyramid, usable: false }
    }

    pub fn resize(&mut self, [width, height]: [u32; 2]) {
        let dimensions = [width.next_power_of_two() / 2, height.next_power_of_two() / 2, 1];
        let mip_levels = crate::gltf_loader::mip_levels_from_size(u32::max(dimensions[0], dimensions[1]));
        if self.pyramid.recreate(&graphics::ImageDesc {
            dimensions,
            mip_levels,
            ..self.pyramid.desc
        }) {
            self.usable = false;
        }
    }

    pub fn get_current(&mut self, context: &mut graphics::Context) -> graphics::GraphImageHandle {
        let pyramid = context.import_with(
            self.pyramid.name.clone(),
            &self.pyramid,
            graphics::GraphResourceImportDesc {
                initial_access: AccessKind::ComputeShaderReadGeneral,
                target_access: AccessKind::None,
                ..Default::default()
            },
        );

        pyramid
    }

    pub fn update(&mut self, context: &mut graphics::Context, depth_buffer: graphics::GraphImageHandle) {
        self.usable = true;

        let pyramid = context.import_with(
            self.pyramid.name.clone(),
            &self.pyramid,
            graphics::GraphResourceImportDesc {
                initial_access: AccessKind::ComputeShaderReadGeneral,
                target_access: AccessKind::None,
                ..Default::default()
            },
        );

        let reduce_pipeline = context.create_compute_pipeline(
            "depth_reduce_pipeline",
            graphics::ShaderSource::spv("shaders/depth_reduce.comp.spv"),
        );

        context
            .add_pass("depth_pyramid_reduce")
            .with_dependency(pyramid, AccessKind::ComputeShaderWrite)
            .with_dependency(depth_buffer, AccessKind::ComputeShaderRead)
            .render(move |cmd, graph| {
                let depth_buffer = graph.get_image(depth_buffer);
                let pyramid = graph.get_image(pyramid);

                cmd.bind_compute_pipeline(reduce_pipeline);

                for mip_level in 0..pyramid.mip_view_count() {
                    let src_view = if mip_level == 0 {
                        depth_buffer.full_view
                    } else {
                        cmd.barrier(
                            &[],
                            &[],
                            &[vk::MemoryBarrier2 {
                                src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                                dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                dst_access_mask: vk::AccessFlags2::SHADER_READ,
                                ..Default::default()
                            }],
                        );

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

#[inline]
pub fn update_multiple_depth_pyramids<const C: usize>(
    context: &mut graphics::Context,
    depth_pyramids: [graphics::GraphImageHandle; C],
    depth_buffers: [graphics::GraphImageHandle; C],
) {
    let reduce_pipeline = context.create_compute_pipeline(
        "depth_reduce_pipeline",
        graphics::ShaderSource::spv("shaders/depth_reduce.comp.spv"),
    );

    context
        .add_pass("depth_pyramid_reduce_multiple")
        .with_dependencies(depth_pyramids.into_iter().map(|i| (i, AccessKind::ComputeShaderWrite)))
        .with_dependencies(depth_buffers.into_iter().map(|i| (i, AccessKind::ComputeShaderRead)))
        .render(move |cmd, graph| {
            let depth_pyramids = depth_pyramids.map(|i| graph.get_image(i));
            let depth_buffers = depth_buffers.map(|i| graph.get_image(i));

            let max_mip_level = depth_pyramids.iter().map(|i| i.mip_view_count()).max().unwrap();

            cmd.bind_compute_pipeline(reduce_pipeline);

            for mip_level in 0..max_mip_level {
                if mip_level != 0 {
                    cmd.barrier(
                        &[],
                        &[],
                        &[vk::MemoryBarrier2 {
                            src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            dst_access_mask: vk::AccessFlags2::SHADER_READ,
                            ..Default::default()
                        }],
                    );
                }

                for i in 0..C {
                    if depth_pyramids[i].mip_view_count() <= mip_level {
                        continue;
                    }

                    let src_view = if mip_level == 0 {
                        depth_buffers[i].full_view
                    } else {
                        depth_pyramids[i].mip_view(mip_level - 1).unwrap()
                    };
                    let dst_view = depth_pyramids[i].mip_view(mip_level).unwrap();

                    cmd.build_constants()
                        .uint(dst_view.width())
                        .uint(dst_view.height())
                        .sampled_image(&src_view)
                        .storage_image(&dst_view);
                    cmd.dispatch([dst_view.width() / 16 + 1, dst_view.height() / 16 + 1, 1]);
                }
            }
        });
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct AlphaModeFlags(u32);
ash::vk_bitflags_wrapped!(AlphaModeFlags, u32);

impl AlphaModeFlags {
    pub const OPAQUE: Self = Self(1 << (AlphaMode::Opaque as usize));
    pub const MASKED: Self = Self(1 << (AlphaMode::Masked as usize));
    pub const TRANSPARENT: Self = Self(1 << (AlphaMode::Transparent as usize));

    pub const ALL: Self = Self(0b111);
}
