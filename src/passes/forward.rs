use ash::vk;
use glam::{vec3, Mat4, Vec3};
use rand::prelude::Distribution;

use crate::{
    assets::{AssetGraphData, GpuAssets, GpuMeshletDrawCommand},
    graphics::{self, AccessKind, ShaderStage},
    math,
    passes::draw_gen::{
        create_meshlet_dispatch_command, create_meshlet_draw_commands, meshlet_dispatch_size, AlphaModeFlags, CullInfo,
        OcclusionCullInfo,
    },
    scene::{SceneData, SceneGraphData},
    App, Camera, Settings,
};

use super::{
    cluster::GraphClusterInfo,
    draw_gen::{DepthPyramid, MAX_DRAW_COUNT},
    shadow_renderer::ShadowRenderer,
};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Shaded = 0,
    Cascade = 1,
    Normal = 2,
    Metalic = 3,
    Roughness = 4,
    Emissive = 5,
    Ao = 6,
    Overdraw = 7,
    ClusterSlice = 8,
    ClusterId = 9,
}

impl From<u32> for RenderMode {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Shaded,
            1 => Self::Cascade,
            2 => Self::Normal,
            3 => Self::Metalic,
            4 => Self::Roughness,
            5 => Self::Emissive,
            6 => Self::Ao,
            7 => Self::Overdraw,
            8 => Self::ClusterSlice,
            9 => Self::ClusterId,
            _ => Self::Shaded,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TargetAttachments {
    pub color_target: graphics::GraphImageHandle,
    pub color_resolve: Option<graphics::GraphImageHandle>,
    pub depth_target: graphics::GraphImageHandle,
    pub depth_resolve: Option<graphics::GraphImageHandle>,
}

impl TargetAttachments {
    pub fn dependencies(&self) -> impl Iterator<Item = (graphics::GraphImageHandle, AccessKind)> {
        [
            (self.color_target, AccessKind::ColorAttachmentWrite),
            (self.depth_target, AccessKind::DepthAttachmentWrite),
        ]
        .into_iter()
        .chain(self.color_resolve.map(|i| (i, AccessKind::ColorAttachmentWrite)))
        .chain(self.depth_resolve.map(|i| (i, AccessKind::DepthAttachmentWrite)))
    }

    pub fn non_msaa_depth_target(&self) -> graphics::GraphImageHandle {
        self.depth_resolve.unwrap_or(self.depth_target)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ForwardFrameData {
    view_projection: Mat4,
    view: Mat4,
    view_pos: Vec3,
    render_mode: u32,
    screen_size: [u32; 2],
    z_near: f32,
    _padding: u32,
}

pub struct ForwardRenderer {
    brdf_integration_map: graphics::Image,
    jitter_offset_texture: graphics::Image,

    pub depth_pyramid: DepthPyramid,
    pub entity_visibility_buffer: graphics::Buffer,
    is_visibility_buffer_initialized: bool,
}

impl ForwardRenderer {
    const LUT_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;

    pub fn new(context: &mut graphics::Context) -> Self {
        let brdf_integration_pipeline = context.create_raster_pipeline(
            "brdf_integration_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/blit.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv("shaders/brdf_integration.frag.spv"))
                .rasterizer(graphics::RasterizerDesc {
                    cull_mode: vk::CullModeFlags::NONE,
                    ..Default::default()
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: Self::LUT_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }])
                .depth_state(Some(graphics::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: graphics::PipelineState::Static(true),
                    write: false,
                    compare: vk::CompareOp::GREATER,
                })),
        );

        let brdf_integration_map = context.create_image(
            "brdf_integration_map",
            &graphics::ImageDesc {
                ty: graphics::ImageType::Single2D,
                format: Self::LUT_FORMAT,
                dimensions: [512, 512, 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                aspect: vk::ImageAspectFlags::COLOR,
                subresource_desc: graphics::ImageSubresourceViewDesc::default(),
                ..Default::default()
            },
        );

        let visibility_buffer = context.create_buffer(
            "visibility_buffer",
            &graphics::BufferDesc {
                size: 4 * MAX_DRAW_COUNT,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: gpu_allocator::MemoryLocation::GpuOnly,
            },
        );

        context.record_and_submit(|cmd| {
            cmd.barrier(
                &[],
                &[graphics::image_barrier(
                    &brdf_integration_map,
                    AccessKind::None,
                    AccessKind::ColorAttachmentWrite,
                )],
                &[],
            );

            let color_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(brdf_integration_map.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::builder()
                .render_area(brdf_integration_map.full_rect())
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            cmd.begin_rendering(&rendering_info);
            cmd.bind_raster_pipeline(brdf_integration_pipeline);
            cmd.draw(0..6, 0..1);
            cmd.end_rendering();

            cmd.barrier(
                &[],
                &[graphics::image_barrier(
                    &brdf_integration_map,
                    AccessKind::ColorAttachmentWrite,
                    AccessKind::AllGraphicsRead,
                )],
                &[],
            );
        });

        let screen_extent = context.swapchain.extent();
        let depth_pyramid = DepthPyramid::new(
            context,
            "camera_depth_pyramid".into(),
            [screen_extent.width, screen_extent.height],
        );

        Self {
            brdf_integration_map,
            jitter_offset_texture: create_jittered_offset_texture(context, 16, 8),
            entity_visibility_buffer: visibility_buffer,
            is_visibility_buffer_initialized: false,
            depth_pyramid,
        }
    }

    pub fn render_depth_prepass(
        &mut self,
        context: &mut graphics::Context,
        settings: &Settings,

        assets: AssetGraphData,
        scene: SceneGraphData,

        target_attachments: &TargetAttachments,

        camera: &Camera,
        frozen_camera: &Camera,
    ) {
        puffin::profile_function!();

        let frustum_culling = settings.camera_debug_settings.frustum_culling;
        let occlusion_culling = settings.camera_debug_settings.occlusion_culling;
        let camera_frozen = settings.camera_debug_settings.freeze_camera;

        let target_attachments = target_attachments.clone();

        let screen_extent = context.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        let camera_view_matrix = camera.transform.compute_matrix().inverse();
        let camera_projection_matrix = camera.projection.compute_matrix(aspect_ratio);
        let camera_view_projection_matrix = camera_projection_matrix * camera_view_matrix;

        let mesh_shading = settings.use_mesh_shading;

        let visibility_buffer = context.import_with(
            "visibility_buffer",
            &self.entity_visibility_buffer,
            graphics::GraphResourceImportDesc {
                initial_access: if self.is_visibility_buffer_initialized {
                    AccessKind::ComputeShaderWrite
                } else {
                    AccessKind::None
                },
                ..Default::default()
            },
        );

        let meshlet_visibility_buffer = Some(scene.meshlet_visibility_buffer);

        if !camera_frozen {
            self.depth_pyramid.resize([screen_extent.width, screen_extent.height]);
        }

        let projection_matrix = frozen_camera.compute_projection_matrix();
        let view_matrix = frozen_camera.transform.compute_matrix().inverse();
        let frustum_planes = math::frustum_planes_from_matrix(&projection_matrix).map(math::normalize_plane);

        let cull_info = CullInfo {
            view_matrix,
            view_space_cull_planes: if frustum_culling { &frustum_planes[0..5] } else { &[] },
            projection: frozen_camera.projection,
            occlusion_culling: occlusion_culling
                .then_some(OcclusionCullInfo::VisibilityRead {
                    visibility_buffer,
                    meshlet_visibility_buffer,
                })
                .unwrap_or_default(),
            alpha_mode_filter: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,

            debug_print: false,

            lod_range: settings.lod_range(),
            lod_base: settings.lod_base,
            lod_step: settings.lod_step,
            lod_target_pos_view_space: vec3(0.0, 0.0, 0.0),
        };

        let (cull_info_buffer, mut draw_commands_buffer) =
            create_meshlet_dispatch_command(context, "early_forward_depth_prepass".into(), assets, scene, &cull_info);

        if !mesh_shading {
            draw_commands_buffer = create_meshlet_draw_commands(
                context,
                "early_forward_depth_prepass".into(),
                assets,
                scene,
                &cull_info,
                draw_commands_buffer,
            );
        }

        let depth_prepass_pipeline = {
            let mut desc = graphics::RasterPipelineDesc::builder()
                .fragment_shader(graphics::ShaderSource::spv(
                    "shaders/forward/forward_depth_prepass.frag.spv",
                ))
                .depth_state(Some(graphics::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: graphics::PipelineState::Static(true),
                    write: true,
                    compare: vk::CompareOp::GREATER,
                }))
                .multisample_state(graphics::MultisampleState {
                    sample_count: settings.msaa,
                    alpha_to_coverage: true,
                });

            if mesh_shading {
                let mesh_shader_props = context.device.gpu.mesh_shader_properties().unwrap();
                desc = desc
                    .task_shader(
                        ShaderStage::spv("shaders/forward/forward_depth_prepass.task.spv")
                            .spec_u32(0, meshlet_dispatch_size(context)),
                    )
                    .mesh_shader(
                        ShaderStage::spv("shaders/forward/forward_depth_prepass.mesh.spv")
                            .spec_u32(0, mesh_shader_props.max_preferred_mesh_work_group_invocations),
                    );
            } else {
                desc = desc.vertex_shader(graphics::ShaderSource::spv(
                    "shaders/forward/forward_depth_prepass.vert.spv",
                ));
            }

            context.create_raster_pipeline("forward_depth_prepass_pipeline", &desc)
        };

        context
            .add_pass("early_forward_depth_prepass")
            .with_dependency(target_attachments.depth_target, AccessKind::DepthAttachmentWrite)
            .with_dependencies(target_attachments.depth_resolve.map(|i| (i, AccessKind::DepthAttachmentWrite)))
            .with_dependency(draw_commands_buffer, AccessKind::IndirectBuffer)
            .with_dependencies(
                mesh_shading.then_some(scene.meshlet_visibility_buffer).map(|b| (b, AccessKind::TaskShaderRead)),
            )
            .render(move |cmd, graph| {
                let depth_target = graph.get_image(target_attachments.depth_target);
                let depth_resolve = target_attachments.depth_resolve.map(|i| graph.get_image(i));

                let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
                let meshlet_buffer = graph.get_buffer(assets.meshlet_buffer);
                let meshlet_data_buffer = graph.get_buffer(assets.meshlet_data_buffer);
                let entity_buffer = graph.get_buffer(scene.entity_buffer);
                let draw_commands_buffer = graph.get_buffer(draw_commands_buffer);
                let materials_buffer = graph.get_buffer(assets.materials_buffer);
                let cull_info_buffer = graph.get_buffer(cull_info_buffer);

                let mut depth_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(depth_target.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                if let Some(depth_resolve) = depth_resolve {
                    depth_attachment = depth_attachment
                        .resolve_image_view(depth_resolve.view)
                        .resolve_image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .resolve_mode(vk::ResolveModeFlags::MIN);
                }

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(depth_target.full_rect())
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(depth_prepass_pipeline);
                cmd.bind_index_buffer(&meshlet_data_buffer, 0, vk::IndexType::UINT8_EXT);

                cmd.build_constants()
                    .mat4(&camera_view_projection_matrix)
                    .buffer(draw_commands_buffer)
                    .buffer(cull_info_buffer)
                    .buffer(vertex_buffer)
                    .buffer(meshlet_buffer)
                    .buffer(meshlet_data_buffer)
                    .buffer(entity_buffer)
                    .buffer(materials_buffer);

                if mesh_shading {
                    cmd.draw_mesh_tasks_indirect(&draw_commands_buffer, 0, 1, 0);
                } else {
                    cmd.draw_indexed_indirect_count(
                        draw_commands_buffer,
                        4,
                        draw_commands_buffer,
                        0,
                        MAX_DRAW_COUNT as u32,
                        std::mem::size_of::<GpuMeshletDrawCommand>() as u32,
                    );
                }

                cmd.end_rendering();
            });

        if !occlusion_culling {
            return;
        }

        if !camera_frozen {
            self.depth_pyramid.update(
                context,
                target_attachments.depth_resolve.unwrap_or(target_attachments.depth_target),
            );
        }

        let depth_pyramid = self.depth_pyramid.get_current(context);

        let cull_info = CullInfo {
            view_matrix,
            view_space_cull_planes: if frustum_culling { &frustum_planes[0..5] } else { &[] },
            projection: frozen_camera.projection,
            occlusion_culling: OcclusionCullInfo::VisibilityWrite {
                visibility_buffer,
                meshlet_visibility_buffer,
                depth_pyramid,
                // noskip_alphamode: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,
                noskip_alphamode: AlphaModeFlags::empty(),
                aspect_ratio: frozen_camera.aspect_ratio,
            },
            alpha_mode_filter: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,
            lod_range: settings.lod_range(),
            lod_base: settings.lod_base,
            lod_step: settings.lod_step,
            lod_target_pos_view_space: vec3(0.0, 0.0, 0.0),
            debug_print: false,
        };

        let (cull_info_buffer, mut draw_commands_buffer) =
            create_meshlet_dispatch_command(context, "late_forward_depth_prepass".into(), assets, scene, &cull_info);

        if !mesh_shading {
            draw_commands_buffer = create_meshlet_draw_commands(
                context,
                "late_forward_depth_prepass".into(),
                assets,
                scene,
                &cull_info,
                draw_commands_buffer,
            );
        }

        context
            .add_pass("late_forward_depth_prepass")
            .with_dependency(target_attachments.depth_target, AccessKind::DepthAttachmentWrite)
            .with_dependencies(target_attachments.depth_resolve.map(|i| (i, AccessKind::DepthAttachmentWrite)))
            .with_dependency(draw_commands_buffer, AccessKind::IndirectBuffer)
            .with_dependencies(
                mesh_shading.then_some(scene.meshlet_visibility_buffer).map(|b| (b, AccessKind::TaskShaderWrite)),
            )
            .with_dependencies(mesh_shading.then_some(depth_pyramid).map(|i| (i, AccessKind::TaskShaderReadGeneral)))
            .render(move |cmd, graph| {
                let depth_target = graph.get_image(target_attachments.depth_target);
                let depth_resolve = target_attachments.depth_resolve.map(|i| graph.get_image(i));

                let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
                let meshlet_buffer = graph.get_buffer(assets.meshlet_buffer);
                let meshlet_data_buffer = graph.get_buffer(assets.meshlet_data_buffer);
                let entity_buffer = graph.get_buffer(scene.entity_buffer);
                let draw_commands_buffer = graph.get_buffer(draw_commands_buffer);
                let materials_buffer = graph.get_buffer(assets.materials_buffer);
                let cull_info_buffer = graph.get_buffer(cull_info_buffer);

                let mut depth_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(depth_target.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                if let Some(depth_resolve) = depth_resolve {
                    depth_attachment = depth_attachment
                        .resolve_image_view(depth_resolve.view)
                        .resolve_image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .resolve_mode(vk::ResolveModeFlags::MIN);
                }

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(depth_target.full_rect())
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(depth_prepass_pipeline);
                cmd.bind_index_buffer(&meshlet_data_buffer, 0, vk::IndexType::UINT8_EXT);

                cmd.build_constants()
                    .mat4(&camera_view_projection_matrix)
                    .buffer(draw_commands_buffer)
                    .buffer(cull_info_buffer)
                    .buffer(vertex_buffer)
                    .buffer(meshlet_buffer)
                    .buffer(meshlet_data_buffer)
                    .buffer(entity_buffer)
                    .buffer(materials_buffer);

                if mesh_shading {
                    cmd.draw_mesh_tasks_indirect(&draw_commands_buffer, 0, 1, 0);
                } else {
                    cmd.draw_indexed_indirect_count(
                        draw_commands_buffer,
                        4,
                        draw_commands_buffer,
                        0,
                        MAX_DRAW_COUNT as u32,
                        std::mem::size_of::<GpuMeshletDrawCommand>() as u32,
                    );
                }

                cmd.end_rendering();
            });
    }

    pub fn render(
        &mut self,
        context: &mut graphics::Context,
        settings: &Settings,

        assets: &GpuAssets,
        scene: &SceneData,

        skybox: Option<graphics::GraphImageHandle>,
        ssao_image: Option<graphics::GraphImageHandle>,
        shadow_renderer: &ShadowRenderer,

        target_attachments: &TargetAttachments,
        cluster_info: GraphClusterInfo,

        camera: &Camera,
        frozen_camera: &Camera,

        selected_light: Option<usize>,
    ) {
        puffin::profile_function!();

        let assets = assets.import_to_graph(context);
        let scene = scene.import_to_graph(context);
        let shadow_data_buffer = context.import(&shadow_renderer.shadow_data_buffer);
        let shadow_settings_buffer = shadow_renderer.gpu_shadow_settings(context);

        let render_mode = settings.camera_debug_settings.render_mode;
        let frustum_culling = settings.camera_debug_settings.frustum_culling;
        let occlusion_culling = settings.camera_debug_settings.occlusion_culling;

        let target_attachments = target_attachments.clone();

        let screen_extent = context.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        let camera_view_matrix = camera.transform.compute_matrix().inverse();
        let camera_projection_matrix = camera.projection.compute_matrix(aspect_ratio);
        let camera_view_projection_matrix = camera_projection_matrix * camera_view_matrix;
        let skybox_view_projection_matrix =
            camera.projection.compute_matrix(aspect_ratio) * Mat4::from_quat(camera.transform.orientation).inverse();

        let mesh_shading = settings.use_mesh_shading;

        let frame_data = context.transient_storage_data(
            "per_frame_data",
            bytemuck::bytes_of(&ForwardFrameData {
                view_projection: camera_view_projection_matrix,
                view: camera_view_matrix,
                view_pos: camera.transform.position,
                render_mode: render_mode as u32,
                screen_size: [screen_extent.width, screen_extent.height],
                z_near: frozen_camera.z_near(),
                _padding: 0,
            }),
        );

        let brdf_integration_map = context.import(&self.brdf_integration_map);
        let jitter_texture = context.import(&self.jitter_offset_texture);

        let visibility_buffer = context.import_with(
            "visibility_buffer",
            &self.entity_visibility_buffer,
            graphics::GraphResourceImportDesc {
                initial_access: if self.is_visibility_buffer_initialized {
                    AccessKind::ComputeShaderWrite
                } else {
                    AccessKind::None
                },
                ..Default::default()
            },
        );

        let meshlet_visibility_buffer = Some(scene.meshlet_visibility_buffer);

        let projection_matrix = frozen_camera.compute_projection_matrix();
        let view_matrix = frozen_camera.transform.compute_matrix().inverse();
        let frustum_planes = math::frustum_planes_from_matrix(&projection_matrix).map(math::normalize_plane);

        let cull_info = CullInfo {
            view_matrix,
            view_space_cull_planes: if frustum_culling { &frustum_planes[0..5] } else { &[] },
            projection: frozen_camera.projection,
            occlusion_culling: occlusion_culling
                .then_some(OcclusionCullInfo::VisibilityRead {
                    visibility_buffer,
                    meshlet_visibility_buffer,
                })
                .unwrap_or_default(),
            alpha_mode_filter: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,
            lod_range: settings.lod_range(),
            lod_base: settings.lod_base,
            lod_step: settings.lod_step,
            lod_target_pos_view_space: vec3(0.0, 0.0, 0.0),
            debug_print: false,
        };

        let (cull_info_buffer, mut draw_commands_buffer) =
            create_meshlet_dispatch_command(context, "forward".into(), assets, scene, &cull_info);

        if !mesh_shading {
            draw_commands_buffer = create_meshlet_draw_commands(
                context,
                "forward".into(),
                assets,
                scene,
                &cull_info,
                draw_commands_buffer,
            );
        }

        let forward_pipeline = {
            let mut desc = graphics::RasterPipelineDesc::builder()
                .fragment_shader(graphics::ShaderSource::spv("shaders/forward/forward.frag.spv"))
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: App::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }])
                .depth_state(Some(graphics::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: graphics::PipelineState::Static(true),
                    write: false,
                    compare: vk::CompareOp::EQUAL,
                }))
                .multisample_state(graphics::MultisampleState {
                    sample_count: settings.msaa,
                    alpha_to_coverage: true,
                });

            if mesh_shading {
                let mesh_shader_props = context.device.gpu.mesh_shader_properties().unwrap();
                desc = desc
                    .task_shader(
                        ShaderStage::spv("shaders/forward/forward.task.spv")
                            .spec_u32(0, meshlet_dispatch_size(context)),
                    )
                    .mesh_shader(
                        ShaderStage::spv("shaders/forward/forward.mesh.spv")
                            .spec_u32(0, mesh_shader_props.max_preferred_mesh_work_group_invocations),
                    );
            } else {
                desc = desc.vertex_shader(graphics::ShaderSource::spv("shaders/forward/forward.vert.spv"));
            }

            context.create_raster_pipeline("forward_pipeline", &desc)
        };

        let overdraw_pipeline = {
            let mut desc = graphics::RasterPipelineDesc::builder()
                .fragment_shader(graphics::ShaderSource::spv("shaders/forward/forward.frag.spv"))
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: App::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: Some(graphics::ColorBlendState {
                        src_color_blend_factor: vk::BlendFactor::ONE,
                        dst_color_blend_factor: vk::BlendFactor::ONE,
                        color_blend_op: vk::BlendOp::ADD,
                        src_alpha_blend_factor: vk::BlendFactor::ONE,
                        dst_alpha_blend_factor: vk::BlendFactor::ONE,
                        alpha_blend_op: vk::BlendOp::ADD,
                    }),
                }])
                .depth_state(Some(graphics::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: graphics::PipelineState::Static(true),
                    write: true,
                    compare: vk::CompareOp::GREATER_OR_EQUAL,
                }))
                .multisample_state(graphics::MultisampleState {
                    sample_count: settings.msaa,
                    alpha_to_coverage: true,
                });

            if mesh_shading {
                let mesh_shader_props = context.device.gpu.mesh_shader_properties().unwrap();
                desc = desc
                    .task_shader(
                        ShaderStage::spv("shaders/forward/forward.task.spv")
                            .spec_u32(0, meshlet_dispatch_size(context)),
                    )
                    .mesh_shader(
                        ShaderStage::spv("shaders/forward/forward.mesh.spv")
                            .spec_u32(0, mesh_shader_props.max_preferred_mesh_work_group_invocations),
                    );
            } else {
                desc = desc.vertex_shader(graphics::ShaderSource::spv("shaders/forward/forward.vert.spv"));
            }

            context.create_raster_pipeline("forward_overdraw_pipeline", &desc)
        };

        let skybox_pipeline = context.create_raster_pipeline(
            "skybox_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/skybox.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv("shaders/skybox.frag.spv"))
                .rasterizer(graphics::RasterizerDesc {
                    front_face: vk::FrontFace::CLOCKWISE,
                    ..Default::default()
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: App::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }])
                .depth_state(Some(graphics::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: graphics::PipelineState::Static(true),
                    write: false,
                    compare: vk::CompareOp::EQUAL,
                }))
                .multisample_state(graphics::MultisampleState {
                    sample_count: settings.msaa,
                    alpha_to_coverage: false,
                }),
        );

        let selected_light = selected_light.map(|x| x as u32).unwrap_or(u32::MAX);

        let mut pass_builder = context.add_pass("forward_pass");

        pass_builder = pass_builder
            .with_dependency(draw_commands_buffer, AccessKind::IndirectBuffer)
            .with_dependencies(
                mesh_shading.then_some(scene.meshlet_visibility_buffer).map(|b| (b, AccessKind::TaskShaderRead)),
            )
            .with_dependency(frame_data, AccessKind::VertexShaderRead)
            .with_dependency(cluster_info.light_offset_image, AccessKind::FragmentShaderReadGeneral)
            .with_dependency(cluster_info.light_index_list, AccessKind::FragmentShaderRead)
            .with_dependency(cluster_info.info_buffer, AccessKind::FragmentShaderRead)
            .with_dependency(target_attachments.depth_target, AccessKind::DepthAttachmentWrite)
            .with_dependencies(target_attachments.depth_resolve.map(|i| (i, AccessKind::DepthAttachmentWrite)))
            .with_dependency(target_attachments.color_target, AccessKind::ColorAttachmentWrite)
            .with_dependencies(target_attachments.color_resolve.map(|i| (i, AccessKind::ColorAttachmentWrite)))
            .with_dependencies(shadow_renderer.rendered_shadow_maps().map(|h| (h, AccessKind::FragmentShaderRead)))
            .with_dependencies(ssao_image.map(|i| (i, AccessKind::FragmentShaderRead)));

        pass_builder.render(move |cmd, graph| {
            let color_target = graph.get_image(target_attachments.color_target);
            let color_resolve = target_attachments.color_resolve.map(|i| graph.get_image(i));
            let depth_target = graph.get_image(target_attachments.depth_target);
            let depth_resolve = target_attachments.depth_resolve.map(|i| graph.get_image(i));

            let brdf_integration_map = graph.get_image(brdf_integration_map);
            let jitter_texture = graph.get_image(jitter_texture);

            let skybox = skybox.map(|h| graph.get_image(h));
            let ssao_image = ssao_image.map(|h| graph.get_image(h));

            let per_frame_data = graph.get_buffer(frame_data);
            let cluster_info_buffer = graph.get_buffer(cluster_info.info_buffer);

            let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
            let meshlet_buffer = graph.get_buffer(assets.meshlet_buffer);
            let meshlet_data_buffer = graph.get_buffer(assets.meshlet_data_buffer);
            let materials_buffer = graph.get_buffer(assets.materials_buffer);

            let entity_buffer = graph.get_buffer(scene.entity_buffer);
            let light_buffer = graph.get_buffer(scene.light_data_buffer);
            let draw_commands_buffer = graph.get_buffer(draw_commands_buffer);
            let cull_info_buffer = graph.get_buffer(cull_info_buffer);

            let shadow_data_buffer = graph.get_buffer(shadow_data_buffer);
            let shadow_settings_buffer = graph.get_buffer(shadow_settings_buffer);

            let mut color_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(color_target.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(if skybox.is_none() || render_mode == RenderMode::Overdraw {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    // vk::AttachmentLoadOp::DONT_CARE
                    vk::AttachmentLoadOp::CLEAR
                })
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 0.0, 1.0, 0.0],
                        // float32: [0.0, 0.0, 0.0, 0.0],
                    },
                })
                .store_op(vk::AttachmentStoreOp::STORE);

            if let Some(color_resolve) = color_resolve {
                color_attachment = color_attachment
                    .resolve_image_view(color_resolve.view)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .resolve_mode(vk::ResolveModeFlags::AVERAGE);
            }

            let mut depth_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(depth_target.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                })
                .store_op(vk::AttachmentStoreOp::STORE);

            if let Some(depth_resolve) = depth_resolve {
                depth_attachment = depth_attachment
                    .resolve_image_view(depth_resolve.view)
                    .resolve_image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .resolve_mode(vk::ResolveModeFlags::MIN);
            }

            let rendering_info = vk::RenderingInfo::builder()
                .render_area(color_target.full_rect())
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            cmd.begin_rendering(&rendering_info);

            let draw_skybox = render_mode != RenderMode::Overdraw;
            if let Some(skybox) = (draw_skybox).then_some(skybox).flatten() {
                cmd.bind_raster_pipeline(skybox_pipeline);

                cmd.build_constants().mat4(&skybox_view_projection_matrix).sampled_image(skybox);

                cmd.draw(0..36, 0..1);
            }

            cmd.bind_index_buffer(&meshlet_data_buffer, 0, vk::IndexType::UINT8_EXT);

            if render_mode == RenderMode::Overdraw {
                cmd.bind_raster_pipeline(overdraw_pipeline);
            } else {
                cmd.bind_raster_pipeline(forward_pipeline);
            }

            let mut constants = cmd.build_constants()
                .buffer(draw_commands_buffer)
                .buffer(cull_info_buffer)
                .buffer(per_frame_data)
                .buffer(vertex_buffer)
                .buffer(meshlet_buffer)
                .buffer(meshlet_data_buffer)
                .buffer(entity_buffer)
                .buffer(materials_buffer)
                .buffer(cluster_info_buffer)
                .uint(scene.light_count as u32)
                .buffer(light_buffer)
                .buffer(shadow_data_buffer)
                .buffer(shadow_settings_buffer)
                .uint(selected_light)
                .sampled_image(brdf_integration_map)
                .sampled_image(jitter_texture);

            if let Some(ssao_image) = ssao_image {
                constants = constants.sampled_image(ssao_image);
            } else {
                constants = constants.uint(u32::MAX);
            }

            constants.push();

            if mesh_shading {
                cmd.draw_mesh_tasks_indirect(&draw_commands_buffer, 0, 1, 0);
            } else {
                cmd.draw_indexed_indirect_count(
                    draw_commands_buffer,
                    4,
                    draw_commands_buffer,
                    0,
                    MAX_DRAW_COUNT as u32,
                    std::mem::size_of::<GpuMeshletDrawCommand>() as u32,
                );
            }

            cmd.end_rendering();
        });
    }
}

fn gen_offset_texture_data(window_size: usize, filter_size: usize) -> Vec<f32> {
    use std::f32::consts::PI;

    let mut data = Vec::with_capacity(window_size * window_size * filter_size * 2);

    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Uniform::from(-0.5f32..=0.5f32);

    for _ in 0..window_size {
        for _ in 0..window_size {
            for v in 0..filter_size {
                for u in 0..filter_size {
                    let v = filter_size - 1 - v;

                    let x = (u as f32 + 0.5 + dist.sample(&mut rng)) / filter_size as f32;
                    let y = (v as f32 + 0.5 + dist.sample(&mut rng)) / filter_size as f32;

                    data.push(y.sqrt() * f32::cos(2.0 * PI * x));
                    data.push(y.sqrt() * f32::sin(2.0 * PI * x));
                }
            }
        }
    }

    data
}

fn create_jittered_offset_texture(
    context: &mut graphics::Context,
    window_size: usize,
    filter_size: usize,
) -> graphics::Image {
    let data = gen_offset_texture_data(window_size, filter_size);
    let num_filter_samples = filter_size * filter_size;

    let image = context.create_image(
        "jittered_offset_texture",
        &graphics::ImageDesc {
            ty: graphics::ImageType::Single3D,
            format: vk::Format::R32G32B32A32_SFLOAT,
            dimensions: [num_filter_samples as u32 / 2, window_size as u32, window_size as u32],
            mip_levels: 1,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: graphics::ImageSubresourceViewDesc::default(),
            default_sampler: Some(graphics::SamplerKind::NearestRepeat),
        },
    );

    context.immediate_write_image(
        &image,
        0,
        0..1,
        AccessKind::None,
        Some(AccessKind::FragmentShaderRead),
        bytemuck::cast_slice(data.as_slice()),
        None,
    );

    image
}
