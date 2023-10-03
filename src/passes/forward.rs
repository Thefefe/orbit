use ash::vk;
use glam::{Mat4, Vec3};
use rand::prelude::Distribution;

use crate::{graphics, Camera, EnvironmentMap, assets::AssetGraphData, scene::{SceneGraphData, GpuDrawCommand}, MAX_DRAW_COUNT, App, Settings, passes::draw_gen::{create_draw_commands, CullInfo, OcclusionCullInfo, AlphaModeFlags}, math};

use super::{shadow_renderer::DirectionalLightGraphData, env_map_loader::GraphEnvironmentMap, draw_gen::DepthPyramid};

#[derive(Debug, Clone, Copy)]
pub struct TargetAttachments {
    pub color_target: graphics::GraphImageHandle,
    pub color_resolve: Option<graphics::GraphImageHandle>,
    pub depth_target: graphics::GraphImageHandle,
    pub depth_resolve: Option<graphics::GraphImageHandle>,
}

impl TargetAttachments {
    fn dependencies(&self) -> impl Iterator<Item = (graphics::GraphImageHandle, graphics::AccessKind)> {
        [
            (self.color_target, graphics::AccessKind::ColorAttachmentWrite),
            (self.depth_target, graphics::AccessKind::DepthAttachmentWrite),
        ].into_iter()
        .chain(self.color_resolve.map(|i| (i, graphics::AccessKind::ColorAttachmentWrite)))
        .chain(self.depth_resolve.map(|i| (i, graphics::AccessKind::DepthAttachmentWrite)))
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ForwardFrameData {
    view_projection: Mat4,
    view: Mat4,
    view_pos: Vec3,
    render_mode: u32,
}

pub struct ForwardRenderer {
    brdf_integration_map: graphics::Image,
    jitter_offset_texture: graphics::Image,
    visibility_buffer: graphics::Buffer,
    is_visibility_buffer_initialized: bool,
    pub depth_pyramid: DepthPyramid,
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
                }))
        );

        let brdf_integration_map = context.create_image("brdf_integration_map", &graphics::ImageDesc {
            ty: graphics::ImageType::Single2D,
            format: Self::LUT_FORMAT,
            dimensions: [512, 512, 1],
            mip_levels: 1,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: graphics::ImageSubresourceViewDesc::default(),
            ..Default::default()
        });

        
        let visibility_buffer = context.create_buffer("visibility_buffer", &graphics::BufferDesc {
            size: 4 * MAX_DRAW_COUNT,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: gpu_allocator::MemoryLocation::GpuOnly,
        });

        context.record_and_submit(|cmd| {
            cmd.fill_buffer(&visibility_buffer, 0, vk::WHOLE_SIZE, 0);

            cmd.barrier(&[], &[graphics::image_barrier(
                &brdf_integration_map,
                graphics::AccessKind::None,
                graphics::AccessKind::ColorAttachmentWrite
            )], &[]);

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

            cmd.barrier(&[], &[graphics::image_barrier(
                &brdf_integration_map,
                graphics::AccessKind::ColorAttachmentWrite,
                graphics::AccessKind::AllGraphicsRead
            )], &[]);
        });

        let screen_extent = context.swapchain.extent();
        let depth_pyramid = DepthPyramid::new(context, "camera_depth_pyramid".into(), [screen_extent.width, screen_extent.height]);

        Self {
            brdf_integration_map,
            jitter_offset_texture: create_jittered_offset_texture(context, 16, 8),
            visibility_buffer,
            is_visibility_buffer_initialized: false,
            depth_pyramid,
        }
    }

    pub fn render(
        &mut self,
        context: &mut graphics::Context,
        settings: &Settings,
        
        assets: AssetGraphData,
        scene: SceneGraphData,

        camera_frozen: bool,
        frustum_culling: bool,
        occlusion_culling: bool,

        environment_map: Option<&EnvironmentMap>,
        directional_light: DirectionalLightGraphData,

        target_attachments: &TargetAttachments,
        
        camera: &Camera,
        frozen_camera: &Camera,
        render_mode: u32,
    ) {
        puffin::profile_function!();

        let target_attachments = target_attachments.clone();

        let screen_extent = context.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        let camera_view_matrix = camera.transform.compute_matrix().inverse();
        let camera_projection_matrix = camera.projection.compute_matrix(aspect_ratio);
        let camera_view_projection_matrix = camera_projection_matrix * camera_view_matrix;
        let skybox_view_projection_matrix =
            camera.projection.compute_matrix(aspect_ratio) *
            Mat4::from_quat(camera.transform.orientation).inverse();

        let frame_data = context.transient_storage_data("per_frame_data", bytemuck::bytes_of(&ForwardFrameData {
            view_projection: camera_view_projection_matrix,
            view: camera_view_matrix,
            view_pos: camera.transform.position,
            render_mode,
        }));

        let environment_map = environment_map.map(|e| e.import_to_graph(context));

        let brdf_integration_map = context.import(&self.brdf_integration_map);
        let jitter_texture = context.import(&self.jitter_offset_texture);

        let visibility_buffer = context.import_with(
            "visibility_buffer",
            &self.visibility_buffer,
            graphics::GraphResourceImportDesc {
                initial_access: if self.is_visibility_buffer_initialized {
                    graphics::AccessKind::ComputeShaderWrite
                } else {
                    graphics::AccessKind::None
                },
                ..Default::default()
            }
        );

        if !camera_frozen {
            self.depth_pyramid.resize([screen_extent.width, screen_extent.height]);
        }

        let projection_matrix = frozen_camera.compute_projection_matrix();
        let view_matrix = frozen_camera.transform.compute_matrix().inverse();
        let frustum_planes = math::frustum_planes_from_matrix(&projection_matrix)
            .map(math::normalize_plane);

        let depth_pyramid = self.depth_pyramid.get_current(context);

        let draw_commands = create_draw_commands(context, "forward_draw_commands".into(), assets, scene,
            &CullInfo {
                view_matrix,
                view_space_cull_planes: if frustum_culling { &frustum_planes[0..5] } else { &[] },
                occlusion_culling: occlusion_culling
                    .then_some(OcclusionCullInfo::FirstPass { visibility_buffer })
                    .unwrap_or_default(),
                alpha_mode_filter: AlphaModeFlags::OPAQUE,
            },
            None
        );

        forward_pass(
            context,
            "forward_depth_prepass",
            settings,

            assets,
            scene,

            skybox_view_projection_matrix,
            
            frame_data,
            
            target_attachments,
            draw_commands,
            
            environment_map,
            directional_light,
            
            brdf_integration_map,
            jitter_texture,
            
            false,
        );

        if !camera_frozen {
            self.depth_pyramid.update(context, target_attachments.depth_resolve
                .unwrap_or(target_attachments.depth_target));
        }

        let draw_commands = create_draw_commands(context, "forward_draw_command".into(), assets, scene,
            &CullInfo {
                view_matrix,
                view_space_cull_planes: if frustum_culling { &frustum_planes[0..5] } else { &[] },
                occlusion_culling: OcclusionCullInfo::SecondPass {
                    visibility_buffer,
                    depth_pyramid,
                    p00: projection_matrix.col(0)[0],
                    p11: projection_matrix.col(1)[1],
                    z_near: frozen_camera.projection.near(),
                },
                alpha_mode_filter: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,
            },
            Some(draw_commands),
        );

        forward_pass(
            context,
            "forward_color_pass",
            settings,

            assets,
            scene,

            skybox_view_projection_matrix,
            
            frame_data,
            
            target_attachments,
            draw_commands,
            
            environment_map,
            directional_light,
            
            brdf_integration_map,
            jitter_texture,
            
            true,
        );
    }
}

fn forward_pass(
    context: &mut graphics::Context,
    pass_name: &'static str,

    settings: &Settings,
        
    assets: AssetGraphData,
    scene: SceneGraphData,

    skybox_view_projection_matrix: Mat4,

    frame_data: graphics::GraphBufferHandle,

    target_attachments: TargetAttachments,
    draw_commands: graphics::GraphBufferHandle,
        
    environment_map: Option<GraphEnvironmentMap>,
    directional_light: DirectionalLightGraphData,

    brdf_integration_map: graphics::GraphImageHandle,
    jitter_texture: graphics::GraphImageHandle,

    color_pass: bool,
) {
    let depth_prepass_pipeline = context.create_raster_pipeline(
        "forward_depth_prepass_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/forward.vert.spv"))
            .depth_state(Some(graphics::DepthState {
                format: App::DEPTH_FORMAT,
                test: graphics::PipelineState::Static(true),
                write: true,
                compare: vk::CompareOp::GREATER,
            }))
            .multisample_state(graphics::MultisampleState {
                sample_count: settings.msaa,
                alpha_to_coverage: true
            })
    );

    let forward_pipeline = context.create_raster_pipeline(
        "forward_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/forward.vert.spv"))
            .fragment_shader(graphics::ShaderSource::spv("shaders/forward.frag.spv"))
            .color_attachments(&[graphics::PipelineColorAttachment {
                format: App::COLOR_FORMAT,
                color_mask: vk::ColorComponentFlags::RGBA,
                color_blend: None,
            }])
            .depth_state(Some(graphics::DepthState {
                format: App::DEPTH_FORMAT,
                test: graphics::PipelineState::Static(true),
                write: true,
                compare: vk::CompareOp::GREATER,
            }))
            .multisample_state(graphics::MultisampleState {
                sample_count: settings.msaa,
                alpha_to_coverage: true
            })
    );

    
    let skybox_pipeline = context.create_raster_pipeline(
        "skybox_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/unit_cube.vert.spv"))
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
                test: graphics::PipelineState::Static(false),
                write: false,
                compare: vk::CompareOp::GREATER,
            }))
            .multisample_state(graphics::MultisampleState {
                sample_count: settings.msaa,
                alpha_to_coverage: true
            })
    );

    let mut pass_builder = context.add_pass(pass_name);

    pass_builder = pass_builder
        .with_dependency(draw_commands, graphics::AccessKind::IndirectBuffer)
        .with_dependency(target_attachments.depth_target, graphics::AccessKind::DepthAttachmentWrite)
        .with_dependencies(target_attachments.depth_resolve.map(|i| (i, graphics::AccessKind::DepthAttachmentWrite)));

    if color_pass {
        pass_builder = pass_builder
            .with_dependency(target_attachments.color_target, graphics::AccessKind::ColorAttachmentWrite)
            .with_dependencies(target_attachments.color_resolve.map(|i| (i, graphics::AccessKind::ColorAttachmentWrite)))
            .with_dependencies(directional_light.shadow_maps.map(|h| (h, graphics::AccessKind::FragmentShaderRead)));
    }

    pass_builder.render(move |cmd, graph| {
        let color_target = graph.get_image(target_attachments.color_target);
        let color_resolve = target_attachments.color_resolve.map(|i| graph.get_image(i));
        let depth_target = graph.get_image(target_attachments.depth_target);
        let depth_resolve = target_attachments.depth_resolve.map(|i| graph.get_image(i));

        let brdf_integration_map = graph.get_image(brdf_integration_map);
        let jitter_texture = graph.get_image(jitter_texture);
        
        let environment_map = environment_map.map(|e| e.get(graph));

        let per_frame_data = graph.get_buffer(frame_data);
        
        let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
        let index_buffer = graph.get_buffer(assets.index_buffer);
        let materials_buffer = graph.get_buffer(assets.materials_buffer);

        let entity_buffer = graph.get_buffer(scene.entity_buffer);
        let light_buffer = graph.get_buffer(scene.light_data_buffer);
        let draw_commands_buffer = graph.get_buffer(draw_commands);
        let directional_light_buffer = graph.get_buffer(directional_light.buffer);

        let color_attachment = if color_pass {
            let mut color_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(color_target.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(if environment_map.is_none() {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::LOAD
                })
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })
                .store_op(vk::AttachmentStoreOp::STORE);

            if let Some(color_resolve) = color_resolve {
                color_attachment = color_attachment
                .resolve_image_view(color_resolve.view)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE);
            }

            Some(color_attachment)
        } else {
            None
        };

        let mut depth_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(depth_target.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            })
            .store_op(vk::AttachmentStoreOp::STORE);

        if let Some(depth_resolve) = depth_resolve {
            depth_attachment = depth_attachment
                .resolve_image_view(depth_resolve.view)
                .resolve_image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::MIN);
        }

        let mut rendering_info = vk::RenderingInfo::builder()
            .render_area(color_target.full_rect())
            .layer_count(1)
            .depth_attachment(&depth_attachment);

        if let Some(color_attachment) = &color_attachment {
            rendering_info = rendering_info.color_attachments(std::slice::from_ref(color_attachment));
        }

        cmd.begin_rendering(&rendering_info);
        
        if let Some(environment_map) = color_pass.then_some(environment_map).flatten() {
            cmd.bind_raster_pipeline(skybox_pipeline);

            cmd.build_constants()
                .mat4(&skybox_view_projection_matrix)
                .sampled_image(environment_map.skybox);

            cmd.draw(0..36, 0..1);
        }

        cmd.bind_index_buffer(&index_buffer, 0);
        
        if color_pass { // color pass
            cmd.bind_raster_pipeline(forward_pipeline);
        } else { // depth pre-pass
            cmd.bind_raster_pipeline(depth_prepass_pipeline);
        }

        let mut push_constants = cmd.build_constants()
            .buffer(per_frame_data)
            .buffer(vertex_buffer)
            .buffer(entity_buffer)
            .buffer(draw_commands_buffer)
            .buffer(materials_buffer)
            .buffer(directional_light_buffer)
            .uint(scene.light_count as u32)
            .buffer(light_buffer);

        if let Some(environment_map) = &environment_map {
            push_constants = push_constants
                .sampled_image(environment_map.irradiance)
                .sampled_image(environment_map.prefiltered)
        } else {
            push_constants = push_constants
                .uint(u32::MAX)
                .uint(u32::MAX)
        };

        push_constants = push_constants
            .sampled_image(brdf_integration_map)
            .sampled_image(jitter_texture);

        drop(push_constants);

        cmd.draw_indexed_indirect_count(
            draw_commands_buffer,
            4,
            draw_commands_buffer,
            0,
            MAX_DRAW_COUNT as u32,
            std::mem::size_of::<GpuDrawCommand>() as u32,
        );

        cmd.end_rendering();
    });
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

fn create_jittered_offset_texture(context: &mut graphics::Context, window_size: usize, filter_size: usize) -> graphics::Image {
    let data = gen_offset_texture_data(window_size, filter_size);
    let num_filter_samples = filter_size * filter_size;
    
    let image = context.create_image("jittered_offset_texture", &graphics::ImageDesc {
        ty: graphics::ImageType::Single3D,
        format: vk::Format::R32G32B32A32_SFLOAT,
        dimensions: [num_filter_samples as u32 / 2, window_size as u32, window_size as u32],
        mip_levels: 1,
        samples: graphics::MultisampleCount::None,
        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        aspect: vk::ImageAspectFlags::COLOR,
        subresource_desc: graphics::ImageSubresourceViewDesc::default(),
        default_sampler: Some(graphics::SamplerKind::NearestRepeat),
    });

    context.immediate_write_image(
        &image, 0, 0..1,
        graphics::AccessKind::None,
        Some(graphics::AccessKind::FragmentShaderRead),
        bytemuck::cast_slice(data.as_slice()),
        None
    );

    image
}