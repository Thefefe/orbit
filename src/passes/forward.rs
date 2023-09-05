use ash::vk;
use glam::{Mat4, Vec3};

use crate::{graphics, Camera, EnvironmentMap, assets::AssetGraphData, scene::{SceneGraphData, GpuDrawCommand}, MAX_DRAW_COUNT, App, Settings};

use super::shadow_renderer::DirectionalLightGraphData;

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

        context.record_and_submit(|cmd| {
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

        Self {
            brdf_integration_map,
        }
    }

    pub fn render(
        &mut self,
        context: &mut graphics::Context,
        settings: &Settings,
        
        draw_commands: graphics::GraphBufferHandle,
        color_target: graphics::GraphImageHandle,
        color_resolve: Option<graphics::GraphImageHandle>,
        depth_target: graphics::GraphImageHandle,
        
        camera: &Camera,
        focused_camera: &Camera,
        render_mode: u32,
        
        environment_map: Option<&EnvironmentMap>,
        directional_light: DirectionalLightGraphData,
        
        assets: AssetGraphData,
        scene: SceneGraphData,
    ) {
        puffin::profile_function!();

        let screen_extent = context.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        let focused_view_matrix = focused_camera.transform.compute_matrix().inverse();
        let camera_view_matrix = camera.transform.compute_matrix().inverse();
        let camera_projection_matrix = camera.projection.compute_matrix(aspect_ratio);
        let camera_view_projection_matrix = camera_projection_matrix * camera_view_matrix;
        let skybox_view_projection_matrix =
            camera.projection.compute_matrix(aspect_ratio) *
            Mat4::from_quat(camera.transform.orientation).inverse();

        let frame_data = context.transient_storage_data("per_frame_data", bytemuck::bytes_of(&ForwardFrameData {
            view_projection: camera_view_projection_matrix,
            view: focused_view_matrix,
            view_pos: camera.transform.position,
            render_mode,
        }));

        let environment_map = environment_map.map(|e| e.import_to_graph(context));

        let brdf_integration_map = context.import(&self.brdf_integration_map);

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
                    test: graphics::PipelineState::Static(true),
                    write: false,
                    compare: vk::CompareOp::GREATER,
                }))
                .multisample_state(graphics::MultisampleState {
                    sample_count: settings.msaa,
                    alpha_to_coverage: true
                })
        );

        context.add_pass("forward_pass")
            .with_dependency(color_target, graphics::AccessKind::ColorAttachmentWrite)
            .with_dependencies(color_resolve.map(|i| (i, graphics::AccessKind::ColorAttachmentWrite)))
            .with_dependency(depth_target, graphics::AccessKind::DepthAttachmentWrite)
            .with_dependency(draw_commands, graphics::AccessKind::IndirectBuffer)
            .with_dependencies(directional_light.shadow_maps.map(|h| (h, graphics::AccessKind::FragmentShaderRead)))    
            .render(move |cmd, graph| {
                let color_target = graph.get_image(color_target);
                let color_resolve = color_resolve.map(|i| graph.get_image(i));
                let depth_target = graph.get_image(depth_target);

                let brdf_integration_map = graph.get_image(brdf_integration_map);
                
                let environment_map = environment_map.map(|e| e.get(graph));

                let per_frame_data = graph.get_buffer(frame_data);
                
                let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
                let index_buffer = graph.get_buffer(assets.index_buffer);
                let materials_buffer = graph.get_buffer(assets.materials_buffer);

                let instance_buffer = graph.get_buffer(scene.entity_buffer);
                let light_buffer = graph.get_buffer(scene.light_data_buffer);
                let draw_commands_buffer = graph.get_buffer(draw_commands);
                let directional_light_buffer = graph.get_buffer(directional_light.buffer);

                let mut color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(color_target.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(if environment_map.is_some() {
                        vk::AttachmentLoadOp::DONT_CARE 
                    } else {
                        vk::AttachmentLoadOp::CLEAR
                    })
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })
                    .store_op(vk::AttachmentStoreOp::STORE);

                if let Some(color_resolve) = color_resolve {
                    color_attachment = color_attachment
                        .resolve_image_view(color_resolve.view)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE);
                }

                let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
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

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(color_target.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .depth_attachment(&depth_attachemnt);

                cmd.begin_rendering(&rendering_info);
                
                if let Some(environment_map) = &environment_map {
                    cmd.bind_raster_pipeline(skybox_pipeline);

                    cmd.build_constants()
                        .mat4(&skybox_view_projection_matrix)
                        .sampled_image(environment_map.skybox);
    
                    cmd.draw(0..36, 0..1);    
                }

                cmd.bind_raster_pipeline(forward_pipeline);
                cmd.bind_index_buffer(&index_buffer, 0);

                let mut push_constants = cmd.build_constants()
                    .uint(screen_extent.width)
                    .uint(screen_extent.height)
                    .buffer(&per_frame_data)
                    .buffer(&vertex_buffer)
                    .buffer(&instance_buffer)
                    .buffer(&draw_commands_buffer)
                    .buffer(&materials_buffer)
                    .buffer(&directional_light_buffer)
                    .uint(scene.light_count as u32)
                    .buffer(&light_buffer);

                if let Some(environment_map) = &environment_map {
                    push_constants = push_constants
                        .sampled_image(environment_map.irradiance)
                        .sampled_image(environment_map.prefiltered)
                } else {
                    push_constants = push_constants
                        .uint(u32::MAX)
                        .uint(u32::MAX)
                };

                push_constants = push_constants.sampled_image(&brdf_integration_map);

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
}