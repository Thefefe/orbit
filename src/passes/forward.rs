use ash::vk;
use glam::{Mat4, Vec3};

use crate::{render, utils, Camera, EnvironmentMap, assets::AssetGraphData, scene::{SceneGraphData, GpuDrawCommand}, MAX_DRAW_COUNT, App};

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
    forward_pipeline: render::RasterPipeline,
    skybox_pipeline: render::RasterPipeline,

    brdf_integration_pipeline: render::RasterPipeline,
    brdf_integration_map: render::Image,
}

impl ForwardRenderer {
    pub fn new(context: &render::Context) -> Self {
        let forward_pipeline = {
            let vertex_shader = utils::load_spv("shaders/mesh.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/mesh.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "mesh_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "mesh_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("forward_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: Some(render::ShaderStage {
                    module: fragment_module,
                    entry,
                }),
                vertex_input: render::VertexInput::default(),
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::BACK,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: App::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: Some(render::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: true,
                    write: true,
                    compare: vk::CompareOp::GREATER,
                }),
                multisample: App::MULTISAMPLING,
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        
        let skybox_pipeline = {
            let vertex_shader = utils::load_spv("shaders/unit_cube.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/skybox.frag.spv").unwrap();

            let vertex_module = context
                .create_shader_module(&vertex_shader, "unit_cube_vertex_shader");
            let fragment_module = context
                .create_shader_module(&fragment_shader, "skybox_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("skybox_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: Some(render::ShaderStage {
                    module: fragment_module,
                    entry,
                }),
                vertex_input: render::VertexInput::default(),
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::FRONT,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: App::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: Some(render::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: false,
                    write: false,
                    compare: vk::CompareOp::GREATER,
                }),
                multisample: App::MULTISAMPLING,
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        
        let brdf_integration_pipeline = {
            let vertex_shader = utils::load_spv("shaders/blit.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/brdf_integration.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "blit_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "brdf_integration_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("brdf_integration_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: Some(render::ShaderStage {
                    module: fragment_module,
                    entry,
                }),
                vertex_input: render::VertexInput::default(),
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: vk::Format::R16G16B16A16_SFLOAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: None,
                multisample: render::MultisampleCount::None,
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        let brdf_integration_map = context.create_image("brdf_integration_map", &render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: vk::Format::R16G16B16A16_SFLOAT,
            dimensions: [512, 512, 1],
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        context.record_and_submit(|cmd| {
            cmd.barrier(&[], &[render::image_barrier(
                &brdf_integration_map,
                render::AccessKind::None,
                render::AccessKind::ColorAttachmentWrite
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

            cmd.barrier(&[], &[render::image_barrier(
                &brdf_integration_map,
                render::AccessKind::ColorAttachmentWrite,
                render::AccessKind::AllGraphicsRead
            )], &[]);
        });

        Self {
            forward_pipeline,
            skybox_pipeline,

            brdf_integration_pipeline,
            brdf_integration_map,
        }
    }

    pub fn render(
        &mut self,
        context: &mut render::Context,
        
        draw_commands: render::GraphBufferHandle,
        color_target: render::GraphImageHandle,
        color_resolve: Option<render::GraphImageHandle>,
        depth_target: render::GraphImageHandle,
        
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

        let brdf_integration_map = context.import_image(&self.brdf_integration_map);

        let forward_pipeline = self.forward_pipeline;
        let skybox_pipeline = self.skybox_pipeline;

        context.add_pass("forward_pass")
            .with_dependency(color_target, render::AccessKind::ColorAttachmentWrite)
            .with_dependencies(color_resolve.map(|i| (i, render::AccessKind::ColorAttachmentWrite)))
            .with_dependency(depth_target, render::AccessKind::DepthAttachmentWrite)
            .with_dependency(draw_commands, render::AccessKind::IndirectBuffer)
            .with_dependencies(directional_light.shadow_maps.map(|h| (h, render::AccessKind::FragmentShaderRead)))    
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
                        .image(environment_map.skybox);
    
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
                        .image(environment_map.irradiance)
                        .image(environment_map.prefiltered)
                } else {
                    push_constants = push_constants
                        .uint(u32::MAX)
                        .uint(u32::MAX)
                };

                push_constants = push_constants.image(&brdf_integration_map);

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

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_image(&self.brdf_integration_map);
        context.destroy_pipeline(&self.forward_pipeline);
        context.destroy_pipeline(&self.skybox_pipeline);
        context.destroy_pipeline(&self.brdf_integration_pipeline);
    }
}