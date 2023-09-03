use ash::vk;

use crate::{graphics, utils};

pub struct ScreenPostProcess {
    pipeline: graphics::RasterPipeline,
}

impl ScreenPostProcess {
    pub fn new(context: &graphics::Context) -> Self {
        let pipeline = {
            let vertex_shader = utils::load_spv("shaders/blit.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/blit.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "blit_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "blit_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("blit_pipeline", &graphics::RasterPipelineDesc {
                vertex_stage: graphics::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: Some(graphics::ShaderStage {
                    module: fragment_module,
                    entry,
                }),
                rasterizer: graphics::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[graphics::PipelineColorAttachment {
                    format: context.swapchain.format(),
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: None,
                multisample_state: Default::default(),
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        Self { pipeline }
    }

    pub fn render(
        &self,
        context: &mut graphics::Context,
        src_image: graphics::GraphImageHandle,
        exposure: f32,
    ) {
        let swapchain_image = context.get_swapchain_image();
        let pipeline = self.pipeline;

        context.add_pass("blit_present_pass")
            .with_dependency(swapchain_image, graphics::AccessKind::ColorAttachmentWrite)
            .with_dependency(src_image, graphics::AccessKind::FragmentShaderRead)
            .render(move |cmd, graph| {
                let swapchain_image = graph.get_image(swapchain_image);
                let src_image = graph.get_image(src_image);

                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(swapchain_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(swapchain_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));
            
                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(pipeline);

                cmd.build_constants()
                    .sampled_image(&src_image)
                    .float(exposure);

                cmd.draw(0..6, 0..1);

                cmd.end_rendering();
            });
    }

    pub fn destroy(&self, context: &graphics::Context) {
        context.destroy_pipeline(&self.pipeline);
    }
}