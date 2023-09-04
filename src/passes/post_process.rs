use ash::vk;

use crate::graphics;

pub struct ScreenPostProcess {
    pipeline: graphics::RasterPipeline,
}

impl ScreenPostProcess {
    pub fn new(context: &mut graphics::Context) -> Self {
        let pipeline = context.create_raster_pipeline(
            "blit_pipeline",
            &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/blit.vert.spv"))
            .fragment_shader(graphics::ShaderSource::spv("shaders/blit.frag.spv"))
            .rasterizer(graphics::RasterizerDesc {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .color_attachments(&[graphics::PipelineColorAttachment {
                format: context.swapchain.format(),
                color_mask: vk::ColorComponentFlags::RGBA,
                color_blend: None,
            }]),
        );

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