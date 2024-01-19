use ash::vk;

use crate::{graphics::{self, AccessKind}, Settings};

use super::forward::RenderMode;

pub fn render_post_process(
    context: &mut graphics::Context,
    src_image: graphics::GraphImageHandle,
    bloom_image: Option<graphics::GraphImageHandle>,
    settings: &Settings,
    depth_pyramid_debug: Option<(graphics::GraphImageHandle, u32, f32)>,
    render_mode: RenderMode,
) {
    let swapchain_image = context.get_swapchain_image();
    let pipeline = context.create_raster_pipeline(
        "blit_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/blit.vert.spv"))
            .fragment_shader(graphics::ShaderSource::spv("shaders/post_process.frag.spv"))
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

    let exposure = settings.camera_exposure;
    let bloom_intensity = settings.bloom_settings.intensity;

    context
        .add_pass("blit_present_pass")
        .with_dependency(swapchain_image, AccessKind::ColorAttachmentWrite)
        .with_dependency(src_image, AccessKind::FragmentShaderRead)
        .with_dependencies(bloom_image.map(|h| (h, AccessKind::FragmentShaderReadGeneral)))
        .with_dependencies(
            depth_pyramid_debug.map(|(pyramid, _, _)| (pyramid, AccessKind::FragmentShaderReadGeneral)),
        )
        .render(move |cmd, graph| {
            let swapchain_image = graph.get_image(swapchain_image);
            let src_image = graph.get_image(src_image);
            let bloom_image = bloom_image.map(|i| graph.get_image(i));

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

            let mut constants = cmd.build_constants()
                .uint(render_mode as u32)
                .float(exposure)
                .float(bloom_intensity)
                .sampled_image(&src_image);
            
            if let Some(bloom_image) = bloom_image {
                constants = constants.sampled_image(bloom_image);
            } else {
                constants = constants.uint(u32::MAX);
            }

            if let Some((depth_pyramid, depth_pyramid_level, far_depth)) = depth_pyramid_debug {
                let depth_pyramid = graph.get_image(depth_pyramid);
                constants = constants.sampled_image(depth_pyramid).uint(depth_pyramid_level).float(far_depth);
            } else {
                constants = constants.uint(u32::MAX);
            }

            constants.push();

            cmd.draw(0..6, 0..1);

            cmd.end_rendering();
        });
}
