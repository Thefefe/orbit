use ash::vk;

use crate::{
    app::Settings,
    graphics::{self, ColorAttachmentDesc, DrawPass, LoadOp, RenderPass},
};

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
            .vertex_shader(graphics::ShaderSource::spv("shaders/utils/blit.vert.spv"))
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

    let mut render_pass = RenderPass::new(context, "post_process_tonemap").color_attachments(&[ColorAttachmentDesc {
        target: swapchain_image,
        resolve: None,
        load_op: LoadOp::DontCare,
        store: true,
    }]);

    let (pyramid_mip_level, pyramid_far_depth) = depth_pyramid_debug.map_or((0, 0.0), |(_, l, d)| (l, d));

    DrawPass::new(&mut render_pass, pipeline)
        .push_data::<u32>(render_mode as u32)
        .push_data::<f32>(exposure)
        .push_data::<f32>(bloom_intensity)
        .read_image(src_image)
        .read_image_general(bloom_image)
        .read_image_general(depth_pyramid_debug.map(|(i, _, _)| i))
        .push_data::<u32>(pyramid_mip_level)
        .push_data::<f32>(pyramid_far_depth)
        .draw(0..3, 0..1);

    render_pass.finish();
}
