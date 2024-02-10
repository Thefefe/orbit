use ash::vk;

use crate::graphics::{self, ColorAttachmentDesc, DrawPass, LoadOp, MultisampleCount, RenderPass};

pub fn hdr_resolve(
    context: &mut graphics::Context,
    msaa_image: graphics::GraphImageHandle,
    resolve_image: graphics::GraphImageHandle,
) {
    let msaa_desc = context.get_image_desc(msaa_image);
    let resolve_desc = context.get_image_desc(resolve_image);

    assert_eq!(msaa_desc.dimensions, resolve_desc.dimensions);
    assert_eq!(msaa_desc.format, resolve_desc.format);
    assert_ne!(msaa_desc.samples, MultisampleCount::None);
    assert_eq!(resolve_desc.samples, MultisampleCount::None);

    let sample_count = msaa_desc.samples.sample_count();
    
    let pipeline = context.create_raster_pipeline(
        "hdr_resolve_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderStage::spv("shaders/utils/blit.vert.spv"))
            .fragment_shader(graphics::ShaderStage::spv("shaders/hdr_resolve.frag.spv")
                .spec_u32(0, sample_count))
            .rasterizer(graphics::RasterizerDesc {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .color_attachments(&[graphics::PipelineColorAttachment {
                format: resolve_desc.format,
                color_mask: vk::ColorComponentFlags::RGBA,
                color_blend: None,
            }]),
    );

    let mut render_pass = RenderPass::new(context, "hdr_resolve_pass")
        .color_attachments(&[ColorAttachmentDesc {
            target: resolve_image,
            resolve: None,
            load_op: LoadOp::DontCare,
            store: true,
        }]);

    DrawPass::new(&mut render_pass, pipeline)
        .push_data::<[u32; 2]>([resolve_desc.dimensions[0], resolve_desc.dimensions[1]])
        .read_image(msaa_image)
        .draw(0..3, 0..1);

    render_pass.finish();
}
