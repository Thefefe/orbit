use ash::vk;
use glam::Vec4;

use crate::{
    graphics::{self, AccessKind, ShaderStage},
    math,
};

const MAX_MIP_LEVEL: u32 = 6;

#[derive(Debug, Clone, Copy)]
pub struct BloomSettings {
    pub intensity: f32,
    pub filter_radius: f32,
    pub threshold: f32,
    pub soft_threshold: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            intensity: 0.025,
            filter_radius: 0.003,
            threshold: 0.0,
            soft_threshold: 0.0,
        }
    }
}

impl BloomSettings {
    pub fn edit(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("intensity");
            ui.add(egui::DragValue::new(&mut self.intensity).clamp_range(0.0..=1.0).speed(0.001));
        });

        ui.horizontal(|ui| {
            ui.label("filter radius");
            ui.add(egui::DragValue::new(&mut self.filter_radius).clamp_range(0.001..=0.025).speed(0.001));
        });

        ui.horizontal(|ui| {
            ui.label("threshold");
            ui.add(egui::DragValue::new(&mut self.threshold).clamp_range(0.0..=16.0).speed(0.01));
        });

        ui.horizontal(|ui| {
            ui.label("soft threshold");
            ui.add(egui::DragValue::new(&mut self.soft_threshold).clamp_range(0.0..=1.0).speed(0.01));
        });
    }
}

pub fn compute_bloom(
    context: &mut graphics::Context,
    settings: &BloomSettings,
    color_image: graphics::GraphImageHandle,
) -> graphics::GraphImageHandle {
    let input_image_desc = context.get_image_desc(color_image);
    let mip_levels = math::mip_levels_from_size(input_image_desc.dimensions[0].max(input_image_desc.dimensions[1]))
        .min(MAX_MIP_LEVEL);

    let bloom_image = context.create_transient_image(
        "bloom_downsample_image",
        graphics::ImageDesc {
            ty: graphics::ImageType::Single2D,
            format: vk::Format::B10G11R11_UFLOAT_PACK32,
            dimensions: input_image_desc.dimensions,
            mip_levels,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: graphics::ImageSubresourceViewDesc {
                mip_count: mip_levels,
                mip_descriptors: graphics::ImageDescriptorFlags::STORAGE | graphics::ImageDescriptorFlags::SAMPLED,
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let downsample_pipeline = context.create_compute_pipeline(
        "bloom_downsample_pipeline",
        ShaderStage::spv("shaders/bloom_downsample.comp.spv"),
    );

    let upsample_pipeline = context.create_compute_pipeline(
        "bloom_upsample_pipeline",
        ShaderStage::spv("shaders/bloom_upsample.comp.spv"),
    );

    let filter_radius = settings.filter_radius;
    let threshold = settings.threshold;
    let soft_threshold = settings.soft_threshold;

    context
        .add_pass("bloom_pass")
        .with_dependency(bloom_image, AccessKind::ComputeShaderWrite)
        .with_dependency(color_image, AccessKind::ComputeShaderWrite)
        .render(move |cmd, graph| {
            let color_image = graph.get_image(color_image);
            let downsample_image = graph.get_image(bloom_image);

            let mip_count = downsample_image.mip_view_count();

            let knee = threshold * soft_threshold;
            let mut threshold_filter = Vec4::ZERO;
            threshold_filter.x = threshold;
            threshold_filter.y = threshold_filter.x - knee;
            threshold_filter.z = 2.0 * knee;
            threshold_filter.w = 0.25 / (knee + 0.00001);

            cmd.bind_compute_pipeline(downsample_pipeline);
            for mip_level in 0..mip_count {
                let src_view = if mip_level == 0 {
                    color_image.full_view
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

                    downsample_image.mip_view(mip_level - 1).unwrap()
                };
                let dst_view = downsample_image.mip_view(mip_level).unwrap();

                cmd.build_constants()
                    .uint(dst_view.width())
                    .uint(dst_view.height())
                    .sampled_image(&src_view)
                    .storage_image(&dst_view)
                    .vec4(threshold_filter)
                    .uint(mip_level as u32);
                cmd.dispatch([dst_view.width().div_ceil(8), dst_view.height().div_ceil(8), 1]);
            }

            cmd.bind_compute_pipeline(upsample_pipeline);
            for i in 0..mip_count - 1 {
                let src_mip_level = mip_count - 1 - i;

                let src_view = downsample_image.mip_view(src_mip_level).unwrap();
                let dst_view = downsample_image.mip_view(src_mip_level - 1).unwrap();

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

                cmd.build_constants()
                    .uint(dst_view.width())
                    .uint(dst_view.height())
                    .sampled_image(&src_view)
                    .storage_image(&dst_view)
                    .float(filter_radius);
                cmd.dispatch([dst_view.width().div_ceil(8), dst_view.height().div_ceil(8), 1]);
            }
        });

    bloom_image
}
