use crate::{
    graphics::{self, AccessKind},
    Camera,
};
use ash::vk;
use glam::Mat4;

#[derive(Debug, Clone, Copy)]
pub struct SsaoSettings {
    pub sample_count: usize,
    pub min_radius: f32,
    pub max_radius: f32,
    pub full_res: bool,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            sample_count: 32,
            min_radius: 0.1,
            max_radius: 0.5,
            full_res: false,
        }
    }
}

impl SsaoSettings {
    pub fn edit(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("sample count");
            ui.add(egui::Slider::new(&mut self.sample_count, 1..=64));
        });

        ui.horizontal(|ui| {
            ui.label("min radius");
            ui.add(egui::DragValue::new(&mut self.min_radius).clamp_range(0.0..=1.0))
        });

        ui.horizontal(|ui| {
            ui.label("max radius");
            ui.add(egui::DragValue::new(&mut self.max_radius).clamp_range(0.0..=1.0))
        });

        ui.checkbox(&mut self.full_res, "full res");
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuSSAOInfo {
    projection_matrix: Mat4,
    inverse_projection_matrix: Mat4,
    
    ssao_resolution: [u32; 2],
    depth_texture: u32,
    ssao_image: u32,
    
    noise_image: u32,
    noise_size: u32,
    samples_image: u32,
    samples_size: u32,
    
    sample_count: u32,
    min_radius: f32,
    max_radius: f32,
    _padding: u32,
}

pub struct SsaoRenderer {
    noise_texture: graphics::Image,
    samples_texture: graphics::Image,
}

impl SsaoRenderer {
    pub fn new(context: &mut graphics::Context) -> Self {
        let noise_texture = compute_noise_texture(context);
        let samples_texture = compute_samples(context);

        Self {
            noise_texture,
            samples_texture,
        }
    }

    pub fn compute_ssao(
        &self,
        context: &mut graphics::Context,
        settings: &SsaoSettings,
        depth_buffer: graphics::GraphImageHandle,
        camera: &Camera,
        screen_resolution: [u32; 2],
    ) -> graphics::GraphImageHandle {
        let ssao_resolution = if settings.full_res {
            screen_resolution
        } else {
            screen_resolution.map(|n| n / 2)
        };

        let ssao_raw_image = context.create_transient_image(
            "ssao_raw_image",
            graphics::ImageDesc {
                ty: graphics::ImageType::Single2D,
                format: vk::Format::R8_UNORM,
                dimensions: [ssao_resolution[0], ssao_resolution[1], 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                aspect: vk::ImageAspectFlags::COLOR,
                default_sampler: Some(graphics::SamplerKind::LinearClamp),
                ..Default::default()
            },
        );

        let ssao_blur_image = context.create_transient_image(
            "ssao_blur_image",
            graphics::ImageDesc {
                ty: graphics::ImageType::Single2D,
                format: vk::Format::R8_UNORM,
                dimensions: [ssao_resolution[0], ssao_resolution[1], 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                aspect: vk::ImageAspectFlags::COLOR,
                default_sampler: Some(graphics::SamplerKind::LinearClamp),
                ..Default::default()
            },
        );

        let projection_matrix = camera.compute_projection_matrix();

        let ssao_info = GpuSSAOInfo {
            projection_matrix,
            inverse_projection_matrix: projection_matrix.inverse(),
            ssao_resolution,
            depth_texture: context.get_resource_descriptor_index(depth_buffer).unwrap(),
            ssao_image: context.get_resource_descriptor_index(ssao_raw_image).unwrap(),
            noise_image: self.noise_texture.descriptor_index().unwrap(),
            noise_size: SSAO_NOISE_SIZE as u32,
            samples_image: self.samples_texture.descriptor_index().unwrap(),
            samples_size: SSAO_SAMPLE_SIZE as u32,
            sample_count: settings.sample_count as u32,
            min_radius: settings.min_radius,
            max_radius: settings.max_radius,
            _padding: 0,
        };
        let ssao_info_buffer = context.transient_storage_data("ssao_info_buffer", bytemuck::bytes_of(&ssao_info));

        let ssao_pipeline =
            context.create_compute_pipeline("ssao_pipeline", graphics::ShaderStage::spv("shaders/ssao.comp.spv"));

        let blur_pipeline =
            context.create_compute_pipeline("ssao_pipeline", graphics::ShaderStage::spv("shaders/ssao_blur.comp.spv"));

        context
            .add_pass("ssao_pass")
            .with_dependency(depth_buffer, AccessKind::ComputeShaderRead)
            .with_dependency(ssao_raw_image, AccessKind::ComputeShaderWrite)
            .render(move |cmd, graph| {
                cmd.bind_compute_pipeline(ssao_pipeline);
                cmd.build_constants().buffer(graph.get_buffer(ssao_info_buffer));
                cmd.dispatch([ssao_resolution[0].div_ceil(8), ssao_resolution[1].div_ceil(8), 1]);
            });

        context
            .add_pass("ssao_blur_pass")
            .with_dependency(ssao_raw_image, AccessKind::ComputeShaderReadGeneral)
            .with_dependency(ssao_blur_image, AccessKind::ComputeShaderWrite)
            .render(move |cmd, graph| {
                cmd.bind_compute_pipeline(blur_pipeline);
                cmd.build_constants()
                    .uvec2(ssao_resolution)
                    .storage_image(graph.get_image(ssao_raw_image))
                    .storage_image(graph.get_image(ssao_blur_image));
                cmd.dispatch([ssao_resolution[0].div_ceil(8), ssao_resolution[1].div_ceil(8), 1]);
            });

        ssao_blur_image
    }
}

const SSAO_NOISE_SIZE: usize = 4;
const SSAO_SAMPLE_SIZE: usize = 64;

pub fn compute_noise_texture(context: &mut graphics::Context) -> graphics::Image {
    let image = context.create_image(
        "ssao_noise_image",
        &graphics::ImageDesc {
            ty: graphics::ImageType::Single2D,
            format: vk::Format::R8G8_SNORM,
            dimensions: [SSAO_NOISE_SIZE as u32, SSAO_NOISE_SIZE as u32, 1],
            mip_levels: 1,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
            default_sampler: Some(graphics::SamplerKind::NearestRepeat),
            ..Default::default()
        },
    );

    let noise_data: [i8; SSAO_NOISE_SIZE * SSAO_NOISE_SIZE * 2] = std::array::from_fn(|_| rand::random());

    context.immediate_write_image(
        &image,
        0,
        0..1,
        AccessKind::None,
        Some(AccessKind::ComputeShaderRead),
        bytemuck::bytes_of(&noise_data),
        None,
    );

    image
}

pub fn compute_samples(context: &mut graphics::Context) -> graphics::Image {
    let image = context.create_image(
        "ssao_samples_image",
        &graphics::ImageDesc {
            ty: graphics::ImageType::Single1D,
            format: vk::Format::R8G8B8A8_UNORM,
            dimensions: [SSAO_SAMPLE_SIZE as u32, 1, 1],
            mip_levels: 1,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
            default_sampler: Some(graphics::SamplerKind::NearestRepeat),
            ..Default::default()
        },
    );

    let samples_data: [u8; SSAO_SAMPLE_SIZE * 4] = std::array::from_fn(|_| rand::random());

    context.immediate_write_image(
        &image,
        0,
        0..1,
        AccessKind::None,
        Some(AccessKind::ComputeShaderRead),
        &samples_data,
        None,
    );

    image
}
