use std::borrow::Cow;

use ash::vk;
use glam::{vec3, Mat4};

use crate::{graphics, math};

pub const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const IRRADIANCE_SIZE: u32 = 32;
const PREFILTERD_SIZE: u32 = 512;
const PREFILTERED_MIP_LEVELS: u32 = 5;

#[derive(Clone, Copy)]
pub struct EnvironmentMapView<'a> {
    pub skybox: &'a graphics::ImageView,
    pub irradiance: &'a graphics::ImageView,
    pub prefiltered: &'a graphics::ImageView,
}

#[derive(Clone, Copy)]
pub struct GraphEnvironmentMap {
    pub skybox: graphics::GraphImageHandle,
    pub irradiance: graphics::GraphImageHandle,
    pub prefiltered: graphics::GraphImageHandle,
}

impl GraphEnvironmentMap {
    pub fn get<'a>(&self, graph: &'a graphics::CompiledRenderGraph) -> EnvironmentMapView<'a> {
        EnvironmentMapView {
            skybox: graph.get_image(self.skybox),
            irradiance: graph.get_image(self.irradiance),
            prefiltered: graph.get_image(self.prefiltered),
        }
    }
}

pub struct EnvironmentMap {
    pub skybox: graphics::Image,
    pub irradiance: graphics::Image,
    pub prefiltered: graphics::Image,
}

impl EnvironmentMap {
    pub fn new(
        context: &mut graphics::Context,
        name: Cow<'static, str>,
        resolution: u32,
        equirectangular_image: &graphics::ImageView,
    ) -> EnvironmentMap {
        let skybox_mip_levels = math::mip_levels_from_size(resolution);
        let skybox = context.create_image(
            format!("{name}_skybox"),
            &graphics::ImageDesc {
                ty: graphics::ImageType::Cube,
                format: vk::Format::R16G16B16A16_SFLOAT,
                dimensions: [resolution, resolution, 1],
                mip_levels: skybox_mip_levels,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                aspect: vk::ImageAspectFlags::COLOR,
                subresource_desc: graphics::ImageSubresourceViewDesc {
                    layer_mip_count: 1,
                    layer_count: 6,
                    layer_descrptors: graphics::ImageDescriptorFlags::NONE,
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let irradiance = context.create_image(
            format!("{name}_convoluted"),
            &graphics::ImageDesc {
                ty: graphics::ImageType::Cube,
                format: vk::Format::R16G16B16A16_SFLOAT,
                dimensions: [IRRADIANCE_SIZE, IRRADIANCE_SIZE, 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                aspect: vk::ImageAspectFlags::COLOR,
                subresource_desc: graphics::ImageSubresourceViewDesc {
                    layer_mip_count: 1,
                    layer_count: 6,
                    layer_descrptors: graphics::ImageDescriptorFlags::NONE,
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let prefiltered = context.create_image(
            format!("{name}_prefilterd"),
            &graphics::ImageDesc {
                ty: graphics::ImageType::Cube,
                format: vk::Format::R16G16B16A16_SFLOAT,
                dimensions: [PREFILTERD_SIZE, PREFILTERD_SIZE, 1],
                mip_levels: PREFILTERED_MIP_LEVELS,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                aspect: vk::ImageAspectFlags::COLOR,
                subresource_desc: graphics::ImageSubresourceViewDesc::default(),
                ..Default::default()
            },
        );

        let view_matrices = [
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 1.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, -1.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 1.0, 0.0)),
        ];

        let scratch_image = context.create_image(
            "scratch_image",
            &graphics::ImageDesc {
                ty: graphics::ImageType::Single2D,
                format: vk::Format::R16G16B16A16_SFLOAT,
                dimensions: [PREFILTERD_SIZE, PREFILTERD_SIZE, 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                subresource_desc: graphics::ImageSubresourceViewDesc::default(),
                ..Default::default()
            },
        );
        let equirectangular_to_cube_pipeline = context.create_raster_pipeline(
            "equirectangular_to_cube_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/unit_cube.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv("shaders/equirectangular_cube_map.frag.spv"))
                .rasterizer(graphics::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_clamp: false,
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: FORMAT,
                    ..Default::default()
                }]),
        );

        let cube_map_convolution_pipeline = context.create_raster_pipeline(
            "cubemap_convolution_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/unit_cube.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv("shaders/cubemap_convolution.frag.spv"))
                .rasterizer(graphics::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_clamp: false,
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: FORMAT,
                    ..Default::default()
                }]),
        );

        let cube_map_prefilter_pipeline = context.create_raster_pipeline(
            "cubemap_prefilter_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/unit_cube.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv(
                    "shaders/environmental_map_prefilter.frag.spv",
                ))
                .rasterizer(graphics::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_clamp: false,
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: FORMAT,
                    ..Default::default()
                }]),
        );

        context.record_and_submit(|cmd| {
            cmd.barrier(
                &[],
                &[graphics::image_barrier(
                    &skybox,
                    graphics::AccessKind::None,
                    graphics::AccessKind::ColorAttachmentWrite,
                )],
                &[],
            );

            for i in 0..6 {
                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(skybox.layer_view(0, i).unwrap().view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.5, 0.9, 1.0],
                        },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .render_area(skybox.full_rect())
                    .layer_count(1);

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(equirectangular_to_cube_pipeline);

                cmd.build_constants().mat4(&view_matrices[i as usize]).sampled_image(&equirectangular_image);

                cmd.draw(0..36, 0..1);

                cmd.end_rendering();
            }

            cmd.generate_mipmaps(
                &skybox,
                0..6,
                graphics::AccessKind::ColorAttachmentWrite,
                graphics::AccessKind::AllGraphicsRead,
            );

            cmd.barrier(
                &[],
                &[
                    graphics::image_barrier(
                        &irradiance,
                        graphics::AccessKind::None,
                        graphics::AccessKind::ColorAttachmentWrite,
                    ),
                    graphics::image_barrier(
                        &prefiltered,
                        graphics::AccessKind::None,
                        graphics::AccessKind::TransferWrite,
                    ),
                ],
                &[],
            );

            for i in 0..6 {
                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(irradiance.layer_view(0, i).unwrap().view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.5, 0.9, 1.0],
                        },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .render_area(irradiance.full_rect())
                    .layer_count(1);

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(cube_map_convolution_pipeline);

                cmd.build_constants().mat4(&view_matrices[i as usize]).sampled_image(&skybox);

                cmd.draw(0..36, 0..1);

                cmd.end_rendering();
            }

            for face in 0..6u32 {
                let mut extent = [PREFILTERD_SIZE; 2];
                for mip_level in 0..PREFILTERED_MIP_LEVELS {
                    cmd.barrier(
                        &[],
                        &[graphics::image_barrier(
                            &scratch_image,
                            graphics::AccessKind::None,
                            graphics::AccessKind::ColorAttachmentWrite,
                        )],
                        &[],
                    );
                    let color_attachment = vk::RenderingAttachmentInfo::builder()
                        .image_view(scratch_image.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.1, 0.5, 0.9, 1.0],
                            },
                        })
                        .store_op(vk::AttachmentStoreOp::STORE);

                    let rendering_info = vk::RenderingInfo::builder()
                        .color_attachments(std::slice::from_ref(&color_attachment))
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: vk::Extent2D {
                                width: extent[0],
                                height: extent[1],
                            },
                        })
                        .layer_count(1);

                    cmd.begin_rendering(&rendering_info);

                    cmd.bind_raster_pipeline(cube_map_prefilter_pipeline);

                    let roughness = mip_level as f32 / (PREFILTERED_MIP_LEVELS - 1) as f32;

                    cmd.build_constants().mat4(&view_matrices[face as usize]).sampled_image(&skybox).float(roughness);

                    cmd.draw(0..36, 0..1);

                    cmd.end_rendering();

                    cmd.barrier(
                        &[],
                        &[graphics::image_barrier(
                            &scratch_image,
                            graphics::AccessKind::ColorAttachmentWrite,
                            graphics::AccessKind::TransferRead,
                        )],
                        &[],
                    );

                    cmd.copy_image(
                        &scratch_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        &prefiltered,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::ImageCopy {
                            src_subresource: scratch_image.subresource_layers(0, ..),
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: prefiltered.subresource_layers(mip_level, face..face + 1),
                            dst_offset: vk::Offset3D::default(),
                            extent: vk::Extent3D {
                                width: extent[0],
                                height: extent[1],
                                depth: 1,
                            },
                        }],
                    );

                    extent = extent.map(math::next_mip_size);
                }
            }

            cmd.barrier(
                &[],
                &[
                    graphics::image_barrier(
                        &irradiance,
                        graphics::AccessKind::ColorAttachmentWrite,
                        graphics::AccessKind::AllGraphicsRead,
                    ),
                    graphics::image_barrier(
                        &prefiltered,
                        graphics::AccessKind::TransferWrite,
                        graphics::AccessKind::AllGraphicsRead,
                    ),
                ],
                &[],
            );
        });

        EnvironmentMap {
            skybox,
            irradiance,
            prefiltered,
        }
    }

    pub fn import_to_graph(&self, context: &mut graphics::Context) -> GraphEnvironmentMap {
        GraphEnvironmentMap {
            skybox: context.import(&self.skybox),
            irradiance: context.import(&self.irradiance),
            prefiltered: context.import(&self.prefiltered),
        }
    }
}
