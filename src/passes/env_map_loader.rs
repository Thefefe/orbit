use std::borrow::Cow;

use ash::vk;
use glam::{vec3, Mat4};

use crate::{render, utils, gltf_loader};

#[derive(Clone, Copy)]
pub struct EnvironmentMapView<'a> {
    pub skybox: &'a render::ImageView,
    pub irradiance: &'a render::ImageView,
    pub prefiltered: &'a render::ImageView,
}

#[derive(Clone, Copy)]
pub struct GraphEnvironmentMap {
    pub skybox: render::GraphImageHandle,
    pub irradiance: render::GraphImageHandle,
    pub prefiltered: render::GraphImageHandle,
}

impl GraphEnvironmentMap {
    pub fn get<'a>(&self, graph: &'a render::CompiledRenderGraph) -> EnvironmentMapView<'a> {
        EnvironmentMapView {
            skybox: graph.get_image(self.skybox),
            irradiance: graph.get_image(self.irradiance),
            prefiltered: graph.get_image(self.prefiltered),
        }
    }
}

pub struct EnvironmentMap {
    pub skybox: render::Image,
    pub irradiance: render::Image,
    pub prefiltered: render::Image,
}

impl EnvironmentMap {
    pub fn import_to_graph(&self, context: &mut render::Context) -> GraphEnvironmentMap {
        GraphEnvironmentMap {
            skybox: context.import_image(&self.skybox),
            irradiance: context.import_image(&self.irradiance),
            prefiltered: context.import_image(&self.prefiltered),
        }
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_image(&self.skybox);
        context.destroy_image(&self.irradiance);
        context.destroy_image(&self.prefiltered);
    }
}

pub struct EnvironmentMapLoader {
    equirectangular_to_cube_pipeline: render::RasterPipeline,
    cube_map_convolution_pipeline: render::RasterPipeline,
    cube_map_prefilter_pipeline: render::RasterPipeline,
}

impl EnvironmentMapLoader {
    pub const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
    const IRRADIANCE_SIZE: u32 = 32;
    const PREFILTERD_SIZE: u32 = 512;
    const PREFILTERED_MIP_LEVELS: u32 = 5;

    pub fn new(context: &render::Context) -> Self {
        let vertex_shader = utils::load_spv("shaders/unit_cube.vert.spv").unwrap();
        let vertex_module = context
            .create_shader_module(&vertex_shader, "equirectangular_cube_map_vertex_shader");

        let equirectangular_to_cube_pipeline = {
            let fragment_shader = utils::load_spv("shaders/equirectangular_cube_map.frag.spv").unwrap();

            let fragment_module = context
                .create_shader_module(&fragment_shader, "equirectangular_cube_map_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("equirectangular_to_cube_pipeline", &render::RasterPipelineDesc {
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
                    format: Self::FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: None,
                multisample_state: Default::default(),
                dynamic_states: &[],
            });

            context.destroy_shader_module(fragment_module);

            pipeline
        };

        let cube_map_convolution_pipeline = {
            let fragment_shader = utils::load_spv("shaders/cubemap_convolution.frag.spv").unwrap();

            let fragment_module = context
                .create_shader_module(&fragment_shader, "cubemap_convolution_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("cubemap_convolution_pipeline", &render::RasterPipelineDesc {
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
                    format: Self::FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: None,
                multisample_state: Default::default(),
                dynamic_states: &[],
            });
            
            context.destroy_shader_module(fragment_module);
            
            pipeline
        };

        let cube_map_prefilter_pipeline = {
            let fragment_shader = utils::load_spv("shaders/environmental_map_prefilter.frag.spv").unwrap();

            let fragment_module = context
                .create_shader_module(&fragment_shader, "cubemap_convolution_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("cubemap_convolution_pipeline", &render::RasterPipelineDesc {
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
                    format: Self::FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: None,
                multisample_state: Default::default(),
                dynamic_states: &[],
            });

            context.destroy_shader_module(fragment_module);

            pipeline
        };

        context.destroy_shader_module(vertex_module);

        Self { equirectangular_to_cube_pipeline, cube_map_convolution_pipeline, cube_map_prefilter_pipeline  }
    }

    pub fn create_from_equirectangular_image(
        &self,
        context: &render::Context,
        name: Cow<'static, str>,
        resolution: u32,
        equirectangular_image: &render::ImageView,
    ) -> EnvironmentMap {
        let skybox_mip_levels = gltf_loader::mip_levels_from_size(resolution);
        let skybox = context.create_image(format!("{name}_skybox"), &render::ImageDesc {
            ty: render::ImageType::Cube,
            format: vk::Format::R16G16B16A16_SFLOAT,
            dimensions: [resolution, resolution, 1],
            mip_levels: skybox_mip_levels,
            samples: render::MultisampleCount::None,
            usage:
                vk::ImageUsageFlags::SAMPLED |
                vk::ImageUsageFlags::COLOR_ATTACHMENT |
                vk::ImageUsageFlags::TRANSFER_SRC |
                vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        let irradiance = context.create_image(format!("{name}_convoluted"), &render::ImageDesc {
            ty: render::ImageType::Cube,
            format: vk::Format::R16G16B16A16_SFLOAT,
            dimensions: [Self::IRRADIANCE_SIZE, Self::IRRADIANCE_SIZE, 1],
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        let prefiltered = context.create_image(format!("{name}_prefilterd"), &render::ImageDesc {
            ty: render::ImageType::Cube,
            format: vk::Format::R16G16B16A16_SFLOAT,
            dimensions: [Self::PREFILTERD_SIZE, Self::PREFILTERD_SIZE, 1],
            mip_levels: Self::PREFILTERED_MIP_LEVELS,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        let view_matrices = [
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3( 1.0,  0.0,  0.0), vec3(0.0,  1.0,  0.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3(-1.0,  0.0,  0.0), vec3(0.0,  1.0,  0.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3( 0.0, -1.0,  0.0), vec3(0.0,  0.0,  1.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3( 0.0,  1.0,  0.0), vec3(0.0,  0.0, -1.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3( 0.0,  0.0,  1.0), vec3(0.0,  1.0,  0.0)),
            Mat4::look_at_rh(vec3(0.0, 0.0, 0.0), vec3( 0.0,  0.0, -1.0), vec3(0.0,  1.0,  0.0)),
        ];

        let scratch_image = context.create_image("scratch_image", &render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: vk::Format::R16G16B16A16_SFLOAT,
            dimensions: [Self::PREFILTERD_SIZE, Self::PREFILTERD_SIZE, 1],
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        context.record_and_submit(|cmd| {
            cmd.barrier(&[], &[
                render::image_barrier(&skybox, render::AccessKind::None, render::AccessKind::ColorAttachmentWrite)
            ], &[]);

            for i in 0..6 {
                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(skybox.layer_views[i])
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {                    
                            color: vk::ClearColorValue {
                            float32: [0.1, 0.5, 0.9, 1.0],
                        }
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .render_area(skybox.full_rect())
                    .layer_count(1);

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(self.equirectangular_to_cube_pipeline);

                cmd.build_constants()
                    .mat4(&view_matrices[i])
                    .image(&equirectangular_image);

                cmd.draw(0..36, 0..1);

                cmd.end_rendering();
            }

            cmd.generate_mipmaps(
                &skybox,
                0..6,
                render::AccessKind::ColorAttachmentWrite,
                render::AccessKind::AllGraphicsRead
            );
            
            cmd.barrier(&[], &[
                // render::image_barrier(
                //     &skybox,
                //     render::AccessKind::ColorAttachmentWrite,
                //     render::AccessKind::AllGraphicsRead,
                // ),
                render::image_barrier(
                    &irradiance,
                    render::AccessKind::None,
                    render::AccessKind::ColorAttachmentWrite,
                ),
                render::image_barrier(
                    &prefiltered,
                    render::AccessKind::None,
                    render::AccessKind::TransferWrite,
                ),
            ], &[]);

            for i in 0..6 {
                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(irradiance.layer_views[i])
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.5, 0.9, 1.0],
                        }
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .render_area(irradiance.full_rect())
                    .layer_count(1);

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(self.cube_map_convolution_pipeline);
                
                cmd.build_constants()
                    .mat4(&view_matrices[i])
                    .image(&skybox);

                cmd.draw(0..36, 0..1);

                cmd.end_rendering();
            }

            for face in 0..6u32 {
                let mut extent = [Self::PREFILTERD_SIZE; 2];
                for mip_level in 0..Self::PREFILTERED_MIP_LEVELS {
                    cmd.barrier(&[], &[render::image_barrier(
                        &scratch_image,
                        render::AccessKind::None,
                        render::AccessKind::ColorAttachmentWrite
                    )], &[]);
                    let color_attachment = vk::RenderingAttachmentInfo::builder()
                        .image_view(scratch_image.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.1, 0.5, 0.9, 1.0],
                            }
                        })
                        .store_op(vk::AttachmentStoreOp::STORE);
    
                    let rendering_info = vk::RenderingInfo::builder()
                        .color_attachments(std::slice::from_ref(&color_attachment))
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: vk::Extent2D { width: extent[0], height: extent[1] }
                        })
                        .layer_count(1);
    
                    cmd.begin_rendering(&rendering_info);
    
                    cmd.bind_raster_pipeline(self.cube_map_prefilter_pipeline);
                    
                    let roughness = mip_level as f32 / (Self::PREFILTERED_MIP_LEVELS - 1) as f32;

                    cmd.build_constants()
                        .mat4(&view_matrices[face as usize])
                        .image(&skybox)
                        .float(roughness);
    
                    cmd.draw(0..36, 0..1);
    
                    cmd.end_rendering();
                    
                    cmd.barrier(&[], &[render::image_barrier(
                        &scratch_image,
                        render::AccessKind::ColorAttachmentWrite,
                        render::AccessKind::TransferRead,
                    )], &[]);

                    cmd.copy_image(
                        &scratch_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        &prefiltered,   vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::ImageCopy {
                            src_subresource: scratch_image.subresource_layers(0, ..),
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: prefiltered.subresource_layers(mip_level, face..face+1),
                            dst_offset: vk::Offset3D::default(),
                            extent: vk::Extent3D {
                                width: extent[0],
                                height: extent[1],
                                depth: 1,
                            },
                        }],
                    );

                    extent = extent.map(gltf_loader::next_mip_size);
                }
            }
            
            cmd.barrier(&[], &[
                render::image_barrier(
                    &irradiance,
                    render::AccessKind::ColorAttachmentWrite,
                    render::AccessKind::AllGraphicsRead,
                ),
                render::image_barrier(
                    &prefiltered,
                    render::AccessKind::TransferWrite,
                    render::AccessKind::AllGraphicsRead,
                ),
            ], &[]);
        });

        context.destroy_image(&scratch_image);

        EnvironmentMap {
            skybox,
            irradiance,
            prefiltered,
        }
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.equirectangular_to_cube_pipeline);
        context.destroy_pipeline(&self.cube_map_convolution_pipeline);
        context.destroy_pipeline(&self.cube_map_prefilter_pipeline);
    }
}