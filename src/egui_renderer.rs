use std::{ops::Range, collections::HashMap};

use crate::{render::{self, DescriptorHandle}, utils::load_spv};
use ash::vk;
use gpu_allocator::MemoryLocation;

struct ClippedBatch {
    clip_rect: vk::Rect2D,
    texture: render::ResourceHandle,
    index_range: Range<u32>,
    vertex_offset: i32,
}

pub struct EguiRenderer {
    pipeline: render::RasterPipeline,
    vertex_buffer: render::Buffer,
    index_buffer: render::Buffer,
    textures: HashMap<egui::TextureId, render::Image>,
}

impl EguiRenderer {
    const MAX_VERTEX_COUNT: usize = 200_000;
    const MAX_INDEX_COUNT: usize = 200_000;
    const IMAGE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

    pub fn new(context: &render::Context) -> Self {
        let pipeline = {
            let (vertex_shader, fragment_shader) = {
                let vertex_spv = load_spv("shaders/egui/egui.vert.spv").expect("failed to load shader");
                let fragment_spv = load_spv("shaders/egui/egui.frag.spv").expect("failed to load shader");

                (
                    context.create_shader_module(&vertex_spv, "egui_vertex_shader"),
                    context.create_shader_module(&fragment_spv, "egui_fragment_shader"),
                )
            };

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("egui_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_shader,
                    entry,
                },
                fragment_stage: render::ShaderStage {
                    module: fragment_shader,
                    entry,
                },
                vertex_input: render::VertexInput {
                    bindings: &[
                        vk::VertexInputBindingDescription {
                            binding: 0,
                            stride: std::mem::size_of::<egui::epaint::Vertex>() as u32,
                            input_rate: vk::VertexInputRate::VERTEX,
                        }
                    ],
                    attributes: &[
                        vk::VertexInputAttributeDescription {
                            location: 0,
                            binding: 0,
                            format: vk::Format::R32G32_SFLOAT,
                            offset: bytemuck::offset_of!(egui::epaint::Vertex, pos) as u32,
                        },
                        vk::VertexInputAttributeDescription {
                            location: 1,
                            binding: 0,
                            format: vk::Format::R32G32_SFLOAT,
                            offset: bytemuck::offset_of!(egui::epaint::Vertex, uv) as u32,
                        },
                        vk::VertexInputAttributeDescription {
                            location: 2,
                            binding: 0,
                            format: vk::Format::R8G8B8A8_UNORM,
                            offset: bytemuck::offset_of!(egui::epaint::Vertex, color) as u32,
                        },
                    ]
                },
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: context.swapchain.format(),
                    color_blend: Some(render::ColorBlendState {
                        src_color_blend_factor: vk::BlendFactor::ONE,
                        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        color_blend_op: vk::BlendOp::ADD,
                        src_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_DST_ALPHA,
                        dst_alpha_blend_factor: vk::BlendFactor::ONE,
                        alpha_blend_op: vk::BlendOp::ADD,
                        
                    }),
                    ..Default::default()
                }],
                depth_state: None,
                multisample: render::MultisampleCount::None,
            });

            context.destroy_shader_module(vertex_shader);
            context.destroy_shader_module(fragment_shader);

            pipeline
        };

        let vertex_buffer_size = std::mem::size_of::<egui::epaint::Vertex>() * Self::MAX_VERTEX_COUNT;
        let vertex_buffer = context.create_buffer("egui_vertex_buffer", &render::BufferDesc {
            size: vertex_buffer_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
        });

        let index_buffer_size = std::mem::size_of::<u32>() * Self::MAX_VERTEX_COUNT;
        let index_buffer = context.create_buffer("egui_index_buffer", &render::BufferDesc {
            size: index_buffer_size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            textures: HashMap::new(),
        }
    }

    fn update(
        &mut self,
        context: &mut render::FrameContext,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
    ) -> Vec<ClippedBatch> {
        puffin::profile_function!();
        for (id, delta) in textures_delta.set.iter() {
            self.update_texture(&context, *id, delta);
        }

        for id in textures_delta.free.iter() {
            if let Some(image) = self.textures.remove(id) {
                context.destroy_image(&image);
            }
        }

        let mut clipped_batches = Vec::new();

        let mut vertex_cursor: u32 = 0;
        let mut index_cursor: u32 = 0;

        for clipped_primitive in clipped_primitives {
            if let egui::epaint::Primitive::Mesh(ref mesh) = clipped_primitive.primitive {
                if vertex_cursor as usize + mesh.vertices.len() > Self::MAX_VERTEX_COUNT ||
                   index_cursor as usize + mesh.indices.len() > Self::MAX_INDEX_COUNT {
                    log::error!("egui buffers are full, but there are to draw");
                    break;
                }

                let vertices = unsafe {
                    std::slice::from_raw_parts(
                        mesh.vertices.as_ptr().cast::<u8>(),
                        mesh.vertices.len() * std::mem::size_of::<egui::epaint::Vertex>(),
                    )
                };

                let rect = clipped_primitive.clip_rect;
                let clip_rect = vk::Rect2D {
                    offset: vk::Offset2D {
                        x: rect.min.x.round() as i32,
                        y: rect.min.y.round() as i32,
                    },
                    extent: vk::Extent2D {
                        width: rect.width().round() as u32,
                        height: rect.height().round() as u32,
                    },
                };

                let image = self.textures.get(&mesh.texture_id).unwrap();
                let texture = context.import_image(
                    "egui_image",
                    image,
                    &render::GraphResourceImportDesc {
                        initial_access: render::AccessKind::FragmentShaderRead,
                        target_access: render::AccessKind::FragmentShaderRead,
                        ..Default::default()
                    },
                );

                clipped_batches.push(ClippedBatch {
                    clip_rect,
                    texture,
                    index_range: index_cursor..index_cursor + mesh.indices.len() as u32,
                    vertex_offset: vertex_cursor as i32,
                });

                context.immediate_write_buffer(
                    &self.vertex_buffer,
                    vertices,
                    vertex_cursor as usize * std::mem::size_of::<egui::epaint::Vertex>()
                );
                context.immediate_write_buffer(
                    &self.index_buffer,
                    bytemuck::cast_slice(&mesh.indices),
                    index_cursor as usize * std::mem::size_of::<u32>(),
                );

                vertex_cursor += mesh.vertices.len() as u32;
                index_cursor += mesh.indices.len() as u32;
            }
        }

        clipped_batches
    }

    pub fn update_texture(&mut self, context: &render::Context, id: egui::TextureId, delta: &egui::epaint::ImageDelta) {
        puffin::profile_function!();
        let [width, height] = delta.image.size();
        let bytes: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "mismatch between texture size and texel count"
                );
                image.pixels.iter().flat_map(|color| color.to_array()).collect()
            }
            egui::ImageData::Font(image) => image.srgba_pixels(None).flat_map(|color| color.to_array()).collect(),
        };

        let (image, is_new) = if let Some(image) = self.textures.get(&id) {
            (*image, false)
        } else {
            let (source, number) = match id {
                egui::TextureId::Managed(number) => ("managed", number),
                egui::TextureId::User(number) => ("user", number),
            };

            let image = context.create_image(&format!("egui_image_{source}_{number}"), &render::ImageDesc {
                format: Self::IMAGE_FORMAT,
                width: width as u32,
                height: height as u32,
                mip_levels: 1,
                samples: render::MultisampleCount::None,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                aspect: vk::ImageAspectFlags::COLOR,
            });

            self.textures.insert(id, image);
            (image, true)
        };

        let offset = delta
            .pos
            .map(|[x, y]| vk::Offset2D {
                x: x as i32,
                y: y as i32,
            })
            .unwrap_or_default();
        let subregion = vk::Rect2D {
            offset,
            extent: vk::Extent2D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
            },
        };

        let prev_access = if is_new {
            render::AccessKind::None
        } else {
            render::AccessKind::FragmentShaderRead
        };

        context.immediate_write_image(
            &image,
            0,
            0..1,
            prev_access,
            Some(render::AccessKind::FragmentShaderRead),
            &bytes,
            Some(subregion),
        );
    }

    pub fn render(
        &mut self,
        context: &mut render::FrameContext,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
        target_image: render::ResourceHandle
    ) {
        puffin::profile_function!();

        let clipped_batches = self.update(context, clipped_primitives, textures_delta);

        if clipped_batches.is_empty() {return;}

        let window_size: [u32; 2] = context.window().inner_size().into();
        let screen_size = window_size.map(|n| n as f32);
        let pipeline = self.pipeline;

        let index_buffer = context.import_buffer(
            "egui_index_buffer",
            &self.index_buffer,
            &render::GraphResourceImportDesc::default(),
        );

        let vertex_buffer = context.import_buffer(
            "egui_vertex_buffer",
            &self.vertex_buffer,
            &render::GraphResourceImportDesc::default(),
        );

        context.add_pass(
            "egui_draw",
            &[(target_image, render::AccessKind::ColorAttachmentWrite)],
            move |cmd, graph| {
                let target_image = graph.get_image(target_image).unwrap();
                
                let index_buffer = graph.get_buffer(index_buffer).unwrap();
                let vertex_buffer = graph.get_buffer(vertex_buffer).unwrap();

                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(target_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(target_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(pipeline);
                cmd.bind_index_buffer(index_buffer);
                cmd.bind_vertex_buffer(0, vertex_buffer, 0);

                for batch in clipped_batches.iter() {
                    cmd.set_scissor(0, &[batch.clip_rect]);

                    let texture = graph.get_image(batch.texture).unwrap();

                    cmd.push_bindings(&[
                        bytemuck::cast(screen_size[0]),
                        bytemuck::cast(screen_size[1]),
                        texture.descriptor_index.unwrap().to_raw(),
                    ]);
                    cmd.draw_indexed(batch.index_range.clone(), 0..1, batch.vertex_offset);                    
                }

                cmd.end_rendering();
            },
        );
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
        context.destroy_buffer(&self.index_buffer);
        context.destroy_buffer(&self.vertex_buffer);
        for image in self.textures.values() {
            context.destroy_image(image);
        }
    }
}