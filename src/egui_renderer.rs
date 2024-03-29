use std::{collections::HashMap, ops::Range};

use crate::graphics;
use ash::vk;
use gpu_allocator::MemoryLocation;

struct ClippedBatch {
    clip_rect: vk::Rect2D,
    texture: graphics::GraphImageHandle,
    index_range: Range<u32>,
    vertex_offset: i32,
}

pub struct EguiRenderer {
    vertex_buffer: graphics::Buffer,
    index_buffer: graphics::Buffer,
    textures: HashMap<u64, graphics::Image>,
}

impl EguiRenderer {
    const MAX_VERTEX_COUNT: usize = 200_000;
    const MAX_INDEX_COUNT: usize = 200_000;
    const IMAGE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

    const PER_FRAME_VERTEX_BYTE_SIZE: usize = Self::MAX_VERTEX_COUNT * std::mem::size_of::<egui::epaint::Vertex>();
    const PER_FRAME_INDEX_BYTE_SIZE: usize = Self::MAX_INDEX_COUNT * std::mem::size_of::<u32>();

    pub fn new(context: &mut graphics::Context) -> Self {
        let vertex_buffer = context.create_buffer(
            "egui_vertex_buffer",
            &graphics::BufferDesc {
                size: Self::PER_FRAME_VERTEX_BYTE_SIZE * graphics::FRAME_COUNT,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: MemoryLocation::CpuToGpu,
            },
        );

        let index_buffer = context.create_buffer(
            "egui_index_buffer",
            &graphics::BufferDesc {
                size: Self::PER_FRAME_INDEX_BYTE_SIZE * graphics::FRAME_COUNT,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                memory_location: MemoryLocation::CpuToGpu,
            },
        );

        Self {
            vertex_buffer,
            index_buffer,
            textures: HashMap::new(),
        }
    }

    fn update(
        &mut self,
        context: &mut graphics::Context,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
    ) -> Vec<ClippedBatch> {
        puffin::profile_function!();
        for (id, delta) in textures_delta.set.iter() {
            self.update_texture(&context, *id, delta);
        }

        for id in textures_delta.free.iter() {
            if let egui::TextureId::Managed(id) = id {
                self.textures.remove(id);
            }
        }

        let mut clipped_batches = Vec::new();

        let mut vertex_cursor: u32 = 0;
        let mut index_cursor: u32 = 0;

        for clipped_primitive in clipped_primitives {
            if let egui::epaint::Primitive::Mesh(ref mesh) = clipped_primitive.primitive {
                if vertex_cursor as usize + mesh.vertices.len() > Self::MAX_VERTEX_COUNT
                    || index_cursor as usize + mesh.indices.len() > Self::MAX_INDEX_COUNT
                {
                    log::error!("egui buffers are full, but there are more to draw");
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

                let texture = match mesh.texture_id {
                    egui::TextureId::Managed(id) => context.import_with(
                        "egui_image",
                        self.textures.get(&id).unwrap(),
                        graphics::GraphResourceImportDesc {
                            initial_access: graphics::AccessKind::FragmentShaderRead,
                            target_access: graphics::AccessKind::FragmentShaderRead,
                            ..Default::default()
                        },
                    ),
                    egui::TextureId::User(handle) => graphics::GraphHandle {
                        resource_index: handle as usize,
                        _phantom: std::marker::PhantomData,
                    },
                };

                clipped_batches.push(ClippedBatch {
                    clip_rect,
                    texture,
                    index_range: index_cursor..index_cursor + mesh.indices.len() as u32,
                    vertex_offset: vertex_cursor as i32,
                });

                context.immediate_write_buffer(
                    &self.vertex_buffer,
                    Self::PER_FRAME_VERTEX_BYTE_SIZE * context.frame_index()
                        + vertex_cursor as usize * std::mem::size_of::<egui::epaint::Vertex>(),
                    vertices,
                );
                context.immediate_write_buffer(
                    &self.index_buffer,
                    Self::PER_FRAME_INDEX_BYTE_SIZE * context.frame_index()
                        + index_cursor as usize * std::mem::size_of::<u32>(),
                    bytemuck::cast_slice(&mesh.indices),
                );

                vertex_cursor += mesh.vertices.len() as u32;
                index_cursor += mesh.indices.len() as u32;
            }
        }

        clipped_batches
    }

    pub fn update_texture(
        &mut self,
        context: &graphics::Context,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
    ) {
        puffin::profile_function!();

        let egui::TextureId::Managed(id) = id else {
            return;
        };

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
            (image.full_view, false)
        } else {
            let image = context.create_image(
                format!("egui_image_{id}"),
                &graphics::ImageDesc {
                    ty: graphics::ImageType::Single2D,
                    format: Self::IMAGE_FORMAT,
                    dimensions: [width as u32, height as u32, 1],
                    mip_levels: 1,
                    samples: graphics::MultisampleCount::None,
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                    aspect: vk::ImageAspectFlags::COLOR,
                    subresource_desc: graphics::ImageSubresourceViewDesc::default(),
                    ..Default::default()
                },
            );
            let image_view = image.full_view;
            self.textures.insert(id, image);
            (image_view, true)
        };

        let offset = delta
            .pos
            .map(|[x, y]| vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: 0,
            })
            .unwrap_or_default();
        let extent = vk::Extent3D {
            width: delta.image.width() as u32,
            height: delta.image.height() as u32,
            depth: 1,
        };

        let prev_access = if is_new {
            graphics::AccessKind::None
        } else {
            graphics::AccessKind::FragmentShaderRead
        };

        context.immediate_write_image(
            &image,
            0,
            0..1,
            prev_access,
            Some(graphics::AccessKind::FragmentShaderRead),
            &bytes,
            Some((offset, extent)),
        );
    }

    pub fn render(
        &mut self,
        context: &mut graphics::Context,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
        target_image: graphics::GraphHandle<graphics::ImageRaw>,
    ) {
        puffin::profile_function!();

        let clipped_batches = self.update(context, clipped_primitives, textures_delta);

        if clipped_batches.is_empty() {
            return;
        }

        let window_size: [u32; 2] = context.window().inner_size().into();
        let screen_size = window_size.map(|n| n as f32);

        let index_buffer = context.import_with(
            "egui_index_buffer",
            &self.index_buffer,
            graphics::GraphResourceImportDesc::default(),
        );

        let vertex_buffer = context.import_with(
            "egui_vertex_buffer",
            &self.vertex_buffer,
            graphics::GraphResourceImportDesc::default(),
        );

        let index_offset = Self::PER_FRAME_INDEX_BYTE_SIZE * context.frame_index();
        let vertex_offset = Self::MAX_VERTEX_COUNT * context.frame_index();

        let pipeline = context.create_raster_pipeline(
            "egui_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/egui/egui.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv("shaders/egui/egui.frag.spv"))
                .rasterizer(graphics::RasterizerDesc {
                    cull_mode: vk::CullModeFlags::NONE,
                    ..Default::default()
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: context.swapchain.format(),
                    color_blend: Some(graphics::ColorBlendState {
                        src_color_blend_factor: vk::BlendFactor::ONE,
                        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        color_blend_op: vk::BlendOp::ADD,
                        src_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_DST_ALPHA,
                        dst_alpha_blend_factor: vk::BlendFactor::ONE,
                        alpha_blend_op: vk::BlendOp::ADD,
                    }),
                    ..Default::default()
                }]),
        );

        context
            .add_pass("egui_draw")
            .with_dependency(target_image, graphics::AccessKind::ColorAttachmentWrite)
            .record_custom(move |cmd, graph| {
                let target_image = graph.get_image(target_image);

                let index_buffer = graph.get_buffer(index_buffer);
                let vertex_buffer = graph.get_buffer(vertex_buffer);

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
                cmd.bind_index_buffer(index_buffer, index_offset as u64, vk::IndexType::UINT32);
                // cmd.bind_vertex_buffer(0, vertex_buffer, vertex_offset as u64);

                for batch in clipped_batches.iter() {
                    cmd.set_scissor(0, &[batch.clip_rect]);

                    let texture = graph.get_image(batch.texture);

                    cmd.build_constants()
                        .vec2(screen_size)
                        .sampled_image(&texture)
                        .buffer(&vertex_buffer)
                        .uint(vertex_offset as u32);

                    cmd.draw_indexed(batch.index_range.clone(), 0..1, batch.vertex_offset);
                }

                cmd.end_rendering();
            });
    }
}
