#![allow(dead_code)]
use std::{collections::HashMap, ops::Range};

use ash::vk;
use gpu_allocator::MemoryLocation;
use render::DescriptorHandle;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode as KeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4};

mod render;
mod utils;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
    color: Vec4,
}

const VERTECES: &[Vertex] = &[
    Vertex {
        pos: vec2(0.0, 0.5),
        uv: vec2(0.0, 0.0),
        color: vec4(1.0, 0.0, 0.0, 1.0),
    },
    Vertex {
        pos: vec2(-0.5, -0.5),
        uv: vec2(0.0, 0.0),
        color: vec4(0.0, 1.0, 0.0, 1.0),
    },
    Vertex {
        pos: vec2(0.5, -0.5),
        uv: vec2(0.0, 0.0),
        color: vec4(0.0, 0.0, 1.0, 1.0),
    },
];

struct ClippedBatch {
    clip_rect: vk::Rect2D,
    texture: render::ResourceHandle,
    index_range: Range<u32>,
    vertex_offset: i32,
}

struct EguiRenderer {
    pipeline: render::RasterPipeline,
    vertex_buffer: render::Buffer,
    index_buffer: render::Buffer,
    textures: HashMap<egui::TextureId, render::Image>,
}

impl EguiRenderer {
    const MAX_VERTEX_COUNT: usize = 20000;
    const MAX_INDEX_COUNT: usize = 20000;
    const IMAGE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

    pub fn new(context: &render::Context) -> Self {
        let pipeline = {
            let (vertex_shader, fragment_shader) = {
                let vertex_spv = utils::load_spv("shaders/egui/egui.vert.spv").expect("failed to load shader");
                let fragment_spv = utils::load_spv("shaders/egui/egui.frag.spv").expect("failed to load shader");

                (
                    context.create_shader_module(&vertex_spv, "egui_vertex_shader"),
                    context.create_shader_module(&fragment_spv, "egui_fragment_shader"),
                )
            };

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline(&render::RasterPipelineDesc {
                name: "egui_pipeline",
                vertex_stage: render::ShaderStage {
                    module: vertex_shader,
                    entry,
                },
                fragment_stage: render::ShaderStage {
                    module: fragment_shader,
                    entry,
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
                        ..Default::default()
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
        let vertex_buffer = context.create_buffer(&render::BufferDesc {
            name: "egui_vertex_buffer",
            size: vertex_buffer_size as u64,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
        });

        let index_buffer_size = std::mem::size_of::<u32>() * Self::MAX_VERTEX_COUNT;
        let index_buffer = context.create_buffer(&render::BufferDesc {
            name: "egui_index_buffer",
            size: index_buffer_size as u64,
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

    pub fn update(
        &mut self,
        context: &mut render::FrameContext,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
    ) -> Vec<ClippedBatch> {
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
                assert!(vertex_cursor as usize + mesh.vertices.len() < Self::MAX_VERTEX_COUNT);
                assert!(index_cursor as usize + mesh.indices.len() < Self::MAX_INDEX_COUNT);

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

            let image = context.create_image(&render::ImageDesc {
                name: &format!("egui_image_{source}_{number}"),
                format: Self::IMAGE_FORMAT,
                width: width as u32,
                height: height as u32,
                mip_levels: 1,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                tiling: vk::ImageTiling::OPTIMAL,
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

        let clipped_batches = self.update(context, clipped_primitives, textures_delta);
        let pipeline = self.pipeline;

        let window_size: [u32; 2] = context.window().inner_size().into();
        let screen_size = window_size.map(|n| n as f32);

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

                for batch in clipped_batches.iter() {
                    cmd.set_scissor(0, &[batch.clip_rect]);

                    let texture = graph.get_image(batch.texture).unwrap();

                    cmd.push_bindings(&[
                        bytemuck::cast(screen_size[0]),
                        bytemuck::cast(screen_size[1]),
                        vertex_buffer.descriptor_index.unwrap().to_raw(),
                        texture.descriptor_index.unwrap().to_raw(),
                    ]);
                    cmd.draw_indexed(batch.index_range.clone(), 0..1, batch.vertex_offset);                    
                }

                cmd.end_rendering();
            },
        );
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_raster_pipeline(&self.pipeline);
        context.destroy_buffer(&self.index_buffer);
        context.destroy_buffer(&self.vertex_buffer);
        for image in self.textures.values() {
            context.destroy_image(image);
        }
    }
}

fn main() {
    utils::init_logger();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("orbit")
        .with_resizable(true)
        .build(&event_loop)
        .expect("failed to build window");

    let mut context = render::Context::new(
        window,
        &render::ContextDesc {
            present_mode: vk::PresentModeKHR::IMMEDIATE,
        },
    );

    let pipeline = {
        let vertex_shader = utils::load_spv("shaders/basic.vert.spv").unwrap();
        let fragment_shader = utils::load_spv("shaders/basic.frag.spv").unwrap();

        let vertex_module = context.create_shader_module(&vertex_shader, "basic_vertex_shader");
        let fragment_module = context.create_shader_module(&fragment_shader, "basic_fragment_shader");

        let entry = cstr::cstr!("main");

        let pipeline = context.create_raster_pipeline(&render::RasterPipelineDesc {
            name: "basic_pipeline",
            vertex_stage: render::ShaderStage {
                module: vertex_module,
                entry,
            },
            fragment_stage: render::ShaderStage {
                module: fragment_module,
                entry,
            },
            rasterizer: render::RasterizerDesc {
                primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                cull_mode: vk::CullModeFlags::NONE,
            },
            color_attachments: &[render::PipelineColorAttachment {
                format: context.swapchain.format(),
                ..Default::default()
            }],
            depth_state: None,
            multisample: render::MultisampleCount::None,
        });

        context.destroy_shader_module(vertex_module);
        context.destroy_shader_module(fragment_module);

        pipeline
    };

    let vertex_buffer = context.create_buffer_init(
        &render::BufferDesc {
            name: "buffer",
            size: 120,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        },
        bytemuck::cast_slice(VERTECES),
    );

    let egui_ctx = egui::Context::default();
    let mut egui_state = egui_winit::State::new(&event_loop);
    let mut egui_renderer = EguiRenderer::new(&context);

    // egui demo stuff
    let mut age = 0;
    let mut name = String::new();

    event_loop.run(move |event, _target, control_flow| match event {
        Event::WindowEvent { event, .. } => {
            let response = egui_state.on_event(&egui_ctx, &event);

            if response.consumed {
                return;
            }

            match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => match keycode {
                    KeyCode::Escape => control_flow.set_exit(),
                    _ => {}
                },
                _ => {}
            }
        }
        Event::MainEventsCleared => {
            let raw_input = egui_state.take_egui_input(&context.window());
            egui_ctx.begin_frame(raw_input);

            egui::Window::new("egui demo").show(&egui_ctx, |ui| {
                ui.heading("My egui Application");
                ui.horizontal(|ui| {
                    ui.label("Your name: ");
                    ui.text_edit_singleline(&mut name);
                });
                ui.add(egui::Slider::new(&mut age, 0..=120).text("age"));
                if ui.button("Click each year").clicked() {
                    age += 1;
                }
                ui.label(format!("Hello '{name}', age {age}"));
            });

            let full_output = egui_ctx.end_frame();
            egui_state.handle_platform_output(&context.window(), &egui_ctx, full_output.platform_output);

            let window_size = context.window().inner_size();
            if window_size.width == 0 || window_size.height == 0 {
                return;
            }

            let mut frame = context.begin_frame();

            let swapchain_image = frame.get_swapchain_image();

            let vertex_buffer = frame.import_buffer("vertex_buffer", &vertex_buffer, &Default::default());

            frame.add_pass(
                "triangle",
                &[
                    (swapchain_image, render::AccessKind::ColorAttachmentWrite),
                    (vertex_buffer, render::AccessKind::VertexShaderRead),
                ],
                move |cmd, graph| {
                    let swapchain_image = graph.get_image(swapchain_image).unwrap();
                    let vertex_buffer = graph.get_buffer(vertex_buffer).unwrap();

                    let color_attachment = vk::RenderingAttachmentInfo::builder()
                        .image_view(swapchain_image.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        })
                        .store_op(vk::AttachmentStoreOp::STORE);

                    let rendering_info = vk::RenderingInfo::builder()
                        .render_area(swapchain_image.full_rect())
                        .layer_count(1)
                        .color_attachments(std::slice::from_ref(&color_attachment));

                    cmd.begin_rendering(&rendering_info);

                    cmd.bind_raster_pipeline(pipeline);

                    cmd.push_bindings(&[vertex_buffer.descriptor_index.unwrap().to_raw()]);
                    cmd.draw(0..3, 0..1);

                    cmd.end_rendering();
                },
            );

            let clipped_primitives = egui_ctx.tessellate(full_output.shapes);
            egui_renderer.render(&mut frame, &clipped_primitives, &full_output.textures_delta, swapchain_image);
        }
        Event::LoopDestroyed => {
            unsafe {
                context.device.raw.device_wait_idle().unwrap();
            }

            egui_renderer.destroy(&context);
            
            context.destroy_raster_pipeline(&pipeline);
            context.destroy_buffer(&vertex_buffer);
        }
        _ => {}
    })
}
