#![allow(dead_code)]

use ash::vk;
use gpu_allocator::MemoryLocation;
use winit::{
    event::{Event,  VirtualKeyCode as KeyCode},
    event_loop::{EventLoop, ControlFlow},
    window::WindowBuilder,
};

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4};

mod render;
mod egui_renderer;
mod input;
mod utils;

use render::DescriptorHandle;
use input::Input;

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

use egui_renderer::EguiRenderer;

struct App {
    basic_pipeline: render::RasterPipeline,
    vertex_buffer: render::Buffer,

    profiler_open: bool,
}

impl App {
    fn new(context: &render::Context) -> Self {
        let basic_pipeline = {
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
                vertex_input: render::VertexInput::default(),
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

        Self {
            basic_pipeline,
            vertex_buffer,

            profiler_open: false,
        }
    }

    fn update(&mut self, input: &Input, egui_ctx: &egui::Context, control_flow: &mut ControlFlow) {
        puffin::profile_function!();

        if input.key_pressed(KeyCode::F3) {
            self.profiler_open = !self.profiler_open;
        }

        if self.profiler_open {
            self.profiler_open = puffin_egui::profiler_window(&egui_ctx);
        }

        if input.close_requested() | input.key_pressed(KeyCode::Escape) {
            control_flow.set_exit();
        }
    }

    fn render(&mut self, frame_ctx: &mut render::FrameContext) {
        puffin::profile_function!();
        let swapchain_image = frame_ctx.get_swapchain_image();

        let vertex_buffer = frame_ctx.import_buffer("vertex_buffer", &self.vertex_buffer, &Default::default());

        let pipeline = self.basic_pipeline;

        frame_ctx.add_pass(
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
    }

    fn destroy(&self, context: &render::Context) {
        context.destroy_raster_pipeline(&self.basic_pipeline);
        context.destroy_buffer(&self.vertex_buffer);
    }
}

fn main() {
    utils::init_logger();
    puffin::set_scopes_on(true);

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

    let egui_ctx = egui::Context::default();
    let mut egui_state = egui_winit::State::new(&event_loop);
    let mut egui_renderer = EguiRenderer::new(&context);

    let mut input = Input::new();

    let mut app = App::new(&context);

    event_loop.run(move |event, _target, control_flow| {
        if let Event::WindowEvent { event, .. } = &event {
            if egui_state.on_event(&egui_ctx, &event).consumed {
                return;
            }
        };

        if event == Event::LoopDestroyed {
            unsafe {
                context.device.raw.device_wait_idle().unwrap();
            }

            app.destroy(&context);

            egui_renderer.destroy(&context);
        }

        if input.handle_event(&event) {
            puffin::GlobalProfiler::lock().new_frame();
            puffin::profile_scope!("update");

            let raw_input = egui_state.take_egui_input(&context.window());
            egui_ctx.begin_frame(raw_input);

            app.update(&input, &egui_ctx, control_flow);

            let full_output = egui_ctx.end_frame();
            egui_state.handle_platform_output(&context.window(), &egui_ctx, full_output.platform_output);

            let window_size = context.window().inner_size();
            if window_size.width == 0 || window_size.height == 0 {
                return;
            }

            let mut frame_ctx = context.begin_frame();

            let swapchain_image = frame_ctx.get_swapchain_image();

            app.render(&mut frame_ctx);            

            let clipped_primitives = {
                puffin::profile_scope!("egui_tessellate");
                egui_ctx.tessellate(full_output.shapes)
            };
            egui_renderer.render(&mut frame_ctx, &clipped_primitives, &full_output.textures_delta, swapchain_image);

            input.clear_frame()
        }
    })
}
