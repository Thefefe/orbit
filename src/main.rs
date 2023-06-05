#![allow(dead_code)]
use ash::vk;
use gpu_allocator::MemoryLocation;
use vulkan::DescriptorHandle;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode as KeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[allow(unused_imports)]
use glam::{Vec2, Vec3, Vec3A, Vec4, vec2, vec3, vec3a, vec4};

mod vulkan;
mod render;
mod utils;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
    color: Vec4
}

const VERTECES: &[Vertex] = &[
    Vertex { pos: vec2( 0.0,  0.5), uv: vec2(0.0, 0.0), color: vec4(1.0, 0.0, 0.0, 1.0) },
    Vertex { pos: vec2(-0.5, -0.5), uv: vec2(0.0, 0.0), color: vec4(0.0, 1.0, 0.0, 1.0) },
    Vertex { pos: vec2( 0.5, -0.5), uv: vec2(0.0, 0.0), color: vec4(0.0, 0.0, 1.0, 1.0) },
];

fn main() {
    if let Err(err) = utils::init_logger() {
        eprintln!("failed to initialize logger: {err}");
    };

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("orbit").build(&event_loop).expect("failed to build window");

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
            color_attachments: &[
                render::PipelineColorAttachment {
                    format: context.swapchain.format(),
                    ..Default::default()
                }
            ],
            depth_state: None,
            multisample: render::MultisampleCount::None,
        });

        context.destroy_shader_module(vertex_module);
        context.destroy_shader_module(fragment_module);
    
        pipeline
    };

    let vertex_buffer = context.create_buffer_init(&render::BufferDesc {
        name: "buffer",
        size: 120,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        memory_location: MemoryLocation::CpuToGpu,
    }, bytemuck::cast_slice(VERTECES));

    event_loop.run(move |event, _target, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
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
        },
        Event::MainEventsCleared => {
            let window_size = context.window().inner_size();
            if window_size.width == 0 || window_size.height == 0 { return }

            let frame = context.begin_frame();

            // temporary
            let device = &frame.context.device;
            let descriptors = &frame.context.descriptors;

            let acquired_image = frame.acquired_image;

            let frame_index = frame.context.frame_index;
            let frame = &mut frame.context.frames[frame_index];

            let command_buffer = frame.command_pool.begin_new(&device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // --------------

            command_buffer.wait_semaphore(
                frame.image_available_semaphore,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            );

            command_buffer.signal_semaphore(
                frame.render_finished_semaphore,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            );

            let recorder = command_buffer.record(&device);

            recorder.barrier(&[], &[
                vulkan::image_barrier(
                    &acquired_image,
                    vulkan::AccessKind::None,
                    vulkan::AccessKind::ColorAttachmentWrite,
                )
            ], &[]);

            {
                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(acquired_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(acquired_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));

                let render_pass = recorder.begin_rendering(&rendering_info);

                descriptors.bind_descriptors(&render_pass, vk::PipelineBindPoint::GRAPHICS);
                render_pass.bind_raster_pipeline(&pipeline);

                // device.raw.cmd_set_scissor(command_buffer.handle(), 0, &[acquired_image.full_rect()]);
                // device.raw.cmd_set_viewport(command_buffer.handle(), 0, &[acquired_image.full_viewport()]);

                descriptors.push_bindings(&render_pass, &[
                    vertex_buffer.descriptor_index.unwrap().to_raw()
                ]);
                render_pass.draw(0..3, 0..1);
            }

            recorder.barrier(&[], &[
                vulkan::image_barrier(
                    &acquired_image,
                    vulkan::AccessKind::ColorAttachmentWrite,
                    vulkan::AccessKind::Present,
                )
            ], &[]);
        }
        Event::LoopDestroyed => {
            unsafe {
                context.device.raw.device_wait_idle().unwrap();
            }
            context.destroy_raster_pipeline(&pipeline);
            context.destroy_buffer(&vertex_buffer);
        }
        _ => {}
    })
}
