#![allow(dead_code)]

use std::f32::consts::PI;

use ash::vk;
use glam::Mat3;
use gpu_allocator::MemoryLocation;
use mesh::{MeshBuffer, MeshIndex};
use time::Time;
use winit::{
    event::{Event,  VirtualKeyCode as KeyCode, MouseButton},
    event_loop::{EventLoop, ControlFlow},
    window::WindowBuilder,
};

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4, Quat, Mat4};

mod input;
mod time;
mod render;
mod utils;

mod egui_renderer;
mod mesh;

use render::DescriptorHandle;
use input::Input;
use egui_renderer::EguiRenderer;

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub orientation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn translate(&mut self, translation: Vec3) {
        self.position += self.orientation * translation;
    }

    pub fn compute_affine(&self) -> glam::Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.orientation, self.position)
    }

    pub fn compute_linear(&self) -> Mat3 {
        Mat3::from_mat4(self.compute_affine())
    }
}

struct CameraController {
    mouse_sensitivity: f32,
    pitch: f32,
    yaw: f32,
    movement_speed: f32,
}

impl CameraController {
    #[rustfmt::skip]
    const CONTROL_KEYS: &[(KeyCode, glam::Vec3)] = &[
        (KeyCode::W, glam::vec3(  0.0,  0.0,  1.0)),
        (KeyCode::S, glam::vec3(  0.0,  0.0, -1.0)),
        (KeyCode::D, glam::vec3(  1.0,  0.0,  0.0)),
        (KeyCode::A, glam::vec3( -1.0,  0.0,  0.0)),
        (KeyCode::E, glam::vec3(  0.0,  1.0,  0.0)),
        (KeyCode::Q, glam::vec3(  0.0, -1.0,  0.0)),
    ];

    pub fn new(movement_speed: f32, mouse_sensitivity: f32) -> Self {
        Self {
            mouse_sensitivity,
            yaw: 0.0,
            pitch: 0.0,
            movement_speed,
        }
    }

    pub fn update(&mut self, input: &Input, delta_time: f32, transform: &mut Transform) {
        if input.mouse_held(MouseButton::Right) {
            let mouse_delta = input.mouse_delta();
            self.pitch += mouse_delta.x * self.mouse_sensitivity;
            self.yaw = f32::clamp(self.yaw + -mouse_delta.y * self.mouse_sensitivity, -PI / 2.0, PI / 2.0);
        }

        let mut move_dir = glam::Vec3::ZERO;

        for (key_code, dir) in Self::CONTROL_KEYS {
            if input.key_held(*key_code) {
                move_dir += *dir;
            }
        }

        let movement_speed = if input.key_held(KeyCode::LShift) {
            self.movement_speed * 8.0
        } else if input.key_held(KeyCode::LControl) {
            self.movement_speed / 8.0
        } else {
            self.movement_speed
        };

        transform.orientation = glam::Quat::from_euler(glam::EulerRot::YXZ, self.pitch, self.yaw, 0.0);
        transform.translate(move_dir.normalize_or_zero() * movement_speed * delta_time);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Projection {
    Orthographic {
        half_width: f32,
        near_clip: f32,
        far_clip: f32,
    },
    Perspective {
        fov: f32,
        near_clip: f32,
    }
}

impl Projection {
    #[inline]
    #[rustfmt::skip]
    pub fn compute_matrix(self, aspect_ratio: f32) -> glam::Mat4 {
    match self {
        Projection::Perspective {
                fov,
                near_clip,
            } => glam::Mat4::perspective_infinite_reverse_lh(fov, aspect_ratio, near_clip),
            Projection::Orthographic {
                half_width,
                near_clip,
                far_clip,
            } => {
                let half_height = half_width * aspect_ratio.recip();

                glam::Mat4::orthographic_lh(
                    -half_width, half_width,
                    -half_height, half_height,
                    near_clip, far_clip,
                )
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub transform: Transform,
    pub projection: Projection,
}

impl Camera {
    #[inline]
    pub fn compute_matrix(&self, aspect_ratio: f32) -> glam::Mat4 {
        let proj = self.projection.compute_matrix(aspect_ratio);
        let view = self.transform.compute_affine().inverse();

        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GlobalData {
    viewproj: Mat4,
}

struct App {
    basic_pipeline: render::RasterPipeline,
    mesh_buffer: MeshBuffer,
    mesh_index: MeshIndex,
    global_buffer: render::Buffer,
    
    camera: Camera,
    camera_controller: CameraController,

    graph_debugger_open: bool,
    profiler_open: bool,
}

impl App {
    const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

    fn new(context: &render::Context) -> Self {
        let mesh_pipeline = {
            let vertex_shader = utils::load_spv("shaders/mesh.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/mesh.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "mesh_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "mesh_fragment_shader");

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
                depth_state: Some(render::DepthState {
                    format: Self::DEPTH_FORMAT,
                    test: true,
                    write: true,
                    compare: vk::CompareOp::GREATER,
                }),
                multisample: render::MultisampleCount::None,
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        let mesh_data = mesh::MeshData::load_obj("D:/dev/models/obj/teapot.obj").unwrap();
        let mut mesh_buffer = MeshBuffer::new(context, mesh_data.vertices.len(), mesh_data.indices.len());
        let mesh_index = mesh_buffer.add_mesh(context, &mesh_data);

        let global_buffer = context.create_buffer("global_buffer", &render::BufferDesc {
            size: std::mem::size_of::<GlobalData>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
        });

        Self {
            basic_pipeline: mesh_pipeline,
            mesh_buffer,
            mesh_index,
            
            global_buffer,
            
            camera: Camera {
                transform: Transform {
                    position: vec3(0.0, 2.0, -5.0),
                    ..Default::default()
                },
                projection: Projection::Perspective { fov: 90f32.to_radians(), near_clip: 0.1 },
            },
            camera_controller: CameraController::new(1.0, 0.003),

            profiler_open: false,
            graph_debugger_open: false,
        }
    }

    fn update(
        &mut self,
        input: &Input,
        time: &Time,
        egui_ctx: &egui::Context,
        context: &render::Context,
        control_flow: &mut ControlFlow
    ) {
        puffin::profile_function!();
        
        self.camera_controller.update(input, time.delta().as_secs_f32(), &mut self.camera.transform);

        if input.key_pressed(KeyCode::F2) {
            self.graph_debugger_open = !self.graph_debugger_open;
        }
        if input.key_pressed(KeyCode::F3) {
            self.profiler_open = !self.profiler_open;
        }

        if self.graph_debugger_open {
            // temporary
            egui::Window::new("rendergraph debugger").open(&mut self.graph_debugger_open).show(egui_ctx, |ui| {
                for (i, batch) in context.compiled_graph.iter_batches().enumerate() {
                    ui.collapsing(format!("batch_{i}"), |ui| {
                        ui.collapsing("memory_barrier", |ui| {
                            ui.label(format!("src_stage: {:?}", batch.memory_barrier.src_stage_mask));
                            ui.label(format!("src_access: {:?}", batch.memory_barrier.src_access_mask));
                            ui.label(format!("dst_stage: {:?}", batch.memory_barrier.dst_stage_mask));
                            ui.label(format!("dst_access: {:?}", batch.memory_barrier.dst_access_mask));
                        });

                        ui.collapsing("begin_image_barriers", |ui| {
                            for (i, image_barrier) in batch.begin_image_barriers.iter().enumerate() {
                                ui.collapsing(&format!("image {i}"), |ui| {
                                    ui.label(format!("src_stage: {:?}", image_barrier.src_stage_mask));
                                    ui.label(format!("src_access: {:?}", image_barrier.src_access_mask));
                                    ui.label(format!("dst_stage: {:?}", image_barrier.dst_stage_mask));
                                    ui.label(format!("dst_access: {:?}", image_barrier.dst_access_mask));
                                    ui.label(format!("src_layout: {:?}", image_barrier.old_layout));
                                    ui.label(format!("dst_layout: {:?}", image_barrier.new_layout));
                                });
                            }
                        });

                        ui.collapsing("finish_image_barriers", |ui| {
                            for (i, image_barrier) in batch.finish_image_barriers.iter().enumerate() {
                                ui.collapsing(&format!("image #{i}"), |ui| {
                                    ui.label(format!("src_stage: {:?}", image_barrier.src_stage_mask));
                                    ui.label(format!("src_access: {:?}", image_barrier.src_access_mask));
                                    ui.label(format!("dst_stage: {:?}", image_barrier.dst_stage_mask));
                                    ui.label(format!("dst_access: {:?}", image_barrier.dst_access_mask));
                                    ui.label(format!("src_layout: {:?}", image_barrier.old_layout));
                                    ui.label(format!("dst_layout: {:?}", image_barrier.new_layout));
                                });
                            }
                        });

                        ui.collapsing("passes", |ui| {
                            for pass in batch.passes {
                                ui.label(&pass.name);
                            }
                        });
                    });
                }
            });
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
        let screen_extent = frame_ctx.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;
        let globals = GlobalData {
            viewproj: self.camera.compute_matrix(aspect_ratio),
        };
        frame_ctx.immediate_write_buffer(&self.global_buffer, bytemuck::bytes_of(&globals), 0);

        let swapchain_image = frame_ctx.get_swapchain_image();
        let depth_image = frame_ctx.create_transient_image("depth_image", render::ImageDesc {
            format: Self::DEPTH_FORMAT,
            width: screen_extent.width,
            height: screen_extent.height,
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            aspect: vk::ImageAspectFlags::DEPTH,
        });

        let vertex_buffer = frame_ctx.import_buffer("mesh_vertex_buffer", &self.mesh_buffer.vertex_buffer, &Default::default());
        let index_buffer = frame_ctx.import_buffer("mesh_index_buffer", &self.mesh_buffer.index_buffer, &Default::default());
        let globals_buffer = frame_ctx.import_buffer("globals_buffer", &self.global_buffer, &Default::default());

        let pipeline = self.basic_pipeline;

        let mesh_index = self.mesh_index;

        frame_ctx.add_pass(
            "mesh_pass",
            &[
                (swapchain_image, render::AccessKind::ColorAttachmentWrite),
                (depth_image, render::AccessKind::DepthAttachmentWrite),
            ],
            move |cmd, graph| {
                let swapchain_image = graph.get_image(swapchain_image).unwrap();
                let depth_image = graph.get_image(depth_image).unwrap();

                let vertex_buffer = graph.get_buffer(vertex_buffer).unwrap();
                let index_buffer = graph.get_buffer(index_buffer).unwrap();
                let globals_buffer = graph.get_buffer(globals_buffer).unwrap();

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

                let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
                    .image_view(depth_image.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(swapchain_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .depth_attachment(&depth_attachemnt);

                cmd.begin_rendering(&rendering_info);
                
                cmd.bind_raster_pipeline(pipeline);
                cmd.bind_index_buffer(&index_buffer);

                cmd.push_bindings(&[
                    vertex_buffer.descriptor_index.unwrap().to_raw(),
                    globals_buffer.descriptor_index.unwrap().to_raw(),
                ]);
                cmd.draw_indexed(mesh_index.index_range(), 0..1, mesh_index.vertex_offset as i32);

                cmd.end_rendering();
            },
        );
    }

    fn destroy(&self, context: &render::Context) {
        context.destroy_raster_pipeline(&self.basic_pipeline);
        self.mesh_buffer.destroy(context);
        context.destroy_buffer(&self.global_buffer);
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
    let mut time = Time::new();

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
            
            time.update_now();
            app.update(&input, &time, &egui_ctx, &context, control_flow);

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
