#![allow(dead_code)]

use std::f32::consts::PI;

use ash::vk;
use gltf_loader::load_gltf;
use gpu_allocator::MemoryLocation;
use assets::{GpuAssetStore};
use scene::{Transform, SceneBuffer, GpuDrawCommand};
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
mod assets;
mod scene;
mod collections;
mod utils;

mod gltf_loader;

mod egui_renderer;

use render::DescriptorHandle;
use input::Input;
use egui_renderer::EguiRenderer;

struct CameraController {
    mouse_sensitivity: f32,
    pitch: f32,
    yaw: f32,
    movement_speed: f32,
}

impl CameraController {
    #[rustfmt::skip]
    const CONTROL_KEYS: &[(KeyCode, glam::Vec3)] = &[
        (KeyCode::W, glam::vec3(  0.0,  0.0, -1.0)),
        (KeyCode::S, glam::vec3(  0.0,  0.0,  1.0)),
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
            self.pitch -= mouse_delta.x * self.mouse_sensitivity;
            self.yaw = f32::clamp(self.yaw + mouse_delta.y * self.mouse_sensitivity, -PI / 2.0, PI / 2.0);
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
        transform.translate_relative(move_dir.normalize_or_zero() * movement_speed * delta_time);
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
            } => glam::Mat4::perspective_infinite_reverse_rh(fov, aspect_ratio, near_clip),
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
        let view = self.transform.compute_matrix().inverse();

        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PerFrameData {
    viewproj: Mat4,
    view_pos: Vec3,
    render_mode: u32,
    light_direction: Vec3,
    _padding: u32,
}

struct App {
    gpu_assets: GpuAssetStore,
    scene: SceneBuffer,
    scene_draw_gen: SceneDrawGen,

    basic_pipeline: render::RasterPipeline,
    per_frame_buffer: render::Buffer,

    camera: Camera,
    camera_controller: CameraController,
    light_direction: Vec3,

    scene_editor_open: bool,
    graph_debugger_open: bool,
    profiler_open: bool,
}

impl App {
    const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
    const MULTISAMPLING: render::MultisampleCount = render::MultisampleCount::X4;

    fn new(context: &render::Context, gltf_path: Option<&str>) -> Self {
        let mesh_pipeline = {
            let vertex_shader = utils::load_spv("shaders/mesh.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/mesh.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "mesh_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "mesh_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("basic_pipeline", &render::RasterPipelineDesc {
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
                    cull_mode: vk::CullModeFlags::BACK,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: context.swapchain.format(),
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: Some(render::DepthState {
                    format: Self::DEPTH_FORMAT,
                    test: true,
                    write: true,
                    compare: vk::CompareOp::GREATER,
                }),
                multisample: Self::MULTISAMPLING,
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        let mut gpu_assets = GpuAssetStore::new(context);
        let mut scene = SceneBuffer::new(context);
        
        if let Some(gltf_path) = gltf_path {
            load_gltf(gltf_path, context, &mut gpu_assets, &mut scene).unwrap();
        }

        scene.update_instances(context);
        scene.update_submeshes(context, &gpu_assets);

        let per_frame_buffer = context.create_buffer("global_buffer", &render::BufferDesc {
            size: std::mem::size_of::<PerFrameData>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
        });

        unsafe {
            per_frame_buffer.mapped_ptr.unwrap().cast::<PerFrameData>().as_mut().render_mode = 0;
        }
        Self {
            gpu_assets,
            scene,
            scene_draw_gen: SceneDrawGen::new(context),

            basic_pipeline: mesh_pipeline,
            per_frame_buffer,
            
            camera: Camera {
                transform: Transform {
                    position: vec3(0.0, 2.0, 0.0),
                    orientation: Quat::from_rotation_y(180.0f32.to_radians()),
                    ..Default::default()
                },
                projection: Projection::Perspective { fov: 90f32.to_radians(), near_clip: 0.001 },
            },
            camera_controller: CameraController::new(1.0, 0.003),
            light_direction: vec3(1.0, 1.0, 1.0),

            scene_editor_open: false,
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

        let delta_time = time.delta().as_secs_f32();
        
        self.camera_controller.update(input, delta_time, &mut self.camera.transform);

        if input.key_pressed(KeyCode::F1) {
            self.scene_editor_open = !self.scene_editor_open;
        }
        if input.key_pressed(KeyCode::F2) {
            self.graph_debugger_open = !self.graph_debugger_open;
        }
        if input.key_pressed(KeyCode::F3) {
            self.profiler_open = !self.profiler_open;
        }

        fn drag_vec3(ui: &mut egui::Ui, label: &str, vec: &mut Vec3, speed: f32) {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(egui::DragValue::new(&mut vec.x).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.y).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.z).speed(speed));
            });
        }

        if self.scene_editor_open {
            egui::Window::new("scene").open(&mut self.scene_editor_open).show(egui_ctx, |ui| {
                drag_vec3(ui, "light_direction", &mut self.light_direction, 0.001);
                for (entity_index, entity) in self.scene.entities.iter_mut().enumerate() {
                    let header = entity.name.as_ref().map_or(
                        format!("entity_{entity_index}"),
                        |name| format!("entity_{entity_index} ({name})")
                    );
                    ui.collapsing(header, |ui| {
                        drag_vec3(ui, "position", &mut entity.transform.position, 0.001);
    
                        let (euler_x, euler_y, euler_z) = entity.transform.orientation.to_euler(glam::EulerRot::YXZ);
                        let mut euler = vec3(euler_x, euler_y, euler_z);
                        drag_vec3(ui, "orientation", &mut euler, 0.05);
                        entity.transform.orientation = Quat::from_euler(glam::EulerRot::YXZ, euler.x, euler.y, euler.z);
    
                        drag_vec3(ui, "scale", &mut entity.transform.scale, 0.001);
                    });
                }
            });
        }

        const RENDER_MODES: &[KeyCode] = &[
            KeyCode::Key0,
            KeyCode::Key1,
            KeyCode::Key2,
            KeyCode::Key3,
            KeyCode::Key4,
            KeyCode::Key5,
            KeyCode::Key6,
            KeyCode::Key7,
            KeyCode::Key8,
            KeyCode::Key9,
        ];
        
        let mut new_render_mode = None;
        for (render_mode, key) in RENDER_MODES.iter().enumerate() {
            if input.key_pressed(*key) {
                new_render_mode = Some(render_mode as u32);
            }
        }

        if let Some(new_render_mode) = new_render_mode {   
            unsafe {
                self.per_frame_buffer.mapped_ptr.unwrap().cast::<PerFrameData>().as_mut().render_mode = new_render_mode;
            }
        }

        if self.graph_debugger_open {
            // temporary
            egui::Window::new("rendergraph debugger").open(&mut self.graph_debugger_open).show(egui_ctx, |ui| {
                for (i, batch) in context.compiled_graph.iter_batches().enumerate() {
                    ui.collapsing(format!("batch {i}"), |ui| {
                        ui.collapsing("memory barrier", |ui| {
                            ui.label(format!("src_stage: {:?}", batch.memory_barrier.src_stage_mask));
                            ui.label(format!("src_access: {:?}", batch.memory_barrier.src_access_mask));
                            ui.label(format!("dst_stage: {:?}", batch.memory_barrier.dst_stage_mask));
                            ui.label(format!("dst_access: {:?}", batch.memory_barrier.dst_access_mask));
                        });

                        ui.collapsing("begin image barriers", |ui| {
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

                        ui.collapsing("finish image barriers", |ui| {
                            for (i, image_barrier) in batch.finish_image_barriers.iter().enumerate() {
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
        self.scene.update_instances(frame_ctx);

        let screen_extent = frame_ctx.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        unsafe {
            let per_frame_data = self.per_frame_buffer.mapped_ptr.unwrap().cast::<PerFrameData>().as_mut();
            per_frame_data.viewproj = self.camera.compute_matrix(aspect_ratio);
            per_frame_data.view_pos = self.camera.transform.position;
            per_frame_data.light_direction = self.light_direction;
        }
        
        let swapchain_image = frame_ctx.get_swapchain_image();
        let (target_image, resolve_image) = if Self::MULTISAMPLING == render::MultisampleCount::None {
            (swapchain_image, None)
        } else {
            let msaa_image = frame_ctx.create_transient_image("msaa_image", render::ImageDesc {
                format: frame_ctx.swapchain_format(),
                width: screen_extent.width,
                height: screen_extent.height,
                mip_levels: 1,
                samples: Self::MULTISAMPLING,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                aspect: vk::ImageAspectFlags::COLOR,
            });
            (msaa_image, Some(swapchain_image))
        };
        let depth_image = frame_ctx.create_transient_image("depth_image", render::ImageDesc {
            format: Self::DEPTH_FORMAT,
            width: screen_extent.width,
            height: screen_extent.height,
            mip_levels: 1,
            samples: Self::MULTISAMPLING,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            aspect: vk::ImageAspectFlags::DEPTH,
        });
        
        let globals_buffer = frame_ctx.import_buffer("globals_buffer", &self.per_frame_buffer, &Default::default());
        
        let vertex_buffer = frame_ctx.import_buffer("mesh_vertex_buffer", &self.gpu_assets.vertex_buffer, &Default::default());
        let index_buffer = frame_ctx.import_buffer("mesh_index_buffer", &self.gpu_assets.index_buffer, &Default::default());
        let material_buffer = frame_ctx.import_buffer("material_buffer", &self.gpu_assets.material_buffer, &Default::default());
        let instance_buffer = frame_ctx.import_buffer("entity_instance_buffer", &self.scene.entity_data_buffer, &Default::default());

        let submesh_buffer = frame_ctx.import_buffer("scene_submeshe_buffer", &self.scene.submesh_buffer, &Default::default());
        let mesh_info_buffer = frame_ctx.import_buffer("mesh_info_buffer", &self.gpu_assets.mesh_info_buffer, &Default::default());
        let submesh_count = self.scene.submesh_data.len() as u32;

        let draw_commands_buffer = self.scene_draw_gen
            .create_draw_commands(frame_ctx, submesh_count, submesh_buffer, mesh_info_buffer, "main_draw_commands");

        let pipeline = self.basic_pipeline;

        let mut dependencies = vec![
            (target_image, render::AccessKind::ColorAttachmentWrite),
            (depth_image, render::AccessKind::DepthAttachmentWrite),
            (draw_commands_buffer, render::AccessKind::IndirectBuffer),
        ];

        if let Some(resolve_image) = resolve_image {
            dependencies.push((resolve_image, render::AccessKind::ColorAttachmentWrite));
        }

        frame_ctx.add_pass(
            "mesh_pass",
            &dependencies,
            move |cmd, graph| {
                let target_image = graph.get_image(target_image).unwrap();
                let resolve_image = resolve_image.map(|handle| graph.get_image(handle).unwrap());
                let depth_image = graph.get_image(depth_image).unwrap();

                let globals_buffer = graph.get_buffer(globals_buffer).unwrap();
                
                let vertex_buffer = graph.get_buffer(vertex_buffer).unwrap();
                let index_buffer = graph.get_buffer(index_buffer).unwrap();
                let material_buffer = graph.get_buffer(material_buffer).unwrap();

                let instance_buffer = graph.get_buffer(instance_buffer).unwrap();
                let draw_commands_buffer = graph.get_buffer(draw_commands_buffer).unwrap();

                let mut color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(target_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                if let Some(resolve_image) = resolve_image {
                    color_attachment = color_attachment
                        .resolve_image_view(resolve_image.view)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE);
                }

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
                    .render_area(target_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .depth_attachment(&depth_attachemnt);

                cmd.begin_rendering(&rendering_info);
                
                cmd.bind_raster_pipeline(pipeline);
                cmd.bind_index_buffer(&index_buffer);

                cmd.push_bindings(&[
                    globals_buffer.descriptor_index.unwrap().to_raw(),
                    vertex_buffer.descriptor_index.unwrap().to_raw(),
                    instance_buffer.descriptor_index.unwrap().to_raw(),
                    draw_commands_buffer.descriptor_index.unwrap().to_raw(),
                    material_buffer.descriptor_index.unwrap().to_raw(),
                ]);
                
                cmd.draw_indexed_indirect_count(
                    draw_commands_buffer,
                    4,
                    draw_commands_buffer,
                    0,
                    MAX_DRAW_COUNT as u32,
                    std::mem::size_of::<GpuDrawCommand>() as u32,
                );

                cmd.end_rendering();
            },
        );
    }

    fn destroy(&self, context: &render::Context) {
        self.gpu_assets.destroy(context);
        self.scene.destroy(context);
        self.scene_draw_gen.destroy(context);
        context.destroy_pipeline(&self.basic_pipeline);
        context.destroy_buffer(&self.per_frame_buffer);
    }
}
const MAX_DRAW_COUNT: usize = 1_000_000;

pub struct SceneDrawGen {
    pipeline: render::ComputePipeline,
}

impl SceneDrawGen {
    pub fn new(context: &render::Context) -> Self {
        let spv = utils::load_spv("shaders/scene_draw_gen.comp.spv").unwrap();
        let module = context.create_shader_module(&spv, "scene_draw_gen_module"); 

        let pipeline = context.create_compute_pipeline("scene_draw_gen_pipeline", &render::ShaderStage {
            module,
            entry: cstr::cstr!("main"),
        });

        context.destroy_shader_module(module);

        Self { pipeline }

    }

    pub fn create_draw_commands(
        &self,
        frame_ctx: &mut render::FrameContext,
        scene_submesh_count: u32,
        scene_submeshes: render::ResourceHandle,
        mesh_infos: render::ResourceHandle,
        draw_commands_name: &str,
    ) -> render::ResourceHandle {
        let draw_commands = frame_ctx.create_transient_buffer(draw_commands_name, render::BufferDesc {
            size: MAX_DRAW_COUNT * std::mem::size_of::<GpuDrawCommand>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        });
        let pipeline = self.pipeline;

        use render::AccessKind;

        frame_ctx.add_pass(
            format!("{draw_commands_name}_generation"),
            &[
                (scene_submeshes, AccessKind::ComputeShaderRead),
                (mesh_infos, AccessKind::ComputeShaderRead),
                (draw_commands, AccessKind::ComputeShaderWrite),
            ],
            move |cmd, graph| {
                let scene_submeshes = graph.get_buffer(scene_submeshes).unwrap();
                let mesh_infos = graph.get_buffer(mesh_infos).unwrap();
                let draw_commands = graph.get_buffer(draw_commands).unwrap();

                cmd.bind_compute_pipeline(pipeline);

                cmd.push_bindings(&[
                    scene_submeshes.descriptor_index.unwrap().to_raw(),
                    mesh_infos.descriptor_index.unwrap().to_raw(),
                    draw_commands.descriptor_index.unwrap().to_raw(),
                ]);
                cmd.dispatch([scene_submesh_count / 256 + 1, 1, 1]);
            }
        );
        
        draw_commands
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
    }
}

fn main() {
    puffin::GlobalProfiler::lock().new_frame();
    utils::init_logger(false);
    puffin::set_scopes_on(true);

    let gltf_path = std::env::args().nth(1);

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

    let mut app = App::new(&context, gltf_path.as_ref().map(|s| s.as_str()));

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
