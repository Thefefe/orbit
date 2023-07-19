#![allow(dead_code)]

use std::{f32::consts::PI, borrow::Cow};

use ash::vk;
use gltf_loader::load_gltf;
use gpu_allocator::MemoryLocation;
use assets::GpuAssetStore;
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

    pub fn update_look(&mut self, input: &Input, transform: &mut Transform) {
        let mouse_delta = input.mouse_delta();
        self.pitch -= mouse_delta.x * self.mouse_sensitivity;
        self.yaw = f32::clamp(self.yaw + mouse_delta.y * self.mouse_sensitivity, -PI / 2.0, PI / 2.0);

        transform.orientation = glam::Quat::from_euler(glam::EulerRot::YXZ, self.pitch, self.yaw, 0.0);
    }

    pub fn update_movement(&mut self, input: &Input, delta_time: f32, transform: &mut Transform) {
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
            } => {
                glam::Mat4::perspective_infinite_reverse_rh(fov, aspect_ratio, near_clip)
            },
            Projection::Orthographic {
                half_width,
                near_clip,
                far_clip,
            } => {
                let half_height = half_width * aspect_ratio.recip();

                glam::Mat4::orthographic_rh(
                    -half_width, half_width,
                    -half_height, half_height,
                    far_clip,
                    near_clip,
                )
            }
        }
    }

    pub fn near(&self) -> f32 {
        match self {
            Projection::Orthographic { near_clip, .. } | Projection::Perspective { near_clip, .. } => *near_clip,
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

const NDC_BOUNDS: [Vec4; 8] = [
    vec4(-1.0, -1.0, 0.0,  1.0),
    vec4( 1.0, -1.0, 0.0,  1.0),
    vec4( 1.0,  1.0, 0.0,  1.0),
    vec4(-1.0,  1.0, 0.0,  1.0),
    
    vec4(-1.0, -1.0, 1.0,  1.0),
    vec4( 1.0, -1.0, 1.0,  1.0),
    vec4( 1.0,  1.0, 1.0,  1.0),
    vec4(-1.0,  1.0, 1.0,  1.0),
];

const MAX_SHADOW_CASCADE_COUNT: usize = 4;
const SHADOW_RESOLUTION: u32 = 1024 * 1;

// fn uniform_frustum_split(index: usize, near: f32, far: f32, cascade_count: usize) -> f32{
//     near + (far - near) * (index as f32 / cascade_count as f32)
// }

// fn logarithmic_frustum_split(index: usize, near: f32, far: f32, cascade_count: usize) -> f32 {
//     near * (far / near).powf(index as f32 / cascade_count as f32)
// }

// fn practical_frustum_split(index: usize, near: f32, far: f32, cascade_count: usize, lambda: f32) -> f32 {
//     logarithmic_frustum_split(index, near, far, cascade_count) * lambda +
//     uniform_frustum_split(index, near, far, cascade_count) * (1.0 - lambda)
// }

fn frustum_split(near: f32, far: f32, lambda: f32, ratio: f32) -> f32 {
    let uniform = near + (far - near) * ratio;
    let log = near * (far / near).powf(ratio);
    
    log * lambda + (1.0 - lambda) * uniform
}

fn frustum_planes_from_matrix(matrix: &Mat4) -> [Vec4; 6] {
    let mut planes = [matrix.row(3); 6];
    planes[0] += matrix.row(0);
    planes[1] -= matrix.row(0);
    planes[2] += matrix.row(1);
    planes[3] -= matrix.row(1);
    planes[4] += matrix.row(2);
    planes[5] -= matrix.row(2);
    planes
}

fn frustum_corners_from_matrix(matrix: &Mat4) -> [Vec4; 8] {
    let inv_matrix = matrix.inverse();
    NDC_BOUNDS.map(|v| {
        let v = inv_matrix * v;
        v / v.w
    })
}

fn perspective_corners(fovy: f32, aspect_ratio: f32, near: f32, far: f32) -> [Vec4; 8] {
    let tan_half_h = f32::tan(fovy / 2.0) * aspect_ratio;
    let tan_half_v = f32::tan(fovy / 2.0);

    let xn = near * tan_half_h;
    let yn = near * tan_half_v;
    let xf = far * tan_half_h;
    let yf = far * tan_half_v;

    [
        vec4(-xn, -yn, -near,  1.0),
        vec4( xn, -yn, -near,  1.0),
        vec4( xn,  yn, -near,  1.0),
        vec4(-xn,  yn, -near,  1.0),
        vec4(-xf, -yf, -far,   1.0),
        vec4( xf, -yf, -far,   1.0),
        vec4( xf,  yf, -far,   1.0),
        vec4(-xf,  yf, -far,   1.0),
    ]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuDirectionalLight {
    projection_matrices: [Mat4; MAX_SHADOW_CASCADE_COUNT],
    shadow_maps: [u32; MAX_SHADOW_CASCADE_COUNT],
    color: Vec3,
    intensity: f32,
    direction: Vec3,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PerFrameData {
    view_projection: Mat4,
    view: Mat4,
    view_pos: Vec3,
    render_mode: u32,
}

struct App {
    gpu_assets: GpuAssetStore,
    scene: SceneBuffer,
    debug_line_renderer: DebugLineRenderer,
    scene_draw_gen: SceneDrawGen,
    shadow_map_renderer: ShadowMapRenderer,
    scree_blitter: ScreenPostProcess,
    equirectangular_cube_map_loader: EnvironmentMapLoader,
    
    forward_pipeline: render::RasterPipeline,
    skybox_pipeline: render::RasterPipeline,
    brdf_integration_pipeline: render::RasterPipeline,

    brdf_integration_map: render::Image,
    environment_map: EnvironmentMap,

    camera: Camera,
    camera_controller: CameraController,
    mock_camera: Camera,
    mock_camera_controller: CameraController,
    sun_light: Camera,
    light_dir_controller: CameraController,
    
    render_mode: u32,
    camera_exposure: f32,
    light_color: Vec3,
    light_intensitiy: f32,
    max_shadow_distance: f32,
    frustum_split_lambda: f32,
    selected_cascade: usize,
    view_mock_camera: bool,

    use_mock_camera: bool,
    show_cascade_view_frustum: bool,
    show_cascade_light_frustum: bool,
    open_scene_editor_open: bool,
    open_graph_debugger: bool,
    open_profiler: bool,
    open_camera_light_editor: bool,
}

impl App {
    pub const COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
    pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
    pub const MULTISAMPLING: render::MultisampleCount = render::MultisampleCount::None;

    fn new(context: &render::Context, gltf_path: Option<&str>) -> Self {
        let forward_pipeline = {
            let vertex_shader = utils::load_spv("shaders/mesh.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/mesh.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "mesh_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "mesh_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("forward_pipeline", &render::RasterPipelineDesc {
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
                    cull_mode: vk::CullModeFlags::BACK,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: Self::COLOR_FORMAT,
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
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        
        let skybox_pipeline = {
            let vertex_shader = utils::load_spv("shaders/unit_cube.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/skybox.frag.spv").unwrap();

            let vertex_module = context
                .create_shader_module(&vertex_shader, "unit_cube_vertex_shader");
            let fragment_module = context
                .create_shader_module(&fragment_shader, "skybox_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("skybox_pipeline", &render::RasterPipelineDesc {
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
                    cull_mode: vk::CullModeFlags::FRONT,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: Self::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: Some(render::DepthState {
                    format: Self::DEPTH_FORMAT,
                    test: false,
                    write: false,
                    compare: vk::CompareOp::GREATER,
                }),
                multisample: render::MultisampleCount::None,
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        
        let brdf_integration_pipeline = {
            let vertex_shader = utils::load_spv("shaders/blit.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/brdf_integration.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "blit_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "blit_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("brdf_integration_pipeline", &render::RasterPipelineDesc {
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
                    format: vk::Format::R16G16B16A16_SFLOAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: None,
                multisample: render::MultisampleCount::None,
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        let brdf_integration_map = context.create_image("brdf_integration_map", &render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: vk::Format::R16G16B16A16_SFLOAT,
            dimensions: [512, 512, 1],
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        context.record_and_submit(|cmd| {
            cmd.barrier(&[], &[render::image_barrier(
                &brdf_integration_map,
                render::AccessKind::None,
                render::AccessKind::ColorAttachmentWrite
            )], &[]);

            let color_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(brdf_integration_map.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::builder()
                .render_area(brdf_integration_map.full_rect())
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));
        
            cmd.begin_rendering(&rendering_info);
            cmd.bind_raster_pipeline(brdf_integration_pipeline);
            cmd.draw(0..6, 0..1);
            cmd.end_rendering();

            cmd.barrier(&[], &[render::image_barrier(
                &brdf_integration_map,
                render::AccessKind::ColorAttachmentWrite,
                render::AccessKind::AllGraphicsRead
            )], &[]);
        });

        let mut gpu_assets = GpuAssetStore::new(context);
        let mut scene = SceneBuffer::new(context);
        
        if let Some(gltf_path) = gltf_path {
            load_gltf(gltf_path, context, &mut gpu_assets, &mut scene).unwrap();
        }

        scene.update_instances(context);
        scene.update_submeshes(context, &gpu_assets);

        let camera = Camera {
            transform: Transform {
                position: vec3(0.0, 2.0, 0.0),
                ..Default::default()
            },
            projection: Projection::Perspective { fov: 90f32.to_radians(), near_clip: 0.03 },
            // projection: Projection::Orthographic { half_width: 32.0, near_clip: 0.0, far_clip: 1000.0 },
        };
        scene.update_instances(context);

        let equirectangular_cube_map_loader = EnvironmentMapLoader::new(context);

        let equirectangular_environment_image = {
            let (image_binary, image_format) = gltf_loader::load_image_data("C:\\Users\\thefe\\Downloads\\spree_bank_4k.dds")
                .unwrap();
            let mut image =
                gltf_loader::load_image(context, "environment_map".into(), &image_binary, image_format, true, true);
            
            image.set_sampler_flags(render::SamplerKind::LinearRepeat);

            image
        };

        let environment_map = equirectangular_cube_map_loader
            .create_from_equirectangular_image(&context, "environment_map".into(), 1024, &equirectangular_environment_image);

        context.destroy_image(&equirectangular_environment_image);

        Self {
            gpu_assets,
            scene,
            debug_line_renderer: DebugLineRenderer::new(context),
            scene_draw_gen: SceneDrawGen::new(context),
            shadow_map_renderer: ShadowMapRenderer::new(context),
            scree_blitter: ScreenPostProcess::new(context),
            equirectangular_cube_map_loader,

            forward_pipeline,
            skybox_pipeline,
            brdf_integration_pipeline,

            environment_map,
            brdf_integration_map,

            camera,
            camera_controller: CameraController::new(1.0, 0.003),
            mock_camera: camera,
            mock_camera_controller: CameraController::new(1.0, 0.003),
            sun_light: Camera {
                transform: Transform::default(),
                projection: Projection::Orthographic { half_width: 20.0, near_clip: -20.0, far_clip: 20.0 },
            },
            light_dir_controller: CameraController::new(1.0, 0.003),

            render_mode: 0,
            camera_exposure: 1.0,
            light_color: Vec3::splat(1.0),
            light_intensitiy: 32.0,
            max_shadow_distance: 80.0,
            frustum_split_lambda: 0.8,
            selected_cascade: 0,
            view_mock_camera: false,

            use_mock_camera: false,
            show_cascade_view_frustum: false,
            show_cascade_light_frustum: false,
            open_scene_editor_open: false,
            open_profiler: false,
            open_graph_debugger: false,
            open_camera_light_editor: false,
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

        self.view_mock_camera = input.key_held(KeyCode::V);

        if self.view_mock_camera {
            if input.mouse_held(MouseButton::Right) {
                self.mock_camera_controller.update_look(input, &mut self.mock_camera.transform);
            }
            self.mock_camera_controller.update_movement(input, delta_time, &mut self.mock_camera.transform);
        } else {
            if input.mouse_held(MouseButton::Right) {
                self.camera_controller.update_look(input, &mut self.camera.transform);
            }
            self.camera_controller.update_movement(input, delta_time, &mut self.camera.transform);
        }

        
        if input.mouse_held(MouseButton::Middle) {
            self.light_dir_controller.update_look(input, &mut self.sun_light.transform);
        }

        if input.key_pressed(KeyCode::F1) {
            self.open_scene_editor_open = !self.open_scene_editor_open;
        }
        if input.key_pressed(KeyCode::F2) {
            self.open_graph_debugger = !self.open_graph_debugger;
        }
        if input.key_pressed(KeyCode::F3) {
            self.open_profiler = !self.open_profiler;
        }
        if input.key_pressed(KeyCode::F4) {
            self.open_camera_light_editor = !self.open_camera_light_editor;
        }

        fn drag_vec4(ui: &mut egui::Ui, label: &str, vec: &mut Vec4, speed: f32) {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(egui::DragValue::new(&mut vec.x).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.y).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.z).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.w).speed(speed));
            });
        }

        fn drag_quat(ui: &mut egui::Ui, label: &str, vec: &mut Quat, speed: f32) {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(egui::DragValue::new(&mut vec.x).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.y).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.z).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.w).speed(speed));
            });
        }

        fn drag_vec3(ui: &mut egui::Ui, label: &str, vec: &mut Vec3, speed: f32) {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(egui::DragValue::new(&mut vec.x).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.y).speed(speed));
                ui.add(egui::DragValue::new(&mut vec.z).speed(speed));
            });
        }

        fn drag_float(ui: &mut egui::Ui, label: &str, float: &mut f32, speed: f32) {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(egui::DragValue::new(float).speed(speed));
            });
        }

        if self.open_scene_editor_open {
            egui::Window::new("scene").open(&mut self.open_scene_editor_open).vscroll(true).show(egui_ctx, |ui| {
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
            self.render_mode = new_render_mode;
        }

        fn draw_graph_info(ui: &mut egui::Ui, graph: &render::CompiledRenderGraph) {
            for (i, batch) in graph.iter_batches().enumerate() {
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
                            ui.label(pass.name.as_ref());
                        }
                    });
                });
            }
        }

        if self.open_graph_debugger {
            // temporary
            egui::Window::new("rendergraph debugger")
                .open(&mut self.open_graph_debugger)
                .show(egui_ctx, |ui| draw_graph_info(ui, &context.compiled_graph));
        }

        if self.open_profiler {
            self.open_profiler = puffin_egui::profiler_window(&egui_ctx);
        }

        if input.close_requested() | input.key_pressed(KeyCode::Escape) {
            control_flow.set_exit();
        }
    }

    fn render(&mut self, context: &mut render::Context, egui_ctx: &egui::Context) {
        puffin::profile_function!();

        let screen_extent = context.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        let view_projection_matrix = self.camera.compute_matrix(aspect_ratio);
        let skybox_view_projection_matrix = self.camera.projection.compute_matrix(aspect_ratio) *
            Mat4::from_quat(self.camera.transform.orientation).inverse();
        
        let main_camera = if self.use_mock_camera {
            &self.mock_camera
        } else {
            &self.camera
        };

        let view_matrix = main_camera.transform.compute_matrix().inverse();

        let per_frame_data = context.transient_storage_data("per_frame_data", bytemuck::bytes_of(&PerFrameData {
            view_projection: view_projection_matrix,
            view: view_matrix,
            view_pos: self.camera.transform.position,
            render_mode: self.render_mode,
        }));
        
        let target_image = context.create_transient_image("target_image", render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: Self::COLOR_FORMAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels: 1,
            samples: Self::MULTISAMPLING,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        let depth_image = context.create_transient_image("depth_image", render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: Self::DEPTH_FORMAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels: 1,
            samples: Self::MULTISAMPLING,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            aspect: vk::ImageAspectFlags::DEPTH,
        });
        
        let vertex_buffer = context
            .import_buffer("mesh_vertex_buffer", &self.gpu_assets.vertex_buffer, &Default::default());
        let index_buffer = context
            .import_buffer("mesh_index_buffer", &self.gpu_assets.index_buffer, &Default::default());
        let material_buffer = context
            .import_buffer("material_buffer", &self.gpu_assets.material_buffer, &Default::default());
        let entity_buffer = context
            .import_buffer("entity_instance_buffer", &self.scene.entity_data_buffer, &Default::default());

        let submeshes = context
            .import_buffer("scene_submesh_buffer", &self.scene.submesh_buffer, &Default::default());
        let mesh_infos = context
            .import_buffer("mesh_info_buffer", &self.gpu_assets.mesh_info_buffer, &Default::default());
        let submesh_count = self.scene.submesh_data.len() as u32;

        let skybox_image = context.import_image("skybox", &self.environment_map.skybox, &Default::default());
        let irradiance_image = context.import_image("irradiance_image", &self.environment_map.irradiance, &Default::default());
        let prefiltered_env_map = context.import_image("prefiltered_env_map", &self.environment_map.prefiltered, &Default::default());
        let brdf_integration_map = context.import_image("brdf_integration_map", &self.brdf_integration_map, &Default::default());

        let draw_commands_buffer = self.scene_draw_gen
            .create_draw_commands(context, submesh_count, submeshes, mesh_infos, "main_draw_commands".into());

        let shadow_map_draw_commands = self.scene_draw_gen
            .create_draw_commands(context, submesh_count, submeshes, mesh_infos, "shadow_map_draw_commands".into());

        let light_direction = -self.sun_light.transform.orientation.mul_vec3(vec3(0.0, 0.0, -1.0));

        let inv_light_direction = self.sun_light.transform.orientation.inverse();
        
        let mut directional_light_data = GpuDirectionalLight {
            projection_matrices: bytemuck::Zeroable::zeroed(),
            shadow_maps: bytemuck::Zeroable::zeroed(),
            color: self.light_color,
            intensity: self.light_intensitiy,
            direction: light_direction,
            _padding: 0,
        };
        
        let lambda = self.frustum_split_lambda;

        let shadow_projections: [Mat4; MAX_SHADOW_CASCADE_COUNT] = std::array::from_fn(|i| {
            let Projection::Perspective { fov, near_clip } = main_camera.projection else { todo!() };
            let far_clip = self.max_shadow_distance;
            
            let near = frustum_split(near_clip, far_clip, lambda, i as f32 / MAX_SHADOW_CASCADE_COUNT as f32);
            let far = frustum_split(near_clip, far_clip, lambda, (i+1) as f32 / MAX_SHADOW_CASCADE_COUNT as f32);

            let light_matrix = Mat4::from_quat(inv_light_direction);
            let view_to_world_matrix = main_camera.transform.compute_matrix();

            let subfrustum_corners_view_space = perspective_corners(fov, aspect_ratio, near, far);
            
            let mut subfrustum_center_view_space = Vec4::ZERO;
            for corner in subfrustum_corners_view_space.iter().copied() {
                subfrustum_center_view_space += corner;
            }
            subfrustum_center_view_space /= 8.0;
            
            let mut radius_sqr = 0.0;
            for corner in subfrustum_corners_view_space.iter().copied() {
                let corner = Vec3A::from(corner);
                radius_sqr = f32::max(radius_sqr, corner.distance_squared(subfrustum_center_view_space.into()));
            }
            let radius = radius_sqr.sqrt();

            let view_to_light_matrix = light_matrix * view_to_world_matrix;
            let subfrustum_center_light_space = view_to_light_matrix * subfrustum_center_view_space;
            let subfrustum_center_light_space = Vec3A::from(subfrustum_center_light_space)
                / subfrustum_center_light_space.w;

            let texel_size_vs = radius * 2.0 / SHADOW_RESOLUTION as f32;
            
            // texel alignment
            let subfrustum_center_light_space = (subfrustum_center_light_space / texel_size_vs).floor() * texel_size_vs;

            let mut max_extent = Vec3A::splat(radius);
            let mut min_extent = -max_extent;

            max_extent += subfrustum_center_light_space;
            min_extent += subfrustum_center_light_space;

            let light_projection = Mat4::orthographic_rh(
                min_extent.x, max_extent.x,
                min_extent.y, max_extent.y,
                150.0, -150.0,
            );

            if self.show_cascade_view_frustum && self.selected_cascade == i {
                let subfrustum_corners_w = subfrustum_corners_view_space.map(|v| view_to_world_matrix * v);
                self.debug_line_renderer.draw_frustum(&subfrustum_corners_w, Vec4::splat(1.0));
            }
            
            if self.show_cascade_light_frustum && self.selected_cascade == i {
                let cascade_frustum_corners = frustum_corners_from_matrix(&light_projection);
                self.debug_line_renderer.draw_frustum(&cascade_frustum_corners, vec4(1.0, 1.0, 0.0, 1.0));
            }
            
            light_projection * light_matrix
        });

        let shadow_maps: [render::GraphImageHandle; MAX_SHADOW_CASCADE_COUNT] = std::array::from_fn(|i| {
            self.shadow_map_renderer.render_shadow_map(
                format!("sun_cascade_{i}").into(),
                context,
                SHADOW_RESOLUTION,
                shadow_projections[i],
                shadow_map_draw_commands,
                vertex_buffer,
                index_buffer,
                entity_buffer
            )
        });

        directional_light_data.projection_matrices = shadow_projections;
        directional_light_data.shadow_maps = shadow_maps.map(|map| context
            .get_transient_resource_descriptor_index(map)
            .unwrap()
        );

        let directional_light_buffer = context
            .transient_storage_data("directional_light_data", bytemuck::bytes_of(&directional_light_data));

        egui::Window::new("camera_and_lighting").open(&mut self.open_camera_light_editor).show(egui_ctx, |ui| {
            ui.checkbox(&mut self.use_mock_camera, "use mock camera");
            ui.checkbox(&mut self.show_cascade_view_frustum, "use cascade view frustum");
            ui.checkbox(&mut self.show_cascade_light_frustum, "use cascade light frustum");

            ui.horizontal(|ui| {
                ui.label("camera_exposure");
                ui.add(egui::DragValue::new(&mut self.camera_exposure).speed(0.05).clamp_range(0.1..=20.0));
            });

            ui.horizontal(|ui| {
                ui.label("light_color");
                let mut array = self.light_color.to_array();
                ui.color_edit_button_rgb(&mut array);
                self.light_color = Vec3::from_array(array);
            });
            ui.horizontal(|ui| {
                ui.label("light_intensitiy");
                ui.add(egui::DragValue::new(&mut self.light_intensitiy).speed(0.4));
            });
            
            let [constant_factor, clamp, slope_factor] = &mut self.shadow_map_renderer.depth_bias;

            ui.horizontal(|ui| {
                ui.label("depth_bias_constant_factor");
                ui.add(egui::DragValue::new(constant_factor).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("depth_bias_clamp");
                ui.add(egui::DragValue::new(clamp).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("depth_bias_slope_factor");
                ui.add(egui::DragValue::new(slope_factor).speed(0.01));
            });

            ui.horizontal(|ui| {
                ui.label("max_shadow_distance");
                ui.add(egui::DragValue::new(&mut self.max_shadow_distance).speed(1.0));
            });

            ui.horizontal(|ui| {
                ui.label("lambda");
                ui.add(egui::Slider::new(&mut self.frustum_split_lambda, 0.0..=1.0));
            });

            ui.horizontal(|ui| {
                ui.label("selected_cascade");
                ui.add(egui::Slider::new(&mut self.selected_cascade, 0..=MAX_SHADOW_CASCADE_COUNT - 1));
            });

            ui.image(shadow_maps[self.selected_cascade], egui::Vec2::new(250.0, 250.0));
        });

        directional_light_data.direction = light_direction;
        directional_light_data.color = self.light_color;
        directional_light_data.intensity = self.light_intensitiy;

        let forward_pipeline = self.forward_pipeline;
        let skybox_pipeline = self.skybox_pipeline;

        context.add_pass("forward_pass")
            .with_dependency(target_image, render::AccessKind::ColorAttachmentWrite)
            .with_dependency(depth_image, render::AccessKind::DepthAttachmentWrite)
            .with_dependency(draw_commands_buffer, render::AccessKind::IndirectBuffer)
            .with_dependencies(shadow_maps.map(|h| (h, render::AccessKind::FragmentShaderRead)))
            .render(move |cmd, graph| {
                let target_image = graph.get_image(target_image);
                let depth_image = graph.get_image(depth_image);

                let skybox_image = graph.get_image(skybox_image);
                let irradiance_image = graph.get_image(irradiance_image);
                let prefiltered_env_map = graph.get_image(prefiltered_env_map);
                let brdf_integration_map = graph.get_image(brdf_integration_map);

                let per_frame_data = graph.get_buffer(per_frame_data);
                
                let vertex_buffer = graph.get_buffer(vertex_buffer);
                let index_buffer = graph.get_buffer(index_buffer);
                let material_buffer = graph.get_buffer(material_buffer);

                let instance_buffer = graph.get_buffer(entity_buffer);
                let draw_commands_buffer = graph.get_buffer(shadow_map_draw_commands);
                let directional_light_buffer = graph.get_buffer(directional_light_buffer);

                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(target_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE);

                // if let Some(resolve_image) = resolve_image {
                //     color_attachment = color_attachment
                //         .resolve_image_view(resolve_image.view)
                //         .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                //         .resolve_mode(vk::ResolveModeFlags::AVERAGE);
                // }

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
                
                cmd.bind_raster_pipeline(skybox_pipeline);

                cmd.build_constants()
                    .mat4(&skybox_view_projection_matrix)
                    .image(&skybox_image);

                cmd.draw(0..36, 0..1);

                cmd.bind_raster_pipeline(forward_pipeline);
                cmd.bind_index_buffer(&index_buffer, 0);

                cmd.build_constants()
                    .buffer(&per_frame_data)
                    .buffer(&vertex_buffer)
                    .buffer(&instance_buffer)
                    .buffer(&draw_commands_buffer)
                    .buffer(&material_buffer)
                    .buffer(&directional_light_buffer)
                    .image(&irradiance_image)
                    .image(&prefiltered_env_map)
                    .image(&brdf_integration_map);

                cmd.draw_indexed_indirect_count(
                    draw_commands_buffer,
                    4,
                    draw_commands_buffer,
                    0,
                    MAX_DRAW_COUNT as u32,
                    std::mem::size_of::<GpuDrawCommand>() as u32,
                );

                cmd.end_rendering();
            });

        self.scree_blitter.render(context, target_image, self.camera_exposure);
        self.debug_line_renderer.render(context, context.get_swapchain_image(), None, depth_image, view_projection_matrix);
    }

    fn destroy(&self, context: &render::Context) {
        self.gpu_assets.destroy(context);
        self.scene.destroy(context);
        self.debug_line_renderer.destroy(context);
        self.scene_draw_gen.destroy(context);
        self.shadow_map_renderer.destroy(context);
        self.scree_blitter.destroy(context);
        self.equirectangular_cube_map_loader.destroy(context);
        self.environment_map.destroy(context);
        context.destroy_image(&self.brdf_integration_map);
        context.destroy_pipeline(&self.forward_pipeline);
        context.destroy_pipeline(&self.skybox_pipeline);
        context.destroy_pipeline(&self.brdf_integration_pipeline);
    }
}

pub struct ScreenPostProcess {
    pipeline: render::RasterPipeline,
}

impl ScreenPostProcess {
    pub fn new(context: &render::Context) -> Self {
        let pipeline = {
            let vertex_shader = utils::load_spv("shaders/blit.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/blit.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "blit_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "blit_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("blit_pipeline", &render::RasterPipelineDesc {
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
                    format: context.swapchain.format(),
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: None,
                multisample: render::MultisampleCount::None,
                dynamic_states: &[],
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        Self { pipeline }
    }

    pub fn render(
        &self,
        context: &mut render::Context,
        src_image: render::GraphImageHandle,
        exposure: f32,
    ) {
        let swapchain_image = context.get_swapchain_image();
        let pipeline = self.pipeline;

        context.add_pass("blit_present_pass")
            .with_dependency(swapchain_image, render::AccessKind::ColorAttachmentWrite)
            .with_dependency(src_image, render::AccessKind::FragmentShaderRead)
            .render(move |cmd, graph| {
                let swapchain_image = graph.get_image(swapchain_image);
                let src_image = graph.get_image(src_image);

                let color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(swapchain_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(swapchain_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));
            
                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(pipeline);

                cmd.build_constants()
                    .image(&src_image)
                    .float(exposure);

                cmd.draw(0..6, 0..1);

                cmd.end_rendering();
            });
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
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
        frame_ctx: &mut render::Context,
        scene_submesh_count: u32,
        scene_submeshes: render::GraphBufferHandle,
        mesh_infos: render::GraphBufferHandle,
        draw_commands_name: Cow<'static, str>,
    ) -> render::GraphBufferHandle {
        let pass_name = format!("{draw_commands_name}_generation");
        let draw_commands = frame_ctx.create_transient_buffer(draw_commands_name, render::BufferDesc {
            size: MAX_DRAW_COUNT * std::mem::size_of::<GpuDrawCommand>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        });
        let pipeline = self.pipeline;

        use render::AccessKind;

        frame_ctx.add_pass(pass_name)
            .with_dependency(scene_submeshes, AccessKind::ComputeShaderRead)
            .with_dependency(mesh_infos, AccessKind::ComputeShaderRead)
            .with_dependency(draw_commands, AccessKind::ComputeShaderWrite)
            .render(move |cmd, graph| {
                let scene_submeshes = graph.get_buffer(scene_submeshes);
                let mesh_infos = graph.get_buffer(mesh_infos);
                let draw_commands = graph.get_buffer(draw_commands);

                cmd.bind_compute_pipeline(pipeline);

                cmd.build_constants()
                    .buffer(&scene_submeshes)
                    .buffer(&mesh_infos)
                    .buffer(&draw_commands);

                cmd.dispatch([scene_submesh_count / 256 + 1, 1, 1]);
            });
        
        draw_commands
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
    }
}

pub struct ShadowMapRenderer {
    pipeline: render::RasterPipeline,
    depth_bias: [f32; 3],
}

impl ShadowMapRenderer {
    pub fn new(context: &render::Context) -> Self {
        let pipeline = {
            let vertex_shader = utils::load_spv("shaders/shadow.vert.spv").unwrap();
            let vertex_module = context.create_shader_module(&vertex_shader, "shadow_vertex_shader");
            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("shadowmap_renderer_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: None,
                vertex_input: render::VertexInput::default(),
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::BACK,
                    depth_bias: Some(render::DepthBias::default()),
                    depth_clamp: true,
                },
                color_attachments: &[],
                depth_state: Some(render::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: true,
                    write: true,
                    compare: vk::CompareOp::GREATER,
                }),
                multisample: render::MultisampleCount::None,
                dynamic_states: &[vk::DynamicState::DEPTH_BIAS]
            });

            context.destroy_shader_module(vertex_module);

            pipeline
        };

        Self { pipeline, depth_bias: [-4.0, 0.0, -4.0] }
    }

    pub fn render_shadow_map(
        &self,
        name: Cow<'static, str>,
        frame_ctx: &mut render::Context,

        resolution: u32,
        view_projection: Mat4,
        draw_commands: render::GraphBufferHandle,

        vertex_buffer: render::GraphBufferHandle,
        index_buffer: render::GraphBufferHandle,
        entity_data: render::GraphBufferHandle,
    ) -> render::GraphImageHandle {
        let pass_name = format!("shadow_pass_for_{name}");
        let shadow_map = frame_ctx.create_transient_image(name, render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: App::DEPTH_FORMAT,
            dimensions: [resolution, resolution, 1],
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::DEPTH,
        });

        let pipeline = self.pipeline;

        let depth_bias = self.depth_bias;

        frame_ctx.add_pass(pass_name)
            .with_dependency(shadow_map, render::AccessKind::DepthAttachmentWrite)
            .with_dependency(draw_commands, render::AccessKind::IndirectBuffer)
            .render(move |cmd, graph| {
                let shadow_map = graph.get_image(shadow_map);
                
                let vertex_buffer = graph.get_buffer(vertex_buffer);
                let index_buffer = graph.get_buffer(index_buffer);
                let entity_buffer = graph.get_buffer(entity_data);
                let draw_commands_buffer = graph.get_buffer(draw_commands);

                let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
                    .image_view(shadow_map.view)
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
                    .render_area(shadow_map.full_rect())
                    .layer_count(1)
                    .depth_attachment(&depth_attachemnt);

                cmd.begin_rendering(&rendering_info);
                
                cmd.bind_raster_pipeline(pipeline);
                cmd.bind_index_buffer(&index_buffer, 0);

                let [constant_factor, clamp, slope_factor] = depth_bias;
                cmd.set_depth_bias(constant_factor, clamp, slope_factor);

                cmd.build_constants()
                    .mat4(&view_projection)
                    .buffer(&vertex_buffer)
                    .buffer(&entity_buffer);

                cmd.draw_indexed_indirect_count(
                    draw_commands_buffer,
                    4,
                    draw_commands_buffer,
                    0,
                    MAX_DRAW_COUNT as u32,
                    std::mem::size_of::<GpuDrawCommand>() as u32,
                );

                cmd.end_rendering();
            });

        shadow_map
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct DebugLineVertex {
    position: Vec3,
    color: [u8; 4],
}

pub struct DebugLineRenderer {
    pipeline: render::RasterPipeline,
    line_buffer: render::Buffer,
    vertex_cursor: usize,
    frame_index: usize,
}

impl DebugLineRenderer {
    pub const MAX_VERTEX_COUNT: usize = 1_000_000;
    const CIRCLE_LINE_SEGMENTS: usize = 24;

    pub fn new(context: &render::Context) -> Self {
        let pipeline = {
            let vertex_shader = utils::load_spv("shaders/debug_line.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/debug_line.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "debug_line_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "debug_line_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("basic_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: Some(render::ShaderStage {
                    module: fragment_module,
                    entry,
                }),
                vertex_input: render::VertexInput {
                    bindings: &[vk::VertexInputBindingDescription {
                        binding: 0,
                        stride: std::mem::size_of::<DebugLineVertex>() as u32,
                        input_rate: vk::VertexInputRate::VERTEX,
                    }],
                    attributes: &[
                        vk::VertexInputAttributeDescription {
                            location: 0,
                            binding: 0,
                            format: vk::Format::R32G32B32_SFLOAT,
                            offset: bytemuck::offset_of!(DebugLineVertex, position) as u32,
                        },
                        vk::VertexInputAttributeDescription {
                            location: 1,
                            binding: 0,
                            format: vk::Format::R8G8B8A8_UNORM,
                            offset: bytemuck::offset_of!(DebugLineVertex, color) as u32,
                        },
                    ],
                },
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::LINE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: context.swapchain.format(),
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: Some(render::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: true,
                    write: false,
                    compare: vk::CompareOp::GREATER_OR_EQUAL,
                }),
                multisample: App::MULTISAMPLING,
                dynamic_states: &[vk::DynamicState::DEPTH_TEST_ENABLE]
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        let line_buffer = context.create_buffer("debug_line_buffer", &render::BufferDesc {
            size: render::FRAME_COUNT * Self::MAX_VERTEX_COUNT * std::mem::size_of::<DebugLineVertex>(),
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
        });

        Self { pipeline, line_buffer, vertex_cursor: 0, frame_index: 0 }
    }

    fn remainin_vertex_space(&self) -> usize {
        Self::MAX_VERTEX_COUNT * (self.frame_index + 1) - self.vertex_cursor
    }

    pub fn add_vertices(&mut self, vertices: &[DebugLineVertex]) {
        assert!(self.remainin_vertex_space() >= vertices.len());
        unsafe {
            let dst_ptr = self.line_buffer.mapped_ptr.unwrap()
                .as_ptr()
                .cast::<DebugLineVertex>()
                .add(Self::MAX_VERTEX_COUNT * self.frame_index + self.vertex_cursor);
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), dst_ptr, vertices.len());
        }
        self.vertex_cursor += vertices.len();
    }

    pub fn draw_line(&mut self, start: Vec3, end: Vec3, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);
        self.add_vertices(&[
            DebugLineVertex { position: start, color },
            DebugLineVertex { position: end, color },
        ]);
    }

    pub fn draw_frustum(&mut self, corners: &[Vec4; 8], color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        self.add_vertices(&[
            DebugLineVertex { position: corners[0].truncate(), color },
            DebugLineVertex { position: corners[1].truncate(), color },
            DebugLineVertex { position: corners[1].truncate(), color },
            DebugLineVertex { position: corners[2].truncate(), color },
            DebugLineVertex { position: corners[2].truncate(), color },
            DebugLineVertex { position: corners[3].truncate(), color },
            DebugLineVertex { position: corners[3].truncate(), color },
            DebugLineVertex { position: corners[0].truncate(), color },
            
            DebugLineVertex { position: corners[4].truncate(), color },
            DebugLineVertex { position: corners[5].truncate(), color },
            DebugLineVertex { position: corners[5].truncate(), color },
            DebugLineVertex { position: corners[6].truncate(), color },
            DebugLineVertex { position: corners[6].truncate(), color },
            DebugLineVertex { position: corners[7].truncate(), color },
            DebugLineVertex { position: corners[7].truncate(), color },
            DebugLineVertex { position: corners[4].truncate(), color },

            DebugLineVertex { position: corners[0].truncate(), color },
            DebugLineVertex { position: corners[4].truncate(), color },
            DebugLineVertex { position: corners[1].truncate(), color },
            DebugLineVertex { position: corners[5].truncate(), color },
            DebugLineVertex { position: corners[2].truncate(), color },
            DebugLineVertex { position: corners[6].truncate(), color },
            DebugLineVertex { position: corners[3].truncate(), color },
            DebugLineVertex { position: corners[7].truncate(), color },
        ]);
    }

    pub fn draw_cross(&mut self, pos: Vec3, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        self.add_vertices(&[
            DebugLineVertex { position: pos - vec3( 1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3( 1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos - vec3(-1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3(-1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos - vec3( 1.0, -1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3( 1.0, -1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos - vec3(-1.0, -1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3(-1.0, -1.0, 1.0) * 0.01, color },
        ]);
    }

    pub fn draw_sphere(&mut self, pos: Vec3, radius: f32, color: Vec4) {
        self.draw_cross(pos, color);
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        let mut circle_vertices = [
            DebugLineVertex { position: Vec3::ZERO, color };
            Self::CIRCLE_LINE_SEGMENTS * 2
        ];

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(theta.sin() * radius, theta.cos() * radius, 0.0);
        }
        self.add_vertices(&circle_vertices);

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(theta.sin() * radius, 0.0, theta.cos() * radius);
        }
        self.add_vertices(&circle_vertices);

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(0.0, theta.sin() * radius, theta.cos() * radius);
        }
        self.add_vertices(&circle_vertices);
    
    }

    pub fn render(
        &mut self,
        frame_ctx: &mut render::Context,
        target_image: render::GraphImageHandle,
        resolve_image: Option<render::GraphImageHandle>,
        depth_image: render::GraphImageHandle,
        view_projection: Mat4,
    ) {
        let line_buffer = frame_ctx.import_buffer("debug_line_buffer", &self.line_buffer, &Default::default());
        let buffer_offset = self.frame_index * Self::MAX_VERTEX_COUNT * std::mem::size_of::<DebugLineVertex>();
        let vertex_count = self.vertex_cursor as u32;
        let pipeline = self.pipeline;
        
        let mut dependencies = vec![
            (target_image, render::AccessKind::ColorAttachmentWrite),
            (depth_image, render::AccessKind::DepthAttachmentRead),
        ];

        if let Some(resolve_image) = resolve_image {
            dependencies.push((resolve_image, render::AccessKind::ColorAttachmentWrite));
        }

        frame_ctx.add_pass("debug_line_render")
            .with_dependency(target_image, render::AccessKind::ColorAttachmentWrite)
            .with_dependency(depth_image, render::AccessKind::DepthAttachmentRead)
            .with_dependencies(resolve_image.map(|h| (h, render::AccessKind::ColorAttachmentWrite)))
            .render(move |cmd, graph| {
                let target_image = graph.get_image(target_image);
                let resolve_image = resolve_image.map(|handle| graph.get_image(handle));
                let depth_image = graph.get_image(depth_image);
                let line_buffer = graph.get_buffer(line_buffer);

                let mut color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(target_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
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
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::NONE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(target_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .depth_attachment(&depth_attachemnt);

                cmd.begin_rendering(&rendering_info);
                cmd.bind_raster_pipeline(pipeline);
                cmd.bind_vertex_buffer(0, &line_buffer, buffer_offset as u64);


                cmd.build_constants()
                    .mat4(&view_projection)
                    .float(0.1);
                cmd.set_depth_test_enable(false);
                cmd.draw(0..vertex_count as u32, 0..1);
                
                cmd.build_constants()
                    .mat4(&view_projection)
                    .float(1.0);
                cmd.set_depth_test_enable(true);
                cmd.draw(0..vertex_count as u32, 0..1);

                cmd.end_rendering();
            });

        self.frame_index = (self.frame_index + 1) % render::FRAME_COUNT;
        self.vertex_cursor = 0;
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
        context.destroy_buffer(&self.line_buffer);
    }
}

pub struct EnvironmentMap {
    pub skybox: render::Image,
    pub irradiance: render::Image,
    pub prefiltered: render::Image,
}

impl EnvironmentMap {
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
                multisample: render::MultisampleCount::None,
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
                multisample: render::MultisampleCount::None,
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
                multisample: render::MultisampleCount::None,
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
        let skybox = context.create_image(format!("{name}_skybox"), &render::ImageDesc {
            ty: render::ImageType::Cube,
            format: vk::Format::R16G16B16A16_SFLOAT,
            dimensions: [resolution, resolution, 1],
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
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
            
            cmd.barrier(&[], &[
                render::image_barrier(
                    &skybox,
                    render::AccessKind::ColorAttachmentWrite,
                    render::AccessKind::AllGraphicsRead,
                ),
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

fn main() {
    utils::init_logger(false);
    puffin::set_scopes_on(true);

    let gltf_path = std::env::args().nth(1);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("orbit")
        .with_resizable(true)
        .build(&event_loop)
        .expect("failed to build window");

    let mut input = Input::new(&window);
    let mut time = Time::new();

    let mut context = render::Context::new(
        window,
        &render::ContextDesc {
            present_mode: vk::PresentModeKHR::IMMEDIATE,
        },
    );

    let egui_ctx = egui::Context::default();
    let mut egui_state = egui_winit::State::new(&event_loop);
    let mut egui_renderer = EguiRenderer::new(&context);

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

            context.begin_frame();

            let swapchain_image = context.get_swapchain_image();

            app.render(&mut context, &egui_ctx);

            let clipped_primitives = {
                puffin::profile_scope!("egui_tessellate");
                egui_ctx.tessellate(full_output.shapes)
            };
            egui_renderer.render(&mut context, &clipped_primitives, &full_output.textures_delta, swapchain_image);

            context.end_frame();

            input.clear_frame()
        }
    })
}
