#![allow(dead_code)]

use std::f32::consts::PI;

use ash::vk;
use gltf_loader::load_gltf;
use assets::GpuAssetStore;

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

mod passes;

mod math;
mod collections;
mod utils;

mod gltf_loader;

mod egui_renderer;

use input::Input;
use egui_renderer::EguiRenderer;
use scene::{Transform, SceneData};

use passes::{
    forward::ForwardRenderer,
    env_map_loader::{EnvironmentMapLoader, EnvironmentMap},
    debug_line_renderer::DebugLineRenderer,
    shadow_renderer::ShadowMapRenderer,
    draw_gen::SceneDrawGen,
    post_process::ScreenPostProcess
};

pub const MAX_DRAW_COUNT: usize = 1_000_000;
pub const MAX_SHADOW_CASCADE_COUNT: usize = 4;
pub const SHADOW_RESOLUTION: u32 = 512 * 1;

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



struct App {
    gpu_assets: GpuAssetStore,
    scene: SceneData,

    forward_renderer: ForwardRenderer,
    debug_line_renderer: DebugLineRenderer,
    scene_draw_gen: SceneDrawGen,
    shadow_map_renderer: ShadowMapRenderer,
    post_process: ScreenPostProcess,
    equirectangular_cube_map_loader: EnvironmentMapLoader,
    
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
    selected_cascade: usize,
    view_mock_camera: bool,

    use_mock_camera: bool,
    show_cascade_view_frustum: bool,
    show_cascade_light_frustum: bool,
    show_bounding_boxes: bool,
    show_frustum_planes: bool,
    open_scene_editor_open: bool,
    open_graph_debugger: bool,
    open_profiler: bool,
    open_camera_light_editor: bool,
    open_cull_debugger: bool,
}

impl App {
    pub const COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
    pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
    pub const MULTISAMPLING: render::MultisampleCount = render::MultisampleCount::None;

    fn new(context: &render::Context, gltf_path: Option<&str>) -> Self {
        let mut gpu_assets = GpuAssetStore::new(context);
        let mut scene = SceneData::new(context);
        
        if let Some(gltf_path) = gltf_path {
            load_gltf(gltf_path, context, &mut gpu_assets, &mut scene).unwrap();
        }

        scene.update_instances(context);
        scene.update_submeshes(context, &gpu_assets);

        
        let equirectangular_cube_map_loader = EnvironmentMapLoader::new(context);

        let equirectangular_environment_image = {
            let (image_binary, image_format) =
                gltf_loader::load_image_data("D:\\dev\\models\\fbx\\Bistro_v5_2\\san_giuseppe_bridge_4k.dds")
                .unwrap();
            let mut image =
                gltf_loader::load_image(context, "environment_map".into(), &image_binary, image_format, true, true);
            
            image.set_sampler_flags(render::SamplerKind::LinearRepeat);

            image
        };  

        let environment_map = equirectangular_cube_map_loader
            .create_from_equirectangular_image(&context, "environment_map".into(), 1024, &equirectangular_environment_image);

        context.destroy_image(&equirectangular_environment_image);

        let camera = Camera {
            transform: Transform {
                position: vec3(0.0, 2.0, 0.0),
                ..Default::default()
            },
            projection: Projection::Perspective { fov: 90f32.to_radians(), near_clip: 0.01 },
        };

        Self {
            gpu_assets,
            scene,

            forward_renderer: ForwardRenderer::new(context),
            debug_line_renderer: DebugLineRenderer::new(context),
            scene_draw_gen: SceneDrawGen::new(context),
            shadow_map_renderer: ShadowMapRenderer::new(context),
            post_process: ScreenPostProcess::new(context),
            equirectangular_cube_map_loader,

            environment_map,

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
            selected_cascade: 0,
            view_mock_camera: false,

            use_mock_camera: false,
            show_cascade_view_frustum: false,
            show_cascade_light_frustum: false,
            show_bounding_boxes: false,
            show_frustum_planes: false,
            open_scene_editor_open: false,
            open_graph_debugger: false,
            open_profiler: false,
            open_camera_light_editor: false,
            open_cull_debugger: false,
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
        if input.key_pressed(KeyCode::F5) {
            self.open_cull_debugger = !self.open_cull_debugger;
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

        if self.open_graph_debugger {
            self.open_graph_debugger = context.graph_debugger(egui_ctx);
        }

        if self.open_profiler {
            self.open_profiler = puffin_egui::profiler_window(egui_ctx);
        }

        if self.open_cull_debugger {
            egui::Window::new("culling debugger").open(&mut self.open_cull_debugger).show(egui_ctx, |ui| {
                ui.checkbox(&mut self.show_bounding_boxes, "show bounding boxes");
                ui.checkbox(&mut self.show_frustum_planes, "show frustum planes");
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

        if input.close_requested() | input.key_pressed(KeyCode::Escape) {
            control_flow.set_exit();
        }
    }

    fn render(&mut self, context: &mut render::Context, egui_ctx: &egui::Context) {
        puffin::profile_function!();
        
        let assets = self.gpu_assets.import_to_graph(context);
        let scene = self.scene.import_to_graph(context);
        
        let screen_extent = context.swapchain_extent();
        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        let focused_camera = if self.use_mock_camera {
            &self.mock_camera
        } else {
            &self.camera
        };

        let focused_camera_view_projection = focused_camera.compute_matrix(aspect_ratio);
        let camera_view_projection = self.camera.compute_matrix(aspect_ratio);

        if self.use_mock_camera {
            let Projection::Perspective { fov, near_clip } = focused_camera.projection else { todo!() };
            let far_clip = self.shadow_map_renderer.settings.max_shadow_distance;
            let frustum_corner = math::perspective_corners(fov, aspect_ratio, near_clip, far_clip)
                .map(|corner| focused_camera.transform.compute_matrix() * corner);
            self.debug_line_renderer.draw_frustum(&frustum_corner, vec4(1.0, 1.0, 1.0, 1.0));
        }

        let color_target = context.create_transient_image("color_target", render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: Self::COLOR_FORMAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels: 1,
            samples: Self::MULTISAMPLING,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        let depth_target = context.create_transient_image("depth_target", render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: Self::DEPTH_FORMAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels: 1,
            samples: Self::MULTISAMPLING,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            aspect: vk::ImageAspectFlags::DEPTH,
        });

        let directional_light = self.shadow_map_renderer.render_directional_light(
            context,
            "sun".into(),
            self.sun_light.transform.orientation,
            self.light_color,
            self.light_intensitiy,
            &self.camera,
            aspect_ratio,
            &self.scene_draw_gen,
            assets,
            scene
        );

        let forwad_draw_commands = self.scene_draw_gen.create_draw_commands(
            context,
            "forward_draw_commands".into(),
            &focused_camera_view_projection,
            assets,
            scene
        );

        self.forward_renderer.render(
            context,

            forwad_draw_commands,
            color_target,
            depth_target,

            &self.camera,
            self.render_mode,

            &self.environment_map,
            directional_light,
            
            assets,
            scene,
        );

        let swapchain_image = context.get_swapchain_image();

        self.post_process.render(context, color_target, self.camera_exposure);
        self.debug_line_renderer.render(context, swapchain_image, None, depth_target, camera_view_projection);

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
            
            let shadow_settings = &mut self.shadow_map_renderer.settings;

            ui.horizontal(|ui| {
                ui.label("depth_bias_constant_factor");
                ui.add(egui::DragValue::new(&mut shadow_settings.depth_bias_constant_factor).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("depth_bias_clamp");
                ui.add(egui::DragValue::new(&mut shadow_settings.depth_bias_clamp).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("depth_bias_slope_factor");
                ui.add(egui::DragValue::new(&mut shadow_settings.depth_bias_slope_factor).speed(0.01));
            });

            ui.horizontal(|ui| {
                ui.label("max_shadow_distance");
                ui.add(egui::DragValue::new(&mut shadow_settings.max_shadow_distance).speed(1.0));
            });

            ui.horizontal(|ui| {
                ui.label("lambda");
                ui.add(egui::Slider::new(&mut shadow_settings.cascade_split_lambda, 0.0..=1.0));
            });

            ui.horizontal(|ui| {
                ui.label("selected_cascade");
                ui.add(egui::Slider::new(&mut self.selected_cascade, 0..=MAX_SHADOW_CASCADE_COUNT - 1));
            });

            ui.image(directional_light.shadow_maps[self.selected_cascade], egui::Vec2::new(250.0, 250.0));
        });

        if self.show_bounding_boxes {
            for entity in self.scene.entities.iter() {
                if let Some(model) = entity.model {
                    let transform_matrix = entity.transform.compute_matrix();
                    for submesh in self.gpu_assets.models[model].submeshes.iter() {
                        let aabb = self.gpu_assets.mesh_infos[submesh.mesh_handle].aabb;
                        let corners = [
                            vec3a( 0.0, 0.0, 0.0),
                            vec3a( 1.0, 0.0, 0.0),
                            vec3a( 1.0, 1.0, 0.0),
                            vec3a( 0.0, 1.0, 0.0),
                            vec3a( 0.0, 0.0, 1.0),
                            vec3a( 1.0, 0.0, 1.0),
                            vec3a( 1.0, 1.0, 1.0),
                            vec3a( 0.0, 1.0, 1.0),
                        ].map(|s| transform_matrix * (aabb.min + ((aabb.max - aabb.min) * s)).extend(1.0));
                        self.debug_line_renderer.draw_frustum(&corners, vec4(1.0, 1.0, 1.0, 1.0))
                    }
                }
            }
        }
    }

    fn destroy(&self, context: &render::Context) {
        self.gpu_assets.destroy(context);
        self.scene.destroy(context);

        self.forward_renderer.destroy(context);
        self.debug_line_renderer.destroy(context);
        self.scene_draw_gen.destroy(context);
        self.shadow_map_renderer.destroy(context);
        self.post_process.destroy(context);
        self.equirectangular_cube_map_loader.destroy(context);
        self.environment_map.destroy(context);
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
            present_mode: vk::PresentModeKHR::FIFO,
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
