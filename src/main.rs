#![allow(dead_code)]

use std::f32::consts::PI;

use ash::vk;
use assets::GpuAssetStore;
use gltf_loader::load_gltf;

use time::Time;
use winit::{
    event::{Event, MouseButton, VirtualKeyCode as KeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};

mod assets;
mod input;
mod graphics;
mod scene;
mod time;

mod passes;

mod collections;
mod math;
mod utils;

mod gltf_loader;

mod egui_renderer;

use egui_renderer::EguiRenderer;
use input::Input;
use scene::{SceneData, Transform};

use passes::{
    debug_renderer::DebugRenderer,
    env_map_loader::EnvironmentMap,
    forward::{ForwardRenderer, RenderMode}, shadow_renderer::{ShadowSettings, ShadowRenderer},
};

use crate::passes::{
    post_process::render_post_process,
    forward::TargetAttachments
};

pub const MAX_DRAW_COUNT: usize = 1_000_000;
pub const MAX_SHADOW_CASCADE_COUNT: usize = 4;

/// An experimental Vulkan 1.3 renderer
#[derive(Debug, onlyargs_derive::OnlyArgs)]
struct Args {
    /// The gltf scene to be loaded at startup
    scene: Option<std::path::PathBuf>,
    /// The environment map to be used as a skybox and IBL
    envmap: Option<std::path::PathBuf>,
    /// Write logs to file
    file_log: bool,
}

struct CameraController {
    mouse_sensitivity: f32,
    pitch: f32,
    yaw: f32,
    movement_speed: f32,
}

impl CameraController {
    #[rustfmt::skip]
    const CONTROL_KEYS: &'static [(KeyCode, glam::Vec3)] = &[
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

    pub fn update_look(&mut self, delta: Vec2, transform: &mut Transform) {
        self.pitch -= delta.x * self.mouse_sensitivity;
        self.yaw = f32::clamp(self.yaw + delta.y * self.mouse_sensitivity, -PI / 2.0, PI / 2.0);

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
    },
}

impl Projection {
    #[inline]
    #[rustfmt::skip]
    pub fn compute_matrix(self, aspect_ratio: f32) -> glam::Mat4 {
        match self {
            Projection::Perspective {fov, near_clip } => glam::Mat4::perspective_infinite_reverse_rh(fov, aspect_ratio, near_clip),
            Projection::Orthographic { half_width, near_clip, far_clip } => {
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
    pub aspect_ratio: f32,
}

impl Camera {
    #[inline]
    pub fn compute_matrix(&self) -> Mat4 {
        let proj = self.projection.compute_matrix(self.aspect_ratio);
        let view = self.transform.compute_matrix().inverse();

        proj * view
    }

    pub fn compute_projection_matrix(&self) -> Mat4 {
        self.projection.compute_matrix(self.aspect_ratio)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub present_mode: vk::PresentModeKHR,
    pub msaa: graphics::MultisampleCount,
    pub shadow_settings: ShadowSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            present_mode: vk::PresentModeKHR::IMMEDIATE,
            msaa: Default::default(),
            shadow_settings: Default::default()
        }
    }
}

impl Settings {
    pub fn edit(&mut self, device: &graphics::Device, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let present_mode_display = |p: vk::PresentModeKHR| match p {
                vk::PresentModeKHR::FIFO         => "fifo",
                vk::PresentModeKHR::FIFO_RELAXED => "fifo_relaxed",
                vk::PresentModeKHR::IMMEDIATE    => "immediate",
                vk::PresentModeKHR::MAILBOX      => "mailbox",
                _ => unimplemented!()
            };

            ui.label("present_mode");
            egui::ComboBox::from_id_source("present_mode")
                .selected_text(format!("{}", present_mode_display(self.present_mode)))
                .show_ui(ui, |ui| {
                    for present_mode in device.gpu.surface_info.present_modes.iter().copied() {
                        ui.selectable_value(&mut self.present_mode, present_mode, present_mode_display(present_mode));
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("msaa");
            egui::ComboBox::from_id_source("msaa")
                .selected_text(format!("{}", self.msaa))
                .show_ui(ui, |ui| {
                    for sample_count in device.gpu.supported_multisample_counts() {
                        ui.selectable_value(&mut self.msaa, sample_count, sample_count.to_string());
                    }
                });
        });

        ui.heading("shadow_settings");

        self.shadow_settings.edit(ui);
    }
}

struct App {
    gpu_assets: GpuAssetStore,
    scene: SceneData,

    main_color_image: graphics::Image,
    main_color_resolve_image: Option<graphics::Image>,
    main_depth_image: graphics::Image,
    main_depth_resolve_image: Option<graphics::Image>,

    forward_renderer: ForwardRenderer,
    shadow_renderer: ShadowRenderer,
    debug_renderer: DebugRenderer,
    settings: Settings,

    environment_map: Option<EnvironmentMap>,

    camera: Camera,
    camera_controller: CameraController,
    frozen_camera: Camera,
    sun_light: Camera,
    light_dir_controller: CameraController,

    render_mode: RenderMode,
    camera_exposure: f32,
    light_color: Vec3,
    light_intensitiy: f32,
    selected_cascade: usize,

    freeze_camera: bool,
    show_cascade_view_frustum: bool,
    show_cascade_light_frustum: bool,
    show_cascade_light_frustum_planes: bool,

    enable_frustum_culling: bool,
    enable_occlusion_culling: bool,

    show_frustum_planes: bool,
    show_bounding_boxes: bool,
    show_bounding_spheres: bool,
    show_screen_aabb: bool,

    show_depth_pyramid: bool,
    pyramid_display_far_depth: f32,
    depth_pyramid_level: u32,

    selected_entity_index: Option<usize>,

    open_scene_editor_open: bool,
    open_graph_debugger: bool,
    open_profiler: bool,
    open_camera_light_editor: bool,
    open_settings: bool,
    open_culling_debugger: bool,
}

impl App {
    pub const COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
    pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

    fn new(
        context: &mut graphics::Context,
        gltf_path: Option<std::path::PathBuf>,
        env_map_path: Option<std::path::PathBuf>
    ) -> Self {
        let mut gpu_assets = GpuAssetStore::new(context);
        let mut scene = SceneData::new(context);

        if let Some(gltf_path) = gltf_path {
            load_gltf(&gltf_path, context, &mut gpu_assets, &mut scene).unwrap();
        }

        // let dist_size = 4.0;
        // let light_data = scene::LightData {
        //     color: vec3(1.0, 1.0, 1.0),
        //     intensity: 2.0,
        //     position: Vec3::ZERO,
        //     size: 0.0,
        // };

        // let mut thread_rng = rand::thread_rng();
        // use rand::Rng;

        // for _ in 0..32 {
        //     let position = Vec3 {
        //         x: thread_rng.gen_range(-dist_size..dist_size),
        //         y: thread_rng.gen_range(-dist_size..dist_size),
        //         z: thread_rng.gen_range(-dist_size..dist_size),
        //     };
        //     scene.add_light(scene::LightData { position, ..light_data });
        // }


        // use rand::Rng;
        // let mut rng = rand::thread_rng();

        // let prefab = scene.entities.pop().unwrap();
        // let pos_range = 0.0..=64.0;
        // let rot_range = 0.0..=2.0 * PI;
        
        // for _ in 0..2048 * 8 {
        //     let mut entity = prefab.clone();
            
        //     entity.transform.position = Vec3::from_array(std::array::from_fn(|_| rng.gen_range(pos_range.clone())));
        //     entity.transform.orientation = Quat::from_euler(
        //         glam::EulerRot::YXZ,
        //         rng.gen_range(rot_range.clone()),
        //         rng.gen_range(rot_range.clone()),
        //         rng.gen_range(rot_range.clone()),
        //     );


        //     scene.add_entity(entity);
        // }
        
        scene.update_instances(context);
        scene.update_submeshes(context, &gpu_assets);
        scene.update_lights(context);
        let environment_map = if let Some(env_map_path) = env_map_path {
            let equirectangular_environment_image = {
                let (image_binary, image_format) =
                    gltf_loader::load_image_data(&env_map_path).unwrap();
                let image = gltf_loader::load_image(
                    context,
                    "environment_map".into(),
                    &image_binary,
                    image_format,
                    true,
                    true,
                    graphics::SamplerKind::LinearRepeat
                );

                image
            };

            Some(EnvironmentMap::new(context,
                "environment_map".into(),
                1024,
                &equirectangular_environment_image,
            ))
        } else {
            None
        };

        let screen_extent = context.swapchain.extent();

        let main_color_image = context.create_image("main_color_image", &graphics::ImageDesc {
            ty: graphics::ImageType::Single2D,
            format: Self::COLOR_FORMAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels: 1,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: graphics::ImageSubresourceViewDesc::default(),
            ..Default::default()
        });

        let main_depth_image = context.create_image("main_depth_target", &graphics::ImageDesc {
            ty: graphics::ImageType::Single2D,
            format: Self::DEPTH_FORMAT,
            dimensions: [screen_extent.width, screen_extent.height, 1],
            mip_levels: 1,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::DEPTH,
            subresource_desc: graphics::ImageSubresourceViewDesc::default(),
            ..Default::default()
        });

        let aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        let camera = Camera {
            transform: Transform {
                position: vec3(0.0, 2.0, 0.0),
                ..Default::default()
            },
            projection: Projection::Perspective {
                fov: 90f32.to_radians(),
                near_clip: 0.01,
            },
            aspect_ratio,
        };

        let settings = Settings::default();

        context.swapchain.set_present_mode(settings.present_mode);

        Self {
            gpu_assets,
            scene,

            main_color_image,
            main_color_resolve_image: None,
            main_depth_image,
            main_depth_resolve_image: None,

            forward_renderer: ForwardRenderer::new(context),
            shadow_renderer: ShadowRenderer::new(context, settings.shadow_settings),
            debug_renderer: DebugRenderer::new(context),
            settings,

            environment_map,

            camera,
            camera_controller: CameraController::new(1.0, 0.003),
            frozen_camera: camera,
            sun_light: Camera {
                transform: Transform::default(),
                projection: Projection::Orthographic {
                    half_width: 20.0,
                    near_clip: -20.0,
                    far_clip: 20.0,
                },
                aspect_ratio: 1.0,
            },
            light_dir_controller: CameraController::new(1.0, 0.0003),

            render_mode: RenderMode::Shaded,
            camera_exposure: 1.0,
            light_color: Vec3::splat(1.0),
            light_intensitiy: 10.0,
            selected_cascade: 0,

            freeze_camera: false,
            show_cascade_view_frustum: false,
            show_cascade_light_frustum: false,
            show_cascade_light_frustum_planes: false,

            enable_frustum_culling: true,
            enable_occlusion_culling: true,

            show_frustum_planes: false,
            show_bounding_boxes: false,
            show_bounding_spheres: false,
            show_screen_aabb: false,
            
            show_depth_pyramid: false,
            pyramid_display_far_depth: 0.01,
            depth_pyramid_level: 0,

            selected_entity_index: None,
                
            open_scene_editor_open: false,
            open_graph_debugger: false,
            open_profiler: false,
            open_camera_light_editor: false,
            open_settings: false,
            open_culling_debugger: false
        }
    }

    fn update(
        &mut self,
        input: &Input,
        time: &Time,
        egui_ctx: &egui::Context,
        context: &mut graphics::Context,
        control_flow: &mut ControlFlow,
    ) {
        puffin::profile_function!();

        let delta_time = time.delta().as_secs_f32();

        if input.mouse_held(MouseButton::Right) {
            self.camera_controller.update_look(input.mouse_delta(), &mut self.camera.transform);
        }
        self.camera_controller.update_movement(input, delta_time, &mut self.camera.transform);

        if input.mouse_held(MouseButton::Left) {
            let mut delta = input.mouse_delta();

            if input.key_held(KeyCode::LShift) {
                delta *= 8.0;
            }

            if input.key_held(KeyCode::LControl) {
                delta /= 8.0;
            }

            self.light_dir_controller.update_look(delta, &mut self.sun_light.transform);
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
            self.open_settings = !self.open_settings;
        }
        if input.key_pressed(KeyCode::F6) {
            self.open_culling_debugger = !self.open_culling_debugger;
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
            egui::Window::new("scene")
                .open(&mut self.open_scene_editor_open)
                .default_width(280.0)
                .default_height(400.0)
                .vscroll(false)
                .resizable(true)
                .show(egui_ctx, |ui| {
                    egui::TopBottomPanel::top("scene_enttites")
                        .resizable(true)
                        .frame(egui::Frame::none().inner_margin(egui::Margin {
                            left: 0.0,
                            right: 0.0,
                            top: 2.0,
                            bottom: 12.0,
                        }))
                        .default_height(180.0)
                        .show_inside(ui, |ui| {
                            ui.heading("Entities");
                            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                                for (entity_index, entity) in self.scene.entities.iter_mut().enumerate() {
                                    let header = entity.name.as_ref().map_or(format!("entity_{entity_index}"), |name| {
                                        format!("entity_{entity_index} ({name})")
                                    });
                                    let is_entity_selected = Some(entity_index) == self.selected_entity_index;
                                    if ui.selectable_label(is_entity_selected, &header).clicked() {
                                        self.selected_entity_index = Some(entity_index);
                                    }
                                }
                            });
                        });
                    
                    ui.heading("Properties");
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        if let Some(entity_index) = self.selected_entity_index {
                            let entity = &mut self.scene.entities[entity_index];
                            
                            ui.heading("Transform");
                            drag_vec3(ui, "position", &mut entity.transform.position, 0.001);
                            let (euler_x, euler_y, euler_z) = entity.transform.orientation.to_euler(glam::EulerRot::YXZ);
                            let mut euler = vec3(euler_x, euler_y, euler_z);
                            drag_vec3(ui, "orientation", &mut euler, 0.05);
                            entity.transform.orientation = Quat::from_euler(glam::EulerRot::YXZ, euler.x, euler.y, euler.z);
                            drag_vec3(ui, "scale", &mut entity.transform.scale, 0.001);
                        } else {
                            ui.label("no entity selected");
                        }
                    });

                });
        }

        if self.open_graph_debugger {
            self.open_graph_debugger = context.graph_debugger(egui_ctx);
        }

        if self.open_profiler {
            self.open_profiler = puffin_egui::profiler_window(egui_ctx);
        }

        if self.open_settings {
            egui::Window::new("settings").open(&mut self.open_settings).show(egui_ctx, |ui| {
                self.settings.edit(&context.device, ui);
            });
            self.shadow_renderer.update_settings(&self.settings.shadow_settings);
        }

        if self.open_culling_debugger {
            egui::Window::new("culling debuger").open(&mut self.open_culling_debugger).show(egui_ctx, |ui| {
                ui.checkbox(&mut self.enable_frustum_culling, "enable frustum culling");
                ui.checkbox(&mut self.enable_occlusion_culling, "enable occlusion culling");
                ui.checkbox(&mut self.show_frustum_planes, "show frustum planes");
                ui.checkbox(&mut self.show_bounding_boxes, "show bounding boxes");
                ui.checkbox(&mut self.show_bounding_spheres, "show bounding spheres");
                ui.checkbox(&mut self.show_screen_aabb, "show screen aabb");
                ui.checkbox(&mut self.show_depth_pyramid, "show depth pyramid");
                if self.show_depth_pyramid {
                    ui.indent("depth_pyramid", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("depth pyramid max depth");
                            ui.add(egui::DragValue::new(&mut self.pyramid_display_far_depth).speed(0.005).clamp_range(0.0..=1.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("depth pyramid level");
                            ui.add(egui::Slider::new(
                                &mut self.depth_pyramid_level,
                                0..=self.forward_renderer.depth_pyramid.pyramid.mip_level(),
                            ));
                        });
                    });
                }
            });
        }

        context.swapchain.set_present_mode(self.settings.present_mode);

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
            self.render_mode = RenderMode::from(new_render_mode);
        }

        if input.close_requested() | input.key_pressed(KeyCode::Escape) {
            control_flow.set_exit();
        }
    }

    fn render(&mut self, context: &mut graphics::Context, egui_ctx: &egui::Context) {
        puffin::profile_function!();

        let assets = self.gpu_assets.import_to_graph(context);
        let scene = self.scene.import_to_graph(context);

        let screen_extent = context.swapchain_extent();
        self.camera.aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        if !self.freeze_camera {
            self.frozen_camera = self.camera;
        }

        let camera_view_projection_matrix = self.camera.compute_matrix();

        if self.freeze_camera {
            let Projection::Perspective { fov, near_clip } = self.frozen_camera.projection else { todo!() };
            let far_clip = self.settings.shadow_settings.max_shadow_distance;
            let frustum_corner = math::perspective_corners(fov, self.frozen_camera.aspect_ratio, near_clip, far_clip)
                .map(|corner| self.frozen_camera.transform.compute_matrix() * corner);
            self.debug_renderer.draw_frustum(&frustum_corner, vec4(1.0, 1.0, 1.0, 1.0));
        }

        self.main_color_image.recreate(&graphics::ImageDesc {
            dimensions: [screen_extent.width, screen_extent.height, 1],
            samples: self.settings.msaa,
            ..self.main_color_image.desc
        });
        self.main_depth_image.recreate(&graphics::ImageDesc {
            dimensions: [screen_extent.width, screen_extent.height, 1],
            samples: self.settings.msaa,
            ..self.main_depth_image.desc
        });

        if self.settings.msaa != graphics::MultisampleCount::None {
            if let Some(main_color_resolve_image) = self.main_color_resolve_image.as_mut() {
                main_color_resolve_image.recreate(&graphics::ImageDesc {
                    dimensions: [screen_extent.width, screen_extent.height, 1],
                    ..main_color_resolve_image.desc
                });
            } else {
                let image = context.create_image("main_color_resolve_image", &graphics::ImageDesc {
                    samples: graphics::MultisampleCount::None,
                    ..self.main_color_image.desc
                });
                self.main_color_resolve_image = Some(image);
            }

            if let Some(main_depth_resolve_image) = self.main_depth_resolve_image.as_mut() {
                main_depth_resolve_image.recreate(&graphics::ImageDesc {
                    dimensions: [screen_extent.width, screen_extent.height, 1],
                    ..main_depth_resolve_image.desc
                });
            } else {
                let image = context.create_image("main_depth_resolve_image", &graphics::ImageDesc {
                    samples: graphics::MultisampleCount::None,
                    ..self.main_depth_image.desc
                });
                self.main_depth_resolve_image = Some(image);
            }
        } else {
            self.main_color_resolve_image.take();
            self.main_depth_resolve_image.take();
        }

        let color_target = context.import(&self.main_color_image);
        let color_resolve_target = self.main_color_resolve_image.as_ref().map(|i| context.import(i));
        let depth_target = context.import(&self.main_depth_image);
        let depth_resolve = self.main_depth_resolve_image.as_ref().map(|i| context.import(i));

        let directional_light = self.shadow_renderer.render_directional_light(
            context,
            "sun".into(),
            self.sun_light.transform.orientation,
            self.light_color,
            self.light_intensitiy,
            &self.frozen_camera,
            assets,
            scene,
            self.selected_cascade,
            self.show_cascade_view_frustum,
            self.show_cascade_light_frustum,
            self.show_cascade_light_frustum_planes,
            &mut self.debug_renderer,
        );

        self.forward_renderer.render(
            context,
            &self.settings,

            assets,
            scene,

            self.freeze_camera,
            self.enable_frustum_culling,
            self.enable_occlusion_culling,
            
            self.environment_map.as_ref(),
            directional_light,

            &TargetAttachments {
                color_target,
                color_resolve: None,
                depth_target,
                depth_resolve,
            },
            
            &self.camera,
            &self.frozen_camera,
            self.render_mode,
        );

        egui::Window::new("camera_and_lighting")
            .open(&mut self.open_camera_light_editor)
            .show(egui_ctx, |ui| {
                ui.checkbox(&mut self.freeze_camera, "freeze camera");
                ui.checkbox(&mut self.show_cascade_view_frustum, "show cascade view frustum");
                ui.checkbox(&mut self.show_cascade_light_frustum, "show cascade light frustum");
                ui.checkbox(&mut self.show_cascade_light_frustum_planes, "show cascade light frustum planes");

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

                ui.horizontal(|ui| {
                    ui.label("selected_cascade");
                    ui.add(egui::Slider::new(
                        &mut self.selected_cascade,
                        0..=MAX_SHADOW_CASCADE_COUNT - 1,
                    ));
                });

                ui.image(directional_light.shadow_maps[self.selected_cascade], egui::Vec2::new(250.0, 250.0),
                );
            });

        if self.show_bounding_boxes || self.show_bounding_spheres || self.show_screen_aabb {
            let view_matrix = self.frozen_camera.transform.compute_matrix().inverse();
            let projection_matrix = self.frozen_camera.compute_projection_matrix();
            let screen_to_world_matrix = self.frozen_camera.compute_matrix().inverse();

            for entity in self.scene.entities.iter() {
                if let Some(model) = entity.model {
                    let transform_matrix = entity.transform.compute_matrix();
                    for submesh in self.gpu_assets.models[model].submeshes.iter() {
                        if self.show_bounding_boxes {
                            let aabb = self.gpu_assets.mesh_infos[submesh.mesh_handle].aabb;
                            let corners = [
                                vec3a(0.0, 0.0, 0.0),
                                vec3a(1.0, 0.0, 0.0),
                                vec3a(1.0, 1.0, 0.0),
                                vec3a(0.0, 1.0, 0.0),
                                vec3a(0.0, 0.0, 1.0),
                                vec3a(1.0, 0.0, 1.0),
                                vec3a(1.0, 1.0, 1.0),
                                vec3a(0.0, 1.0, 1.0),
                            ]
                            .map(|s| transform_matrix * (aabb.min + ((aabb.max - aabb.min) * s)).extend(1.0));
                            self.debug_renderer.draw_frustum(&corners, vec4(1.0, 1.0, 1.0, 1.0));
                        }

                        if self.show_bounding_spheres || self.show_screen_aabb {
                            let bounding_sphere = self.gpu_assets.mesh_infos[submesh.mesh_handle].bounding_sphere;
                            let position_world = transform_matrix.transform_point3a(bounding_sphere.into());
                            let radius = bounding_sphere.w * entity.transform.scale.max_element();
    
                            let p00 = projection_matrix.col(0)[0];
                            let p11 = projection_matrix.col(1)[1];
    
                            let mut position_view = view_matrix.transform_point3a(position_world);
                            position_view.z = -position_view.z;
    
                            let z_near = 0.01;
    
                            if let Some(aabb) = math::project_sphere_clip_space(position_view.extend(radius), z_near, p00, p11) {
                                if self.show_bounding_spheres {
                                    self.debug_renderer.draw_sphere(position_world.into(), radius, vec4(0.0, 1.0, 0.0, 1.0));
                                }
                                
                                if self.show_screen_aabb {
                                    let depth = z_near / (position_view.z - radius);
        
                                    let corners = [
                                        vec2(aabb.x, aabb.y),
                                        vec2(aabb.x, aabb.w),
                                        vec2(aabb.z, aabb.w),
                                        vec2(aabb.z, aabb.y),
                                    ].map(|c| {
                                        let v = screen_to_world_matrix * vec4(c.x, c.y, depth, 1.0);
                                        v / v.w
                                    });
                                    self.debug_renderer.draw_quad(&corners, vec4(1.0, 1.0, 1.0, 1.0));
                                }

                            } else if self.show_bounding_spheres {
                                self.debug_renderer.draw_sphere(position_world.into(), radius, vec4(1.0, 0.0, 0.0, 1.0));
                            }
                        }
                    }
                }
            }
        }

        if self.show_frustum_planes {
            let planes = math::frustum_planes_from_matrix(&self.frozen_camera.compute_matrix());

            for plane in planes {
                self.debug_renderer.draw_plane(plane, 2.0, vec4(1.0, 1.0, 1.0, 1.0));
            }
        }

        if let Some(entity_index) = self.selected_entity_index {
            let entity = &self.scene.entities[entity_index];

            if let Some(model) = entity.model {
                self.debug_renderer.draw_model_wireframe(entity.transform.compute_matrix(), model, vec4(1.0, 0.0, 1.0, 1.0));
            }
        }

        self.debug_renderer.render(
            context,
            &self.settings,
            &self.gpu_assets,
            color_target,
            color_resolve_target,
            depth_target,
            camera_view_projection_matrix,
        );

        let depth_pyramid = self.forward_renderer.depth_pyramid.get_current(context);

        render_post_process(
            context,
            if let Some(resolved) = color_resolve_target { resolved } else { color_target },
            self.camera_exposure,
            (self.show_depth_pyramid).then_some((depth_pyramid, self.depth_pyramid_level, self.pyramid_display_far_depth)),
            self.render_mode,
        );
    }
}

fn main() {
    let args: Args = onlyargs::parse().expect("failed to parse arguments");
    utils::init_logger(args.file_log);
    puffin::set_scopes_on(true);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("orbit")
        .with_resizable(true)
        .build(&event_loop)
        .expect("failed to build window");

    let mut input = Input::new(&window);
    let mut time = Time::new();

    let mut context = graphics::Context::new(window);

    let egui_ctx = egui::Context::default();
    let mut egui_state = egui_winit::State::new(&event_loop);
    let mut egui_renderer = EguiRenderer::new(&mut context);

    let mut app = App::new(&mut context, args.scene, args.envmap);

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
        }

        if input.handle_event(&event) {
            puffin::GlobalProfiler::lock().new_frame();
            puffin::profile_scope!("update");

            let raw_input = egui_state.take_egui_input(&context.window());
            egui_ctx.begin_frame(raw_input);

            time.update_now();
            app.update(&input, &time, &egui_ctx, &mut context, control_flow);

            let window_size = context.window().inner_size();
            if window_size.width == 0 || window_size.height == 0 {
                return;
            }

            context.begin_frame();

            let swapchain_image = context.get_swapchain_image();

            app.render(&mut context, &egui_ctx);

            let full_output = egui_ctx.end_frame();
            egui_state.handle_platform_output(&context.window(), &egui_ctx, full_output.platform_output);

            let clipped_primitives = {
                puffin::profile_scope!("egui_tessellate");
                egui_ctx.tessellate(full_output.shapes)
            };
            egui_renderer.render(
                &mut context,
                &clipped_primitives,
                &full_output.textures_delta,
                swapchain_image,
            );

            context.end_frame();

            input.clear_frame()
        }
    })
}
