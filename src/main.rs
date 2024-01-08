#![allow(dead_code)]

use std::{f32::consts::PI, ops::Range};

use ash::vk;
use assets::{GpuAssets, MAX_MESH_LODS};
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
mod graphics;
mod input;
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
use scene::{EntityData, Light, LightParams, SceneData, Transform};

use passes::{
    cluster::{ClusterDebugSettings, ClusterSettings},
    debug_renderer::DebugRenderer,
    env_map_loader::EnvironmentMap,
    forward::{ForwardRenderer, RenderMode},
    shadow_renderer::{ShadowRenderer, ShadowSettings},
};

use crate::passes::{
    cluster::{compute_clusters, debug_cluster_volumes},
    forward::TargetAttachments,
    post_process::render_post_process,
};

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

    pub fn set_look(&mut self, transform: &Transform) {
        let (pitch, yaw, _) = transform.orientation.to_euler(glam::EulerRot::YXZ);
        self.pitch = pitch;
        self.yaw = f32::clamp(yaw, -PI / 2.0, PI / 2.0);
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

    pub fn z_near(&self) -> f32 {
        match self {
            Projection::Orthographic { near_clip, .. } | Projection::Perspective { near_clip, .. } => *near_clip,
        }
    }

    pub fn z_far(&self) -> f32 {
        match self {
            Projection::Orthographic { far_clip, .. } => *far_clip,
            Projection::Perspective { .. } => f32::INFINITY,
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

    pub fn compute_view_matrix(&self) -> Mat4 {
        self.transform.compute_matrix().inverse()
    }

    pub fn compute_projection_matrix(&self) -> Mat4 {
        self.projection.compute_matrix(self.aspect_ratio)
    }

    pub fn z_near(&self) -> f32 {
        self.projection.z_near()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub present_mode: vk::PresentModeKHR,
    pub msaa: graphics::MultisampleCount,
    pub shadow_settings: ShadowSettings,

    pub use_mesh_shading: bool,
    pub min_mesh_lod: usize,
    pub max_mesh_lod: usize,
    pub lod_base: f32,
    pub lod_step: f32,

    pub camera_debug_settings: CameraDebugSettings,
    pub cluster_settings: ClusterSettings,
    pub cluster_debug_settings: ClusterDebugSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            present_mode: vk::PresentModeKHR::IMMEDIATE,
            msaa: Default::default(),
            shadow_settings: Default::default(),

            use_mesh_shading: false,

            min_mesh_lod: 0,
            max_mesh_lod: 7,
            lod_base: 10.0,
            lod_step: 2.5,

            camera_debug_settings: Default::default(),
            cluster_settings: Default::default(),
            cluster_debug_settings: Default::default(),
        }
    }
}

impl Settings {
    pub fn lod_range(&self) -> Range<usize> {
        self.min_mesh_lod..self.max_mesh_lod + 1
    }

    pub fn edit_general(&mut self, device: &graphics::Device, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let present_mode_display = |p: vk::PresentModeKHR| match p {
                vk::PresentModeKHR::FIFO => "fifo",
                vk::PresentModeKHR::FIFO_RELAXED => "fifo_relaxed",
                vk::PresentModeKHR::IMMEDIATE => "immediate",
                vk::PresentModeKHR::MAILBOX => "mailbox",
                _ => unimplemented!(),
            };

            ui.label("present mode");
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
            egui::ComboBox::from_id_source("msaa").selected_text(format!("{}", self.msaa)).show_ui(ui, |ui| {
                for sample_count in device.gpu.supported_multisample_counts() {
                    ui.selectable_value(&mut self.msaa, sample_count, sample_count.to_string());
                }
            });
        });

        ui.add_enabled(
            device.mesh_shader_fns.is_some(),
            egui::Checkbox::new(&mut self.use_mesh_shading, "use mesh shading"),
        );

        ui.horizontal(|ui| {
            ui.label("min mesh lod");
            ui.add(egui::Slider::new(&mut self.min_mesh_lod, 0..=self.max_mesh_lod.min(MAX_MESH_LODS - 1)));
        });

        ui.horizontal(|ui| {
            ui.label("max mesh lod");
            ui.add(egui::Slider::new(&mut self.max_mesh_lod, 0.max(self.min_mesh_lod)..=MAX_MESH_LODS - 1));
        });

        ui.horizontal(|ui| {
            ui.label("lod base");
            ui.add(egui::DragValue::new(&mut self.lod_base));
        });
        ui.horizontal(|ui| {
            ui.label("lod step");
            ui.add(egui::DragValue::new(&mut self.lod_step));
        });

        ui.heading("Shadow Settings");
        self.shadow_settings.edit(ui);
    }

    pub fn edit_cluster(&mut self, ui: &mut egui::Ui) {
        self.cluster_settings.edit(ui);
        ui.heading("Debug Settings");
        self.cluster_debug_settings.edit(&self.cluster_settings, ui);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraDebugSettings {
    pub exposure: f32,
    pub render_mode: RenderMode,
    pub freeze_camera: bool,
    pub show_bounding_boxes: bool,
    pub show_bounding_spheres: bool,

    pub frustum_culling: bool,
    pub occlusion_culling: bool,
    pub show_frustum_planes: bool,
    pub show_screen_space_aabbs: bool,

    pub show_depth_pyramid: bool,
    pub pyramid_display_far_depth: f32,
    pub depth_pyramid_level: u32,
}

impl Default for CameraDebugSettings {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            render_mode: RenderMode::Shaded,
            freeze_camera: false,
            show_bounding_boxes: false,
            show_bounding_spheres: false,

            frustum_culling: true,
            occlusion_culling: true,
            show_frustum_planes: false,
            show_screen_space_aabbs: false,

            show_depth_pyramid: false,
            pyramid_display_far_depth: 0.01,
            depth_pyramid_level: 0,
        }
    }
}

impl CameraDebugSettings {
    fn edit(&mut self, ui: &mut egui::Ui, pyramid_max_mip_level: u32) {
        ui.horizontal(|ui| {
            ui.label("camera exposure");
            ui.add(egui::DragValue::new(&mut self.exposure).clamp_range(0.1..=16.0).speed(0.1));
        });
        ui.checkbox(&mut self.freeze_camera, "freeze camera");
        ui.checkbox(&mut self.show_bounding_boxes, "show bounding boxes");
        ui.checkbox(&mut self.show_bounding_spheres, "show bounding spheres");

        ui.checkbox(&mut self.frustum_culling, "camera frustum culling");
        ui.checkbox(&mut self.occlusion_culling, "camera occlusion culling");
        ui.checkbox(&mut self.show_frustum_planes, "show camera frustum planes");
        ui.checkbox(&mut self.show_screen_space_aabbs, "show camera screen space aabbs");

        ui.checkbox(&mut self.show_depth_pyramid, "show depth pyramid");
        if self.show_depth_pyramid {
            ui.indent("depth_pyramid", |ui| {
                ui.horizontal(|ui| {
                    ui.label("depth pyramid max depth");
                    ui.add(
                        egui::DragValue::new(&mut self.pyramid_display_far_depth).speed(0.005).clamp_range(0.005..=1.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("depth pyramid level");
                    ui.add(egui::Slider::new(
                        &mut self.depth_pyramid_level,
                        0..=pyramid_max_mip_level,
                    ));
                });
            });
        }
    }
}

struct App {
    allocator_visualizer: gpu_allocator::vulkan::AllocatorVisualizer,

    gpu_assets: GpuAssets,
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
    sun_light_entity_index: usize,
    entity_dir_controller: CameraController,

    selected_entity_index: Option<usize>,
    // helpers for stable euler rotation
    selected_entity_euler_index: usize,
    selected_entity_euler_coords: Vec3,

    open_scene_editor_open: bool,
    open_graph_debugger: bool,
    open_profiler: bool,
    open_camera_debug_settings: bool,
    open_settings: bool,
    open_shadow_debug_settings: bool,
    open_cluster_settings: bool,
    open_allocator_visualizer: bool,
}

impl App {
    pub const COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
    pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

    fn new(
        context: &mut graphics::Context,
        gltf_path: Option<std::path::PathBuf>,
        env_map_path: Option<std::path::PathBuf>,
    ) -> Self {
        let mut gpu_assets = GpuAssets::new(context);
        let mut scene = SceneData::new(context);

        let mut settings = Settings::default();
        let mut shadow_renderer = ShadowRenderer::new(context, settings.shadow_settings);

        if context.device.mesh_shader_fns.is_some() {
            settings.use_mesh_shading = true;
        }

        let sun_light_entity_index = scene.add_entity(EntityData {
            name: Some("sun".into()),
            light: Some(Light {
                color: vec3(1.0, 1.0, 1.0),
                intensity: 8.0,
                params: LightParams::Directional { angular_size: 0.6 },
                cast_shadows: true,
                ..Default::default()
            }),
            ..Default::default()
        });

        let environment_map = if let Some(env_map_path) = env_map_path {
            let equirectangular_environment_image = {
                let (image_binary, image_format) = gltf_loader::load_image_data(&env_map_path).unwrap();
                let image = gltf_loader::load_image(
                    context,
                    "environment_map".into(),
                    &image_binary,
                    image_format,
                    true,
                    true,
                    graphics::SamplerKind::LinearRepeat,
                );

                image
            };

            Some(EnvironmentMap::new(
                context,
                "environment_map".into(),
                1024,
                &equirectangular_environment_image,
            ))
        } else {
            None
        };

        if let Some(environment_map) = environment_map.as_ref() {
            scene.add_entity(EntityData {
                name: Some("sky".into()),
                light: Some(Light {
                    color: vec3(1.0, 1.0, 1.0),
                    intensity: 0.8,
                    params: LightParams::Sky {
                        irradiance: environment_map.irradiance.clone(),
                        prefiltered: environment_map.prefiltered.clone(),
                    },
                    cast_shadows: false,
                    ..Default::default()
                }),
                ..Default::default()
            });
        }

        if let Some(gltf_path) = gltf_path {
            load_gltf(&gltf_path, context, &mut gpu_assets, &mut scene).unwrap();
        }

        // use rand::Rng;
        // let mut rng = rand::thread_rng();

        // let prefab = scene.entities.pop().unwrap();
        // let pos_range = -64.0..=64.0;
        // let rot_range = 0.0..=2.0 * PI;
        // for _ in 0..20_000 {
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

        // let horizontal_range = -32.0..=32.0;
        // let vertical_range = 0.0..=16.0;
        // for _ in 0..1000 {
        //     let position = Vec3 {
        //         x: rng.gen_range(horizontal_range.clone()),
        //         y: rng.gen_range(vertical_range.clone()),
        //         z: rng.gen_range(horizontal_range.clone()),
        //     };

        //     let color = egui::epaint::Hsva::new(rng.gen_range(0.0..=1.0), 1.0, 1.0, 1.0).to_rgb();
        //     let color = Vec3::from_array(color);
        //     let intensity = rng.gen_range(1.0..=6.0);

        //     scene.add_entity(EntityData {
        //         name: None,
        //         transform: Transform {
        //             position,
        //             ..Default::default()
        //         },
        //         light: Some(Light {
        //             color,
        //             intensity,
        //             params: LightParams::Point { inner_radius: 0.1 },
        //             ..Default::default()
        //         }),
        //         ..Default::default()
        //     });
        // }

        scene.update_scene(
            context,
            &mut shadow_renderer,
            &gpu_assets,
            settings.cluster_settings.luminance_cutoff,
        );

        let screen_extent = context.swapchain.extent();

        let main_color_image = context.create_image(
            "main_color_image",
            &graphics::ImageDesc {
                ty: graphics::ImageType::Single2D,
                format: Self::COLOR_FORMAT,
                dimensions: [screen_extent.width, screen_extent.height, 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                aspect: vk::ImageAspectFlags::COLOR,
                subresource_desc: graphics::ImageSubresourceViewDesc::default(),
                ..Default::default()
            },
        );

        let main_depth_image = context.create_image(
            "main_depth_target",
            &graphics::ImageDesc {
                ty: graphics::ImageType::Single2D,
                format: Self::DEPTH_FORMAT,
                dimensions: [screen_extent.width, screen_extent.height, 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                aspect: vk::ImageAspectFlags::DEPTH,
                subresource_desc: graphics::ImageSubresourceViewDesc::default(),
                ..Default::default()
            },
        );

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

        context.swapchain.set_present_mode(settings.present_mode);

        Self {
            allocator_visualizer: gpu_allocator::vulkan::AllocatorVisualizer::new(),

            gpu_assets,
            scene,

            main_color_image,
            main_color_resolve_image: None,
            main_depth_image,
            main_depth_resolve_image: None,

            forward_renderer: ForwardRenderer::new(context),
            shadow_renderer,
            debug_renderer: DebugRenderer::new(context),
            settings,

            environment_map,

            camera,
            camera_controller: CameraController::new(1.0, 0.003),
            frozen_camera: camera,
            sun_light_entity_index,
            entity_dir_controller: CameraController::new(1.0, 0.0003),

            selected_entity_index: None,
            selected_entity_euler_index: usize::MAX,
            selected_entity_euler_coords: Vec3::ZERO,

            open_scene_editor_open: false,
            open_graph_debugger: false,
            open_profiler: false,
            open_camera_debug_settings: false,
            open_settings: false,
            open_shadow_debug_settings: false,
            open_cluster_settings: false,
            open_allocator_visualizer: false,
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

        if let Some(entity_index) = self.selected_entity_index {
            const PIS_IN_180: f32 = 57.2957795130823208767981548141051703_f32;
            let entity = &mut self.scene.entities[entity_index];

            if input.mouse_held(MouseButton::Left) {
                let mut delta = input.mouse_delta();

                if input.key_held(KeyCode::LShift) {
                    delta *= 8.0;
                }

                if input.key_held(KeyCode::LControl) {
                    delta /= 8.0;
                }

                self.entity_dir_controller.update_look(delta, &mut entity.transform);
                self.selected_entity_euler_coords.y = self.entity_dir_controller.pitch * PIS_IN_180;
                self.selected_entity_euler_coords.x = self.entity_dir_controller.yaw * PIS_IN_180;
            }
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
            self.open_settings = !self.open_settings;
        }
        if input.key_pressed(KeyCode::F5) {
            self.open_camera_debug_settings = !self.open_camera_debug_settings;
        }
        if input.key_pressed(KeyCode::F6) {
            self.open_shadow_debug_settings = !self.open_shadow_debug_settings;
        }
        if input.key_pressed(KeyCode::F7) {
            self.open_cluster_settings = !self.open_cluster_settings;
        }
        if input.key_pressed(KeyCode::F8) {
            self.open_allocator_visualizer = !self.open_allocator_visualizer;
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
                                    let header =
                                        entity.name.as_ref().map_or(format!("entity_{entity_index}"), |name| {
                                            format!("entity_{entity_index} ({name})")
                                        });
                                    let is_entity_selected = Some(entity_index) == self.selected_entity_index;
                                    if ui.selectable_label(is_entity_selected, &header).clicked() {
                                        self.selected_entity_index = Some(entity_index);
                                        if self.selected_entity_euler_index != entity_index {
                                            const PIS_IN_180: f32 = 57.2957795130823208767981548141051703_f32;
                                            self.selected_entity_euler_index = entity_index;
                                            let (euler_x, euler_y, euler_z) =
                                                entity.transform.orientation.to_euler(glam::EulerRot::YXZ);

                                            self.selected_entity_euler_coords =
                                                vec3(euler_y, euler_x, euler_z) * PIS_IN_180;
                                            self.entity_dir_controller.set_look(&entity.transform);
                                        }
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

                            drag_vec3(ui, "orientation", &mut self.selected_entity_euler_coords, 0.5);
                            let euler_radian = self.selected_entity_euler_coords * (PI / 180.0);
                            entity.transform.orientation =
                                Quat::from_euler(glam::EulerRot::YXZ, euler_radian.y, euler_radian.x, euler_radian.z);
                            drag_vec3(ui, "scale", &mut entity.transform.scale, 0.001);

                            let mut has_light_component = entity.light.is_some();
                            ui.horizontal(|ui| {
                                ui.add(egui::Checkbox::without_text(&mut has_light_component));
                                ui.heading("Light");
                            });

                            if has_light_component != entity.light.is_some() {
                                entity.light = has_light_component.then(Light::default);
                            }

                            if let Some(light) = &mut entity.light {
                                light.edit(ui);
                            }
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
                self.settings.edit_general(&context.device, ui);
            });
            self.shadow_renderer.update_settings(&self.settings.shadow_settings);
        }

        if self.open_allocator_visualizer {
            let allocator_stuff = context.device.allocator_stuff.lock();
            let allocator = &allocator_stuff.allocator;
            self.allocator_visualizer.render_breakdown_window(
                egui_ctx,
                &allocator,
                &mut self.open_allocator_visualizer,
            );
            egui::Window::new("Allocator Memory Blocks")
                .open(&mut self.open_allocator_visualizer)
                .show(egui_ctx, |ui| {
                    self.allocator_visualizer.render_memory_block_ui(ui, &allocator)
                });
            self.allocator_visualizer.render_memory_block_visualization_windows(egui_ctx, &allocator);
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
            self.settings.camera_debug_settings.render_mode = RenderMode::from(new_render_mode);
        }

        if input.close_requested() | input.key_pressed(KeyCode::Escape) {
            control_flow.set_exit();
        }
    }

    fn render(&mut self, context: &mut graphics::Context, egui_ctx: &egui::Context) {
        puffin::profile_function!();

        self.scene.update_scene(
            context,
            &mut self.shadow_renderer,
            &self.gpu_assets,
            self.settings.cluster_settings.luminance_cutoff,
        );

        let assets = self.gpu_assets.import_to_graph(context);
        let scene = self.scene.import_to_graph(context);

        let screen_extent = context.swapchain_extent();

        self.camera.aspect_ratio = screen_extent.width as f32 / screen_extent.height as f32;

        if !self.settings.camera_debug_settings.freeze_camera {
            self.settings.cluster_settings.set_resolution([screen_extent.width, screen_extent.height]);
            self.frozen_camera = self.camera;
        }

        if self.settings.camera_debug_settings.freeze_camera {
            let Projection::Perspective { fov, near_clip } = self.frozen_camera.projection else {
                todo!()
            };
            let far_clip = self.settings.shadow_settings.max_shadow_distance;
            let frustum_corner = math::perspective_corners(fov, self.frozen_camera.aspect_ratio, near_clip, far_clip)
                .map(|corner| self.frozen_camera.transform.compute_matrix() * corner);
            self.debug_renderer.draw_cube_with_corners(&frustum_corner, vec4(1.0, 1.0, 1.0, 1.0));
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
                let image = context.create_image(
                    "main_color_resolve_image",
                    &graphics::ImageDesc {
                        samples: graphics::MultisampleCount::None,
                        ..self.main_color_image.desc
                    },
                );
                self.main_color_resolve_image = Some(image);
            }

            if let Some(main_depth_resolve_image) = self.main_depth_resolve_image.as_mut() {
                main_depth_resolve_image.recreate(&graphics::ImageDesc {
                    dimensions: [screen_extent.width, screen_extent.height, 1],
                    ..main_depth_resolve_image.desc
                });
            } else {
                let image = context.create_image(
                    "main_depth_resolve_image",
                    &graphics::ImageDesc {
                        samples: graphics::MultisampleCount::None,
                        ..self.main_depth_image.desc
                    },
                );
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

        let target_attachments = TargetAttachments {
            color_target,
            color_resolve: None,
            depth_target,
            depth_resolve,
        };

        egui::Window::new("camera debug settings")
            .open(&mut self.open_camera_debug_settings)
            .show(egui_ctx, |ui| {
                self.settings
                    .camera_debug_settings
                    .edit(ui, self.forward_renderer.depth_pyramid.pyramid.mip_level())
            });

        self.forward_renderer.render_depth_prepass(
            context,
            &self.settings,
            assets,
            scene,
            &target_attachments,
            &self.camera,
            &self.frozen_camera,
        );

        self.shadow_renderer.render_shadows(
            context,
            &self.settings,
            &self.frozen_camera,
            &self.gpu_assets,
            &self.scene,
            &mut self.debug_renderer,
        );

        let skybox = self.environment_map.as_ref().map(|e| context.import(&e.skybox));

        let selected_light = self
            .selected_entity_index
            .and_then(|i| self.scene.entities[i].light.as_ref().map(|l| l._light_index))
            .flatten();
        let selected_shadow = selected_light.and_then(|i| {
            let shadow_index = self.scene.light_data_cache[i].shadow_data_index;
            if shadow_index == u32::MAX {
                return None;
            };
            Some(shadow_index as usize - context.frame_index() * ShadowRenderer::MAX_SHADOW_COMMANDS)
        });
        self.shadow_renderer.debug_settings.selected_shadow = selected_shadow;

        let cluster_info = compute_clusters(
            context,
            &self.settings.cluster_settings,
            &self.camera,
            target_attachments.depth_target,
            scene,
        );

        self.forward_renderer.render(
            context,
            &self.settings,
            &self.gpu_assets,
            &self.scene,
            skybox,
            &self.shadow_renderer,
            &target_attachments,
            cluster_info,
            &self.camera,
            &self.frozen_camera,
            selected_light,
        );

        egui::Window::new("shadow debug settings")
            .open(&mut self.open_shadow_debug_settings)
            .show(egui_ctx, |ui| {
                self.shadow_renderer.edit_shadow_debug_settings(ui);
            });

        egui::Window::new("Cluster Settings")
            .open(&mut self.open_cluster_settings)
            .show(egui_ctx, |ui| self.settings.edit_cluster(ui));

        let show_bounding_boxes = self.settings.camera_debug_settings.show_bounding_boxes;
        let show_bounding_spheres = self.settings.camera_debug_settings.show_bounding_spheres;
        let show_screen_space_aabbs = self.settings.camera_debug_settings.show_screen_space_aabbs;

        if show_bounding_boxes || show_bounding_spheres || show_screen_space_aabbs {
            let view_matrix = self.frozen_camera.transform.compute_matrix().inverse();
            let projection_matrix = self.frozen_camera.compute_projection_matrix();
            let screen_to_world_matrix = self.frozen_camera.compute_matrix().inverse();

            for entity in self.scene.entities.iter() {
                let Some(mesh) = entity.mesh else {
                    continue;
                };

                let transform_matrix = entity.transform.compute_matrix();
                if show_bounding_boxes {
                    let aabb = self.gpu_assets.mesh_infos[mesh].aabb;
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
                    self.debug_renderer.draw_cube_with_corners(&corners, vec4(1.0, 1.0, 1.0, 1.0));
                }

                if show_bounding_spheres || show_screen_space_aabbs {
                    let bounding_sphere = self.gpu_assets.mesh_infos[mesh].bounding_sphere;
                    let position_world = transform_matrix.transform_point3a(bounding_sphere.into());
                    let radius = bounding_sphere.w * entity.transform.scale.max_element();

                    let p00 = projection_matrix.col(0)[0];
                    let p11 = projection_matrix.col(1)[1];

                    let mut position_view = view_matrix.transform_point3a(position_world);
                    position_view.z = -position_view.z;

                    let z_near = 0.01;

                    if let Some(aabb) = math::project_sphere_clip_space(position_view.extend(radius), z_near, p00, p11)
                    {
                        if show_bounding_spheres {
                            self.debug_renderer.draw_sphere(position_world.into(), radius, vec4(0.0, 1.0, 0.0, 1.0));
                        }

                        if show_screen_space_aabbs {
                            let depth = z_near / (position_view.z - radius);

                            let corners = [
                                vec2(aabb.x, aabb.y),
                                vec2(aabb.x, aabb.w),
                                vec2(aabb.z, aabb.w),
                                vec2(aabb.z, aabb.y),
                            ]
                            .map(|c| {
                                let v = screen_to_world_matrix * vec4(c.x, c.y, depth, 1.0);
                                v / v.w
                            });
                            self.debug_renderer.draw_quad(&corners, vec4(1.0, 1.0, 1.0, 1.0));
                        }
                    } else if show_bounding_spheres {
                        self.debug_renderer.draw_sphere(position_world.into(), radius, vec4(1.0, 0.0, 0.0, 1.0));
                    }
                }
            }
        }

        if self.settings.camera_debug_settings.show_frustum_planes {
            let planes = math::frustum_planes_from_matrix(&self.frozen_camera.compute_matrix());

            for plane in planes {
                self.debug_renderer.draw_plane(plane, 2.0, vec4(1.0, 1.0, 1.0, 1.0));
            }
        }

        if let Some(entity_index) = self.selected_entity_index {
            let entity = &self.scene.entities[entity_index];

            let pos = entity.transform.position;

            self.debug_renderer.draw_line(pos, pos + vec3(1.0, 0.0, 0.0), vec4(1.0, 0.0, 0.0, 1.0));
            self.debug_renderer.draw_line(pos, pos + vec3(0.0, 1.0, 0.0), vec4(0.0, 1.0, 0.0, 1.0));
            self.debug_renderer.draw_line(pos, pos + vec3(0.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0));

            if let Some(mesh) = entity.mesh {
                self.debug_renderer.draw_model_wireframe(
                    entity.transform.compute_matrix(),
                    mesh,
                    vec4(1.0, 0.0, 1.0, 1.0),
                );
            }

            if let Some(light) = &entity.light {
                if light.is_point() {
                    let range = light.outer_radius(self.settings.cluster_settings.luminance_cutoff);
                    self.debug_renderer.draw_sphere(pos, range, vec4(1.0, 1.0, 0.0, 1.0));
                }
            }
        }

        debug_cluster_volumes(
            &self.settings.cluster_settings,
            &self.settings.cluster_debug_settings,
            &self.frozen_camera,
            &mut self.debug_renderer,
        );

        self.debug_renderer.render(
            context,
            &self.settings,
            &self.gpu_assets,
            color_target,
            color_resolve_target,
            depth_target,
            &self.camera,
        );

        let show_depth_pyramid = self.settings.camera_debug_settings.show_depth_pyramid;
        let depth_pyramid_level = self.settings.camera_debug_settings.depth_pyramid_level;
        let pyramid_display_far_depth = self.settings.camera_debug_settings.pyramid_display_far_depth;
        let depth_pyramid = self.forward_renderer.depth_pyramid.get_current(context);

        render_post_process(
            context,
            if let Some(resolved) = color_resolve_target {
                resolved
            } else {
                color_target
            },
            self.settings.camera_debug_settings.exposure,
            (show_depth_pyramid).then_some((depth_pyramid, depth_pyramid_level, pyramid_display_far_depth)),
            self.settings.camera_debug_settings.render_mode,
        );
    }
}

fn main() {
    let args: Args = onlyargs::parse().expect("failed to parse arguments");
    utils::init_logger(args.file_log);
    puffin::set_scopes_on(true);
    rayon::ThreadPoolBuilder::new().build_global().expect("failed to create threadpool");

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
