use std::{f32::consts::PI, ops::Range};

use crate::{
    assets::{GpuAssets, MAX_MESH_LODS},
    camera::{Camera, CameraController, Projection},
    egui_renderer::EguiRenderer,
    gltf_loader::{self, load_gltf},
    graphics,
    input::Input,
    math,
    passes::{
        bloom::{compute_bloom, BloomSettings},
        cluster::{compute_clusters, debug_cluster_volumes, ClusterDebugSettings, ClusterSettings},
        debug_renderer::DebugRenderer,
        env_map_loader::EnvironmentMap,
        forward::{ForwardRenderer, RenderMode, TargetAttachments},
        post_process::render_post_process,
        shadow_renderer::{ShadowRenderer, ShadowSettings},
        ssao::{SsaoRenderer, SsaoSettings},
    },
    scene::{EntityData, Light, LightParams, SceneData, Transform},
    Args,
};

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};

use ash::vk;
use rand::SeedableRng;

use crate::time::Time;
use smallvec::SmallVec;
use winit::{
    event::{Event, MouseButton, VirtualKeyCode as KeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

#[cfg(windows)]
// winit on windows only gets the monitor GDI not the "friendly" name
mod monitor_name_util {
    use std::{collections::HashMap, ffi::OsString, os::windows::ffi::OsStringExt};

    use windows::Win32::{
        Devices::Display::{
            DisplayConfigGetDeviceInfo, GetDisplayConfigBufferSizes, QueryDisplayConfig,
            DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME,
            DISPLAYCONFIG_MODE_INFO, DISPLAYCONFIG_PATH_INFO, DISPLAYCONFIG_SOURCE_DEVICE_NAME,
            DISPLAYCONFIG_TARGET_DEVICE_NAME, QDC_ONLY_ACTIVE_PATHS, QDC_VIRTUAL_MODE_AWARE,
        },
        Foundation::WIN32_ERROR,
        Graphics::Gdi::{GetMonitorInfoW, HMONITOR, MONITORINFOEXW},
    };

    pub fn generate_gdi_to_monitor_name_lookup() -> HashMap<String, String> {
        let mut paths: Vec<DISPLAYCONFIG_PATH_INFO> = Vec::new();
        let mut modes: Vec<DISPLAYCONFIG_MODE_INFO> = Vec::new();
        let flags = QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_MODE_AWARE;

        let mut path_count: u32 = 0;
        let mut mode_count: u32 = 0;
        unsafe {
            GetDisplayConfigBufferSizes(flags, &mut path_count, &mut mode_count).unwrap();
        }

        paths.resize(path_count as usize, DISPLAYCONFIG_PATH_INFO::default());
        modes.resize(mode_count as usize, DISPLAYCONFIG_MODE_INFO::default());

        unsafe {
            QueryDisplayConfig(
                flags,
                &mut path_count,
                paths.as_mut_ptr(),
                &mut mode_count,
                modes.as_mut_ptr(),
                None,
            )
            .unwrap();
        }
        paths.truncate(path_count as usize);
        modes.truncate(mode_count as usize);

        paths
            .iter()
            .map(|path| {
                let mut target_name = DISPLAYCONFIG_TARGET_DEVICE_NAME::default();
                target_name.header.adapterId = path.targetInfo.adapterId;
                target_name.header.id = path.targetInfo.id;
                target_name.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
                target_name.header.size = std::mem::size_of_val(&target_name) as u32;
                WIN32_ERROR(unsafe { DisplayConfigGetDeviceInfo(&mut target_name.header) } as u32).ok().unwrap();

                let mut source_name = DISPLAYCONFIG_SOURCE_DEVICE_NAME::default();
                source_name.header.adapterId = path.targetInfo.adapterId;
                source_name.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
                source_name.header.size = std::mem::size_of_val(&source_name) as u32;
                WIN32_ERROR(unsafe { DisplayConfigGetDeviceInfo(&mut source_name.header) } as u32).ok().unwrap();

                let monitor_name = wide_to_string(&target_name.monitorFriendlyDeviceName);
                let gdi_name = wide_to_string(&source_name.viewGdiDeviceName);

                (gdi_name, monitor_name)
            })
            .collect()
    }

    fn wide_to_string(mut wide: &[u16]) -> String {
        if let Some(null) = wide.iter().position(|c| *c == 0) {
            wide = &wide[..null];
        }
        let os_string = OsString::from_wide(wide);
        os_string.to_string_lossy().into()
    }

    pub fn monitor_name_form_hmonitor(lookup: &mut HashMap<String, String>, handle: isize) -> Option<String> {
        let handle = HMONITOR(handle);
        let mut monitor_info = MONITORINFOEXW::default();
        monitor_info.monitorInfo.cbSize = std::mem::size_of_val(&monitor_info) as u32;
        if unsafe { GetMonitorInfoW(handle, &mut monitor_info.monitorInfo).as_bool() } {
            let gdi_name = wide_to_string(&monitor_info.szDevice);
            return lookup.remove(&gdi_name);
        }

        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    Windowed,
    FullscreenBorderless,
    FullscreenExclusive,
}

impl WindowMode {
    pub fn name(self) -> &'static str {
        match self {
            WindowMode::Windowed => "Windowed",
            WindowMode::FullscreenBorderless => "Borderless",
            WindowMode::FullscreenExclusive => "Fullscreen",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Monitor {
    pub handle: winit::monitor::MonitorHandle,
    pub name: String,
    pub resolution_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct AvailableDisplayModes {
    pub monitors: SmallVec<[Monitor; 4]>,
    pub resolutions: Vec<[u32; 2]>,
    pub present_modes: graphics::PresentModeFlags,
}

impl AvailableDisplayModes {
    pub fn extract_from_context(context: &graphics::Context) -> Self {
        let mut monitors = SmallVec::new();
        let mut resolutions = Vec::new();

        #[cfg(windows)]
        let mut name_lookup = monitor_name_util::generate_gdi_to_monitor_name_lookup();

        fn aspect_ratio(res: [u32; 2]) -> [u32; 2] {
            let gcd = num_integer::gcd(res[0], res[1]);
            res.map(|n| n / gcd)
        }

        for monitor in context.window.available_monitors() {
            let mut main_aspect_ratio: Option<[u32; 2]> = None;

            let resolutions_range_start = resolutions.len();
            for video_mode in monitor.video_modes() {
                let aspect_ratio = aspect_ratio(video_mode.size().into());
                if aspect_ratio == *main_aspect_ratio.get_or_insert(aspect_ratio) {
                    resolutions.push(video_mode.size().into());
                }
            }
            let resolution_range = resolutions_range_start..resolutions.len();

            #[cfg(not(windows))]
            let name = monitor.name().unwrap();

            #[cfg(windows)]
            let name = {
                use winit::platform::windows::MonitorHandleExtWindows;
                monitor_name_util::monitor_name_form_hmonitor(&mut name_lookup, monitor.hmonitor())
                    .unwrap_or_else(|| monitor.name().unwrap())
            };

            monitors.push(Monitor {
                handle: monitor,
                name,
                resolution_range,
            })
        }

        let present_modes = context.device.gpu.surface_info.present_modes;

        Self {
            monitors,
            resolutions,
            present_modes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisplaySettings {
    pub available_display_modes: AvailableDisplayModes,
    pub window_mode: WindowMode,
    pub selected_monitor: usize,
    pub selected_resolution: usize,
    pub present_mode: vk::PresentModeKHR,
}

impl DisplaySettings {
    pub fn new(context: &graphics::Context) -> Self {
        Self {
            available_display_modes: AvailableDisplayModes::extract_from_context(context),
            window_mode: WindowMode::Windowed,
            selected_monitor: 0,
            selected_resolution: 0,
            present_mode: vk::PresentModeKHR::IMMEDIATE,
        }
    }

    pub fn edit(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Window Mode");
            egui::ComboBox::from_id_source("window_mode")
                .selected_text(self.window_mode.name())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.window_mode,
                        WindowMode::FullscreenExclusive,
                        WindowMode::FullscreenExclusive.name(),
                    );
                    ui.selectable_value(
                        &mut self.window_mode,
                        WindowMode::FullscreenBorderless,
                        WindowMode::FullscreenBorderless.name(),
                    );
                    ui.selectable_value(&mut self.window_mode, WindowMode::Windowed, WindowMode::Windowed.name());
                });
        });

        ui.horizontal(|ui| {
            ui.set_enabled(self.window_mode != WindowMode::Windowed && self.available_display_modes.monitors.len() > 1);
            ui.label("Monitor");
            egui::ComboBox::from_id_source("monitors")
                .selected_text(self.available_display_modes.monitors[self.selected_monitor].name.as_str())
                .show_ui(ui, |ui| {
                    for (idx, monitor) in self.available_display_modes.monitors.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_monitor, idx, monitor.name.as_str());
                    }
                });
        });

        let resolutions = &self.available_display_modes.resolutions
            [self.available_display_modes.monitors[self.selected_monitor].resolution_range.clone()];

        ui.horizontal(|ui| {
            ui.set_enabled(self.window_mode != WindowMode::Windowed && resolutions.len() > 1);
            ui.label("Resolution(TODO)");

            let res_display = |index: usize| format!("{}x{}", resolutions[index][0], resolutions[index][1],);
            egui::ComboBox::from_id_source("resolutions")
                .selected_text(res_display(self.selected_resolution))
                .show_ui(ui, |ui| {
                    for idx in 0..resolutions.len() {
                        ui.selectable_value(&mut self.selected_resolution, idx, res_display(idx));
                    }
                });
        });

        ui.horizontal(|ui| {
            let present_mode_display = |p: vk::PresentModeKHR| match p {
                vk::PresentModeKHR::FIFO => "fifo",
                vk::PresentModeKHR::FIFO_RELAXED => "fifo_relaxed",
                vk::PresentModeKHR::IMMEDIATE => "immediate",
                vk::PresentModeKHR::MAILBOX => "mailbox",
                _ => unimplemented!(),
            };

            ui.label("present mode");
            egui::ComboBox::from_id_source("present_modes")
                .selected_text(format!("{}", present_mode_display(self.present_mode)))
                .show_ui(ui, |ui| {
                    for present_mode in self.available_display_modes.present_modes.iter_supported() {
                        ui.selectable_value(&mut self.present_mode, present_mode, present_mode_display(present_mode));
                    }
                });
        });
    }

    pub fn selected_monitor(&self) -> winit::monitor::MonitorHandle {
        self.available_display_modes.monitors[self.selected_monitor].handle.clone()
    }

    pub fn swapchain_fullscreen_mode(&self) -> graphics::SwapchainFullScreenMode {
        match self.window_mode {
            WindowMode::Windowed => graphics::SwapchainFullScreenMode::None,
            WindowMode::FullscreenBorderless => graphics::SwapchainFullScreenMode::Borderless(self.selected_monitor()),
            WindowMode::FullscreenExclusive => graphics::SwapchainFullScreenMode::Exclusive(self.selected_monitor()),
        }
    }

    pub fn fullscreen_mode(&self) -> Option<winit::window::Fullscreen> {
        match self.window_mode {
            WindowMode::Windowed => None,
            WindowMode::FullscreenBorderless => {
                Some(winit::window::Fullscreen::Borderless(Some(self.selected_monitor())))
            }

            // Using borderless for exclusive too, but set to exclusive with VK_EXT_full_screen_exclusive
            // not sure it's correct but seems to work better then to set it to exclusive here too.
            WindowMode::FullscreenExclusive => {
                Some(winit::window::Fullscreen::Borderless(Some(self.selected_monitor())))
            } // WindowMode::FullscreenExclusive => Some(winit::window::Fullscreen::Exclusive(self.selected_video_mode())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub display_settings: DisplaySettings,

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
    pub ssao_enabled: bool,
    pub ssao_settings: SsaoSettings,
    pub bloom_enabled: bool,
    pub bloom_settings: BloomSettings,
    pub camera_exposure: f32,
}

impl Settings {
    pub fn new(display_settings: DisplaySettings) -> Self {
        Self {
            display_settings,
            msaa: Default::default(),
            shadow_settings: Default::default(),

            use_mesh_shading: false,

            min_mesh_lod: 0,
            max_mesh_lod: 7,
            lod_base: 16.0,
            lod_step: 2.0,

            camera_debug_settings: Default::default(),
            cluster_settings: Default::default(),
            cluster_debug_settings: Default::default(),
            ssao_enabled: true,
            ssao_settings: Default::default(),
            bloom_enabled: true,
            bloom_settings: Default::default(),
            camera_exposure: 1.0,
        }
    }
}

impl Settings {
    pub fn lod_range(&self) -> Range<usize> {
        self.min_mesh_lod..self.max_mesh_lod + 1
    }

    pub fn edit_general(&mut self, device: &graphics::Device, ui: &mut egui::Ui) {
        ui.heading("Display Settings");
        self.display_settings.edit(ui);

        ui.horizontal(|ui| {
            ui.label("MSAA");
            egui::ComboBox::from_id_source("msaa").selected_text(format!("{}", self.msaa)).show_ui(ui, |ui| {
                for sample_count in device.gpu.supported_multisample_counts() {
                    ui.selectable_value(&mut self.msaa, sample_count, sample_count.to_string());
                }
            });
        });

        ui.checkbox(&mut self.ssao_enabled, "SSAO");
        ui.checkbox(&mut self.bloom_enabled, "Bloom");

        ui.collapsing("Mesh Settings", |ui| {
            ui.add_enabled(
                device.mesh_shader_fns.is_some(),
                egui::Checkbox::new(&mut self.use_mesh_shading, "use mesh shading"),
            );

            ui.horizontal(|ui| {
                ui.label("min mesh lod");
                ui.add(egui::Slider::new(
                    &mut self.min_mesh_lod,
                    0..=self.max_mesh_lod.min(MAX_MESH_LODS - 1),
                ));
            });

            ui.horizontal(|ui| {
                ui.label("max mesh lod");
                ui.add(egui::Slider::new(
                    &mut self.max_mesh_lod,
                    0.max(self.min_mesh_lod)..=MAX_MESH_LODS - 1,
                ));
            });

            ui.horizontal(|ui| {
                ui.label("lod base");
                ui.add(egui::DragValue::new(&mut self.lod_base));
            });
            ui.horizontal(|ui| {
                ui.label("lod step");
                ui.add(egui::DragValue::new(&mut self.lod_step));
            });
        });

        ui.collapsing("Shadow Settings", |ui| self.shadow_settings.edit(ui));

        ui.collapsing("Post Proccessing Settings", |ui| {
            ui.horizontal(|ui| {
                ui.label("camera exposure");
                ui.add(egui::DragValue::new(&mut self.camera_exposure).clamp_range(0.1..=16.0).speed(0.1));
            });

            ui.heading("SSAO Settings");
            self.ssao_settings.edit(ui);

            ui.heading("Bloom Settings");
            self.bloom_settings.edit(ui)
        });
    }

    pub fn edit_cluster(&mut self, ui: &mut egui::Ui) {
        self.cluster_settings.edit(ui);
        ui.heading("Debug Settings");
        self.cluster_debug_settings.edit(&self.cluster_settings, ui);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraDebugSettings {
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

pub struct App {
    context: graphics::Context,
    input: Input,
    time: Time,

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: EguiRenderer,

    allocator_visualizer: gpu_allocator::vulkan::AllocatorVisualizer,

    assets: GpuAssets,
    scene: SceneData,

    main_color_image: graphics::Image,
    main_color_resolve_image: Option<graphics::Image>,
    main_depth_image: graphics::Image,
    main_depth_resolve_image: Option<graphics::Image>,

    forward_renderer: ForwardRenderer,
    shadow_renderer: ShadowRenderer,
    ssao_renderer: SsaoRenderer,
    debug_renderer: DebugRenderer,
    settings: Settings,

    environment_map: Option<EnvironmentMap>,

    camera: Camera,
    camera_controller: CameraController,
    frozen_camera: Camera,
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

    pub fn new(
        args: Args,
        event_loop: &EventLoop<()>,
        window: winit::window::Window,
    ) -> Self {
        let input = Input::new(&window);
        let time = Time::new();

        let mut context = graphics::Context::new(window);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(&event_loop);
        let egui_renderer = EguiRenderer::new(&mut context);

        let mut gpu_assets = GpuAssets::new(&context);
        let mut scene = SceneData::new(&context);

        let display_settings = DisplaySettings::new(&context);
        let mut settings = Settings::new(display_settings);
        let mut shadow_renderer = ShadowRenderer::new(&context, settings.shadow_settings);

        if context.device.mesh_shader_fns.is_some() {
            settings.use_mesh_shading = true;
        }

        scene.add_entity(EntityData {
            name: Some("sun".into()),
            transform: Transform {
                orientation: Quat::from_rotation_arc(vec3(0.0, 0.0, 1.0), vec3(-1.0, 1.0, 1.0).normalize()),
                ..Default::default()
            },
            light: Some(Light {
                color: vec3(1.0, 1.0, 1.0),
                intensity: 8.0,
                params: LightParams::Directional { angular_size: 0.6 },
                cast_shadows: true,
                ..Default::default()
            }),
            ..Default::default()
        });

        let environment_map = if let Some(env_map_path) = args.envmap_path {
            let equirectangular_environment_image = {
                let (image_binary, image_format) = gltf_loader::load_image_data(&env_map_path).unwrap();
                let image = gltf_loader::load_image(
                    &context,
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
                &mut context,
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
                    intensity: 0.65,
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

        if let Some(gltf_path) = args.scene_path {
            load_gltf(&gltf_path, &context, &mut gpu_assets, &mut scene).unwrap();
        }

        #[allow(unused_imports)]
        use rand::Rng;
        let mut _rng = rand::rngs::StdRng::from_seed([69; 32]);

        let prefab = scene.entities.pop().unwrap();
        let pos_range = -48.0..=48.0;
        let rot_range = 0.0..=2.0 * PI;
        for _ in 0..20_000 {
            let mut entity = prefab.clone();

            entity.transform.position = Vec3::from_array(std::array::from_fn(|_| _rng.gen_range(pos_range.clone())));
            entity.transform.orientation = Quat::from_euler(
                glam::EulerRot::YXZ,
                _rng.gen_range(rot_range.clone()),
                _rng.gen_range(rot_range.clone()),
                _rng.gen_range(rot_range.clone()),
            );

            scene.add_entity(entity);
        }

        // let horizontal_range = -8.0..8.0;
        // let vertical_range = 0.0..=6.0;
        // for _ in 0..64 {
        //     let position = Vec3 {
        //         x: _rng.gen_range(horizontal_range.clone()),
        //         y: _rng.gen_range(vertical_range.clone()),
        //         z: _rng.gen_range(horizontal_range.clone()),
        //     };

        //     let color = egui::epaint::Hsva::new(_rng.gen_range(0.0..=1.0), 1.0, 1.0, 1.0).to_rgb();
        //     let color = Vec3::from_array(color);
        //     let intensity = _rng.gen_range(1.0..=6.0);

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
            &context,
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

        context.swapchain.set_present_mode(settings.display_settings.present_mode);

        let forward_renderer = ForwardRenderer::new(&mut context);
        let ssao_renderer = SsaoRenderer::new(&mut context);
        let debug_renderer = DebugRenderer::new(&mut context);

        Self {
            context,

            input,
            time,

            egui_ctx,
            egui_state,
            egui_renderer,

            allocator_visualizer: gpu_allocator::vulkan::AllocatorVisualizer::new(),

            assets: gpu_assets,
            scene,

            main_color_image,
            main_color_resolve_image: None,
            main_depth_image,
            main_depth_resolve_image: None,

            forward_renderer,
            shadow_renderer,
            ssao_renderer,
            debug_renderer,
            settings,

            environment_map,

            camera,
            camera_controller: CameraController::new(1.0, 0.003),
            frozen_camera: camera,
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

    fn update(&mut self, control_flow: &mut ControlFlow) {
        puffin::profile_function!();

        let time = &self.time;
        let input = &self.input;

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
                .show(&self.egui_ctx, |ui| {
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
            self.open_graph_debugger = self.context.graph_debugger(&self.egui_ctx);
        }

        if self.open_profiler {
            self.open_profiler = puffin_egui::profiler_window(&self.egui_ctx);
        }

        if self.open_settings {
            egui::Window::new("settings").open(&mut self.open_settings).show(&self.egui_ctx, |ui| {
                self.settings.edit_general(&self.context.device, ui);
            });
            self.shadow_renderer.update_settings(&self.settings.shadow_settings);
        }

        if self.open_allocator_visualizer {
            let allocator_stuff = self.context.device.allocator_stuff.lock();
            let allocator = &allocator_stuff.allocator;
            self.allocator_visualizer.render_breakdown_window(
                &self.egui_ctx,
                &allocator,
                &mut self.open_allocator_visualizer,
            );
            egui::Window::new("Allocator Memory Blocks")
                .open(&mut self.open_allocator_visualizer)
                .show(&self.egui_ctx, |ui| {
                    self.allocator_visualizer.render_memory_block_ui(ui, &allocator)
                });
            self.allocator_visualizer.render_memory_block_visualization_windows(&self.egui_ctx, &allocator);
        }

        self.context.window().set_fullscreen(self.settings.display_settings.fullscreen_mode());
        self.context.swapchain.set_present_mode(self.settings.display_settings.present_mode);
        self.context.swapchain.set_fullscreen(self.settings.display_settings.swapchain_fullscreen_mode());

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

        if input.key_pressed(KeyCode::Escape) {
            self.selected_entity_index = None;
        }

        if input.close_requested() {
            control_flow.set_exit();
        }
    }

    fn render(&mut self) {
        puffin::profile_function!();

        self.scene.update_scene(
            &self.context,
            &mut self.shadow_renderer,
            &self.assets,
            self.settings.cluster_settings.luminance_cutoff,
        );

        let assets = self.assets.import_to_graph(&mut self.context);
        let scene = self.scene.import_to_graph(&mut self.context);

        let screen_extent = self.context.swapchain_extent();

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
                let image = self.context.create_image(
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
                let image = self.context.create_image(
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

        let color_target = self.context.import(&self.main_color_image);
        let color_resolve = self.main_color_resolve_image.as_ref().map(|i| self.context.import(i));
        let depth_target = self.context.import(&self.main_depth_image);
        let depth_resolve = self.main_depth_resolve_image.as_ref().map(|i| self.context.import(i));

        let target_attachments = TargetAttachments {
            color_target,
            color_resolve,
            depth_target,
            depth_resolve,
        };

        egui::Window::new("camera debug settings").open(&mut self.open_camera_debug_settings).show(
            &self.egui_ctx,
            |ui| {
                self.settings
                    .camera_debug_settings
                    .edit(ui, self.forward_renderer.depth_pyramid.pyramid.mip_level())
            },
        );

        self.forward_renderer.render_depth_prepass(
            &mut self.context,
            &self.settings,
            assets,
            scene,
            &target_attachments,
            &self.camera,
            &self.frozen_camera,
        );

        let ssao_image = self.settings.ssao_enabled.then(|| {
            self.ssao_renderer.compute_ssao(
                &mut self.context,
                &self.settings.ssao_settings,
                target_attachments.non_msaa_depth_target(),
                &self.camera,
                [screen_extent.width, screen_extent.height],
            )
        });

        self.shadow_renderer.render_shadows(
            &mut self.context,
            &self.settings,
            &self.frozen_camera,
            &self.assets,
            &self.scene,
            &mut self.debug_renderer,
        );

        let skybox = self.environment_map.as_ref().map(|e| {
            self.context.import_with(
                "skybox",
                &e.skybox,
                graphics::GraphResourceImportDesc {
                    initial_access: graphics::AccessKind::AllGraphicsRead,
                    ..Default::default()
                },
            )
        });

        let selected_light = self
            .selected_entity_index
            .and_then(|i| self.scene.entities[i].light.as_ref().map(|l| l._light_index))
            .flatten();
        let selected_shadow = selected_light.and_then(|i| {
            let shadow_index = self.scene.light_data_cache[i].shadow_data_index;
            if shadow_index == u32::MAX {
                return None;
            };
            Some(shadow_index as usize - self.context.frame_index() * ShadowRenderer::MAX_SHADOW_COMMANDS)
        });
        self.shadow_renderer.debug_settings.selected_shadow = selected_shadow;

        let cluster_info = compute_clusters(
            &mut self.context,
            &self.settings.cluster_settings,
            &self.camera,
            target_attachments.depth_target,
            scene,
        );

        self.forward_renderer.render(
            &mut self.context,
            &self.settings,
            &self.assets,
            &self.scene,
            skybox,
            ssao_image,
            &self.shadow_renderer,
            &target_attachments,
            cluster_info,
            &self.camera,
            &self.frozen_camera,
            selected_light,
        );

        let bloom_image = self.settings.bloom_enabled.then(|| {
            compute_bloom(
                &mut self.context,
                &self.settings.bloom_settings,
                target_attachments.non_msaa_color_target(),
            )
        });

        egui::Window::new("shadow debug settings").open(&mut self.open_shadow_debug_settings).show(
            &self.egui_ctx,
            |ui| {
                self.shadow_renderer.edit_shadow_debug_settings(ui);
            },
        );

        egui::Window::new("Cluster Settings")
            .open(&mut self.open_cluster_settings)
            .show(&self.egui_ctx, |ui| self.settings.edit_cluster(ui));

        let show_bounding_boxes = self.settings.camera_debug_settings.show_bounding_boxes;
        let show_bounding_spheres = self.settings.camera_debug_settings.show_bounding_spheres;
        let show_screen_space_aabbs = self.settings.camera_debug_settings.show_screen_space_aabbs;

        let assets_shared = self.assets.shared_stuff.read();
        if show_bounding_boxes || show_bounding_spheres || show_screen_space_aabbs {
            let view_matrix = self.frozen_camera.transform.compute_matrix().inverse();
            let projection_matrix = self.frozen_camera.compute_projection_matrix();
            let screen_to_world_matrix = self.frozen_camera.compute_matrix().inverse();

            for entity in self.scene.entities.iter() {
                let Some(mesh) = entity.mesh else {
                    continue;
                };

                let aabb = assets_shared.mesh_infos[mesh].aabb;
                let bounding_sphere = assets_shared.mesh_infos[mesh].bounding_sphere;

                let transform_matrix = entity.transform.compute_matrix();

                if show_bounding_boxes {
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
        drop(assets_shared);

        if self.settings.camera_debug_settings.show_frustum_planes {
            let planes = math::frustum_planes_from_matrix(&self.frozen_camera.compute_matrix());

            for plane in planes {
                self.debug_renderer.draw_plane(plane, 2.0, vec4(1.0, 1.0, 1.0, 1.0));
            }
        }

        if let Some(entity_index) = self.selected_entity_index {
            let entity = &self.scene.entities[entity_index];

            let pos = entity.transform.position;

            // self.debug_renderer.draw_line(pos, pos + vec3(1.0, 0.0, 0.0), vec4(1.0, 0.0, 0.0, 1.0));
            // self.debug_renderer.draw_line(pos, pos + vec3(0.0, 1.0, 0.0), vec4(0.0, 1.0, 0.0, 1.0));
            // self.debug_renderer.draw_line(pos, pos + vec3(0.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0));

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
            &mut self.context,
            &self.settings,
            &self.assets,
            target_attachments,
            &self.camera,
        );

        let show_depth_pyramid = self.settings.camera_debug_settings.show_depth_pyramid;
        let depth_pyramid_level = self.settings.camera_debug_settings.depth_pyramid_level;
        let pyramid_display_far_depth = self.settings.camera_debug_settings.pyramid_display_far_depth;
        let depth_pyramid = self.forward_renderer.depth_pyramid.get_current(&mut self.context);

        render_post_process(
            &mut self.context,
            target_attachments.non_msaa_color_target(),
            bloom_image,
            &self.settings,
            (show_depth_pyramid).then_some((depth_pyramid, depth_pyramid_level, pyramid_display_far_depth)),
            self.settings.camera_debug_settings.render_mode,
        );
    }

    pub fn handle_events(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
        if let Event::WindowEvent { event, .. } = &event {
            if self.egui_state.on_event(&self.egui_ctx, &event).consumed {
                return;
            }
        };

        if let Event::WindowEvent {
            event: WindowEvent::Focused(focused),
            ..
        } = &event
        {
            if self.context.window().fullscreen().is_some() {
                self.context.window().set_minimized(!focused);
                self.context.swapchain.set_minimized(&self.context.device, !focused);
            }
        }

        if event == Event::LoopDestroyed {
            unsafe {
                self.context.device.raw.device_wait_idle().unwrap();
            }
        }

        if self.input.handle_event(&event) {
            puffin::GlobalProfiler::lock().new_frame();
            puffin::profile_scope!("update");

            let raw_input = self.egui_state.take_egui_input(&self.context.window());
            self.egui_ctx.begin_frame(raw_input);

            self.time.update_now();
            self.update(control_flow);

            let window_size = self.context.window().inner_size();
            if window_size.width == 0 || window_size.height == 0 {
                let full_output = self.egui_ctx.end_frame();
                self.egui_state.handle_platform_output(
                    &self.context.window(),
                    &self.egui_ctx,
                    full_output.platform_output,
                );
                return;
            }

            self.context
                .swapchain
                .set_minimized(&self.context.device, self.context.window().is_minimized().unwrap());
            self.context.begin_frame();

            let swapchain_image = self.context.get_swapchain_image();

            self.render();

            let full_output = self.egui_ctx.end_frame();
            self.egui_state
                .handle_platform_output(&self.context.window(), &self.egui_ctx, full_output.platform_output);

            let clipped_primitives = {
                puffin::profile_scope!("egui_tessellate");
                self.egui_ctx.tessellate(full_output.shapes)
            };
            self.egui_renderer.render(
                &mut self.context,
                &clipped_primitives,
                &full_output.textures_delta,
                swapchain_image,
            );

            self.context.end_frame();

            self.input.clear_frame()
        }
    }
}
