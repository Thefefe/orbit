#![allow(dead_code)]

mod app;

mod assets;
mod camera;
mod graphics;
mod input;
mod scene;
mod time;

mod passes;

mod collections;
mod math;
mod utils;

mod egui_renderer;
mod gltf_loader;

use app::App;

use winit::{event_loop::EventLoop, window::WindowBuilder};

/// An experimental Vulkan 1.3 renderer
#[derive(Debug, onlyargs_derive::OnlyArgs)]
pub struct Args {
    /// The gltf scene to be loaded at startup
    pub scene_path: Option<std::path::PathBuf>,
    /// The environment map to be used as a skybox and IBL
    pub envmap_path: Option<std::path::PathBuf>,
    /// Write logs to file
    pub file_log: bool,
}

fn main() {
    let args: Args = onlyargs::parse().expect("failed to parse arguments");
    utils::init_logger(args.file_log);
    puffin::set_scopes_on(true);
    // rayon::ThreadPoolBuilder::new().build_global().expect("failed to create threadpool");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("orbit")
        .with_resizable(true)
        .build(&event_loop)
        .expect("failed to build window");

    let mut app = App::new(args, &event_loop, window);

    event_loop.run(move |event, _target, control_flow| app.handle_events(event, control_flow));
}
