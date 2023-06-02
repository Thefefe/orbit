use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode as KeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

fn main() {
    if let Err(err) = init_logger() {
        eprintln!("failed to initialize logger: {err}");
    };

    let event_loop = EventLoop::new();
    let _window = WindowBuilder::new().with_title("orbit").build(&event_loop).expect("failed to build window");
    
    event_loop.run(|event, _target, control_flow| match event {
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
        Event::MainEventsCleared => {}
        _ => {}
    })
}

fn init_logger() -> Result<(), fern::InitError> {
    log_panics::init();
    use fern::colors::*;
    use std::fs;

    let log_file = fs::OpenOptions::new().write(true).create(true).truncate(true).open("log.txt")?;

    let colors_level = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        .info(Color::Blue)
        .debug(Color::BrightBlack)
        .trace(Color::White);

    let color_line = format!("\x1B[{}m", Color::White.to_fg_str());

    #[rustfmt::skip]
    fern::Dispatch::new()
        .level(log::LevelFilter::Trace)
        .chain(
            fern::Dispatch::new()
                .level(log::LevelFilter::Debug)
                .format(move |out, message, record| {
                    out.finish(format_args!(
    "{color_line}[{color_level}{target}{color_line}][{color_level}{level}{color_line}]{color_level} {message}\x1B[0m",
                        color_level = format_args!("\x1B[{}m", colors_level.get_color(&record.level()).to_fg_str()),
                        target = record.target().split("::").next().unwrap_or(""),
                        level = record.level(),
                        message = message,
                    ));
                })
                .chain(std::io::stdout()),
        )
        .chain(
            fern::Dispatch::new()
                .format(move |out, message, record| {
                    out.finish(format_args!(
                        "[{target}][{level}] {message}",
                        target = record.target(),
                        level = record.level(),
                        message = message,
                    ));
                })
                .chain(log_file),
        )
        .apply()?;

    Ok(())
}
