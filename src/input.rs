use std::collections::HashSet;

use winit::event::{DeviceEvent, ElementState, Event, MouseButton, VirtualKeyCode as KeyCode, WindowEvent};

pub struct Input {
    close_requested: bool,

    resized: Option<[u32; 2]>,

    pressed_keys: HashSet<KeyCode>,
    released_keys: HashSet<KeyCode>,
    held_keys: HashSet<KeyCode>,

    pressed_mouse: HashSet<MouseButton>,
    released_mouse: HashSet<MouseButton>,
    held_mouse: HashSet<MouseButton>,
    mouse_delta: glam::Vec2,

    pixels_per_point: f32,
}

impl Input {
    pub fn new() -> Self {
        Self {
            close_requested: false,
            resized: None,
            pressed_keys: HashSet::new(),
            released_keys: HashSet::new(),
            held_keys: HashSet::new(),
            pressed_mouse: HashSet::new(),
            released_mouse: HashSet::new(),
            held_mouse: HashSet::new(),
            mouse_delta: glam::Vec2::ZERO,
            pixels_per_point: 1.0,
        }
    }

    pub fn handle_event(&mut self, event: &Event<()>) -> bool {
        match event {
            Event::WindowEvent { window_id: _, event } => match event {
                WindowEvent::CloseRequested => {
                    self.close_requested = true;
                }
                WindowEvent::Resized(new_size) => {
                    _ = self.resized.insert((*new_size).into());
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    new_inner_size,
                } => {
                    self.pixels_per_point = *scale_factor as f32;
                    _ = self.resized.insert((**new_inner_size).into());
                }
                WindowEvent::KeyboardInput {
                    device_id: _,
                    input,
                    is_synthetic: _,
                } => {
                    if let Some(key_code) = input.virtual_keycode {
                        match input.state {
                            ElementState::Pressed => {
                                self.pressed_keys.insert(key_code);
                                self.held_keys.insert(key_code);
                            }
                            ElementState::Released => {
                                self.released_keys.insert(key_code);
                                self.held_keys.remove(&key_code);
                            }
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => match state {
                    ElementState::Pressed => {
                        self.pressed_mouse.insert(*button);
                        self.held_mouse.insert(*button);
                    }
                    ElementState::Released => {
                        self.released_mouse.insert(*button);
                        self.held_mouse.remove(button);
                    }
                },
                WindowEvent::Focused(false) => {
                    self.released_keys.extend(self.pressed_keys.iter());
                    self.released_mouse.extend(self.pressed_mouse.iter());
                }
                _ => {}
            },
            Event::DeviceEvent { device_id: _, event } => match event {
                DeviceEvent::MouseMotion { delta: (x, y) } => {
                    self.mouse_delta += glam::Vec2::new(*x as f32, -*y as f32);
                }
                _ => {}
            },
            _ => {}
        };
        event == &Event::MainEventsCleared
    }

    pub fn clear_frame(&mut self) {
        self.pressed_keys.clear();
        self.released_keys.clear();
        self.pressed_mouse.clear();
        self.released_mouse.clear();
        self.mouse_delta = glam::Vec2::ZERO;
        self.resized = None;
    }

    pub fn resized(&self) -> Option<[u32; 2]> {
        self.resized
    }

    pub fn close_requested(&self) -> bool {
        self.close_requested
    }

    pub fn key_pressed(&self, key_code: KeyCode) -> bool {
        self.pressed_keys.contains(&key_code)
    }

    pub fn key_release(&self, key_code: KeyCode) -> bool {
        self.released_keys.contains(&key_code)
    }

    pub fn key_held(&self, key_code: KeyCode) -> bool {
        self.held_keys.contains(&key_code)
    }

    pub fn mouse_pressed(&self, key_code: MouseButton) -> bool {
        self.pressed_mouse.contains(&key_code)
    }

    pub fn mouse_release(&self, key_code: MouseButton) -> bool {
        self.released_mouse.contains(&key_code)
    }

    pub fn mouse_held(&self, key_code: MouseButton) -> bool {
        self.held_mouse.contains(&key_code)
    }

    pub fn mouse_delta(&self) -> glam::Vec2 {
        self.mouse_delta
    }

    pub fn pixels_per_point(&self) -> f32 {
        self.pixels_per_point
    }
}
