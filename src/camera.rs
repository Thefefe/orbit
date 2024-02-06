use crate::{input::Input, scene::Transform};
use std::f32::consts::PI;

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};
use winit::event::VirtualKeyCode as KeyCode;

pub struct CameraController {
    pub mouse_sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub movement_speed: f32,
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
