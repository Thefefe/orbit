#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4, Quat, Mat4};

pub fn frustum_split(near: f32, far: f32, lambda: f32, ratio: f32) -> f32 {
    let uniform = near + (far - near) * ratio;
    let log = near * (far / near).powf(ratio);
    
    log * lambda + (1.0 - lambda) * uniform
}

pub const NDC_BOUNDS: [Vec4; 8] = [
    vec4(-1.0, -1.0, 0.0,  1.0),
    vec4( 1.0, -1.0, 0.0,  1.0),
    vec4( 1.0,  1.0, 0.0,  1.0),
    vec4(-1.0,  1.0, 0.0,  1.0),
    
    vec4(-1.0, -1.0, 1.0,  1.0),
    vec4( 1.0, -1.0, 1.0,  1.0),
    vec4( 1.0,  1.0, 1.0,  1.0),
    vec4(-1.0,  1.0, 1.0,  1.0),
];

pub fn frustum_planes_from_matrix(matrix: &Mat4) -> [Vec4; 6] {
    let mut planes = [matrix.row(3); 6];
    
    planes[0] += matrix.row(0);
    planes[1] -= matrix.row(0);
    planes[2] += matrix.row(1);
    planes[3] -= matrix.row(1);
    planes[4] += matrix.row(2);
    planes[5] -= matrix.row(2);
    
    // normalize planes
    // for plane in planes.iter_mut() {
    //     *plane /= Vec3A::from(*plane).length();
    // }

    planes
}

pub fn frustum_corners_from_matrix(matrix: &Mat4) -> [Vec4; 8] {
    let inv_matrix = matrix.inverse();
    NDC_BOUNDS.map(|v| {
        let v = inv_matrix * v;
        v / v.w
    })
}

pub fn perspective_corners(fovy: f32, aspect_ratio: f32, near: f32, far: f32) -> [Vec4; 8] {
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