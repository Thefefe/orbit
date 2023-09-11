use glam::mat2;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4, Quat, Mat4};

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

pub fn lerp_element_wise(x: Vec4, y: Vec4, a: Vec4) -> Vec4 {
    x + ((y - x) * a)
}

pub fn frustum_split(near: f32, far: f32, lambda: f32, ratio: f32) -> f32 {
    let uniform = near + (far - near) * ratio;
    let log = near * (far / near).powf(ratio);
    
    log * lambda + (1.0 - lambda) * uniform
}

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

pub fn project_sphere_clip_space(sphere: Vec4, znear: f32, p00: f32, p11: f32) -> Option<Vec4> {
    use glam::Vec3Swizzles;

    let c = Vec3A::from(sphere);
    let r = sphere.w;

    if c.z < r + znear {
        return None
    };

    let cx:   Vec2 = -c.xz();
    let vx:   Vec2 = vec2(f32::sqrt(Vec2::dot(cx, cx) - r * r), r);
    let minx: Vec2 = mat2(vec2(vx.x, vx.y), vec2(-vx.y, vx.x)) * cx;
    let maxx: Vec2 = mat2(vec2(vx.x, -vx.y), vec2(vx.y, vx.x)) * cx;

    let cy:   Vec2 = -c.yz();
    let vy:   Vec2 = vec2(f32::sqrt(Vec2::dot(cy, cy) - r * r), r);
    let miny: Vec2 = mat2(vec2(vy.x, vy.y), vec2(-vy.y, vy.x)) * cy;
    let maxy: Vec2 = mat2(vec2(vy.x, -vy.y), vec2(vy.y, vy.x)) * cy;

    let aabb = vec4(minx.x / minx.y * p00, miny.x / miny.y * p11, maxx.x / maxx.y * p00, maxy.x / maxy.y * p11);
    // *aabb = aabb.xwzy() * vec4(0.5, -0.5, 0.5, -0.5) + Vec4::splat(0.5); // clip space -> uv space

    return Some(aabb);
}