use std::f32::consts::PI;

use glam::{mat2, Vec4Swizzles};
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

#[inline]
pub fn frustum_planes_from_matrix(matrix: &Mat4) -> [Vec4; 6] {
    let matrix_t = matrix.transpose();
    let mut planes = [matrix_t.col(3); 6];
    
    planes[0] += matrix_t.col(0);
    planes[1] -= matrix_t.col(0);
    planes[2] += matrix_t.col(1);
    planes[3] -= matrix_t.col(1);
    planes[4] += matrix_t.col(2);
    planes[5] -= matrix_t.col(2);

    planes
}

#[inline]
pub fn normalize_plane(plane: Vec4) -> Vec4 {
    plane / Vec3A::from(plane).length()
}

pub fn transform_plane(matrix: &Mat4, plane: Vec4) -> Vec4 {
    let plane_normal = Vec3A::from(plane);
    let mut o = (plane_normal * plane.w).extend(1.0);
    let mut n = plane_normal.extend(0.0);
    o = matrix.mul_vec4(o);
    n = matrix.inverse().transpose() * n;
    Vec3A::from(n).extend(Vec3A::dot(o.into(), n.into()))
}

#[inline]
pub fn frustum_corners_from_matrix(matrix: &Mat4) -> [Vec4; 8] {
    let inv_matrix = matrix.inverse();
    NDC_BOUNDS.map(|v| {
        let v = inv_matrix * v;
        v / v.w
    })
}

pub fn largest_scale_from_matrix(matrix: &Mat4) -> f32 {
    let x = Vec3A::from(matrix.x_axis);
    let y = Vec3A::from(matrix.y_axis);
    let z = Vec3A::from(matrix.z_axis);
    let largest_scale_sqr = x.dot(x).max(y.dot(y)).max(z.dot(z));
    largest_scale_sqr.sqrt()
}

pub fn transform_sphere(matrix: &Mat4, sphere: Vec4) -> Vec4 {
    let center = matrix.project_point3(sphere.xyz());
    let (scale, _, _) = matrix.to_scale_rotation_translation();
    let largest_scale = scale.max_element();
    // let radius = sphere.w * largest_scale_from_matrix(matrix);
    center.extend(largest_scale * sphere.w)
}

#[inline]
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

pub fn pack_f32_to_snorm_u8(f: f32) -> i8 {
    (f.clamp(-1.0, 1.0) * i8::MAX as f32) as i8
}

pub fn unpack_snorm_u8_to_f32(i: i8) -> f32 {
    f32::max(-1.0, i as f32 / i8::MAX as f32)
}

fn octahedron_wrap(v: Vec2) -> Vec2 {
    (Vec2::splat(1.0) - vec2(v.y, v.x).abs()) * Vec2::select(v.cmpge(Vec2::ZERO), Vec2::splat(1.0), Vec2::splat(-1.0))
}

pub fn octahedron_normal_encode(mut n: Vec3) -> Vec2 {
    n /= n.x.abs() + n.y.abs() + n.z.abs();
    let mut xy = vec2(n.x, n.y);
    xy = if n.z >= 0.0 { xy } else { octahedron_wrap(xy) };
    return xy;
}

pub fn octahedron_normal_decode(f: Vec2) -> Vec3 {
    let mut n = vec3(f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
    let t = f32::max(-n.z , 0.0);
    n += Vec3::select(n.cmpge(Vec3::ZERO), vec3(-t, -t, 0.0), vec3(t, t, 0.0));
    return n.normalize();
}

fn reference_orhonormal_vector(v: Vec3A) -> Vec3A {
    // if v.x.abs() > v.z.abs() {
    //     vec3a(-v.y, v.x, 0.0)
    // } else {
    //     vec3a(0.0, -v.z, v.y)
    // }.normalize()
    v.any_orthonormal_vector()
}

// https://advances.realtimerendering.com/s2020/RenderingDoomEternal.pdf page 35
pub fn rotational_tangent_encode(normal: Vec3A, tangent: Vec3A) -> f32 {
    let normal = normal.normalize();
    let tangent = tangent.normalize();
    let reference_tangent = reference_orhonormal_vector(normal).normalize();

    let alpha = f32::atan2(
        tangent.cross(reference_tangent).dot(normal),
        tangent.dot(reference_tangent)
    );

    // let tangent0 = reference_tangent * alpha.cos() - normal.cross(reference_tangent) * alpha.sin();
    // assert_kinda_eq!(tangent, tangent0);

    alpha
}

pub fn rotational_tangent_decode(normal: Vec3A, alpha: f32) -> Vec3A {
    let reference_tangent = reference_orhonormal_vector(normal);

    let tangent = reference_tangent * alpha.cos() + reference_tangent.cross(normal) * alpha.sin();
    tangent.into()
}

pub fn pack_normal_tangent_bitangent(normal: Vec3, tangent: Vec4) -> [i8; 4] {
    let octahedron_normal = octahedron_normal_encode(normal);
    let tangent_alpha = rotational_tangent_encode(normal.into(), tangent.truncate().into()) / PI;

    [octahedron_normal.x, octahedron_normal.y, tangent_alpha, tangent.w].map(pack_f32_to_snorm_u8)
}

pub fn unpack_normal_tangent_bitangent(packed: [i8; 4]) -> (Vec3, Vec4) {
    let [oct_norm_x, oct_norm_y, tangent_alpha, bitangent] = packed.map(unpack_snorm_u8_to_f32);
    let normal = octahedron_normal_decode(vec2(oct_norm_x, oct_norm_y));
    let tangent = rotational_tangent_decode(normal.into(), tangent_alpha);

    (normal, tangent.extend(bitangent))
}

#[cfg(test)]
mod tests {
    use glam::*;
    use crate::math::*;

    use super::octahedron_normal_decode;

    macro_rules! assert_kinda_eq {
        ($left:expr, $right:expr $(,)?) => {
            match (&$left, &$right) {
                (left_val, right_val) => {
                    if !left_val.abs_diff_eq(*right_val, 1e-6) {
                        panic!("assert_kinda_eq failed: {:?}, {:?}", left_val, right_val);
                    }
                }
            }
        };
    }

    #[test]
    fn octahedron_normal() {
        fn encode_decode(v: Vec3) -> Vec3 {
            octahedron_normal_decode(octahedron_normal_encode(v))
        }
        
        let normals = [
            vec3( 1.0,  0.0,  0.0),
            vec3( 0.0,  1.0,  0.0),
            vec3( 0.0,  0.0,  1.0),
            vec3(-1.0,  0.0,  0.0),
            vec3( 0.0, -1.0,  0.0),
            vec3( 0.0,  0.0, -1.0),
            vec3(-1.0,  0.0,  0.0),
            vec3( 0.0, -1.0,  0.0),
            vec3( 0.0,  0.0, -1.0),
            vec3( 1.0,  1.0,  0.0).normalize(),
            vec3( 0.0,  1.0,  1.0).normalize(),
            vec3( 1.0,  0.0,  1.0).normalize(),
            vec3(-1.0,  1.0,  0.0).normalize(),
            vec3( 0.0, -1.0,  1.0).normalize(),
            vec3( 1.0,  0.0, -1.0).normalize(),
            vec3( 321.0,  12.0,  543.0).normalize(),
            vec3( 432.0,  23.0,  43.0).normalize(),
            vec3( -431.0,  -20.0,  21.0).normalize(),
            vec3(-1.0,  21.0,  -30.0).normalize(),
            vec3( -30.0, -1.0,  1.0).normalize(),
            vec3( 1.0,  10.0, -1.0).normalize(),
        ];

        for normal in normals {
            assert_kinda_eq!(normal, encode_decode(normal));
        }
    }

    #[test]
    fn rotational_tangent() {
        fn encode_decode(normal: Vec3, tangent: Vec3) -> Vec3 {
            rotational_tangent_decode(normal.into(), rotational_tangent_encode(normal.into(), tangent.into())).into()
        }

        let normals = [
            vec3( 1.0,  0.0,  0.0),
            vec3( 0.0,  1.0,  0.0),
            vec3( 0.0,  0.0,  1.0),
            vec3(-1.0,  0.0,  0.0),
            vec3( 0.0, -1.0,  0.0),
            vec3( 0.0,  0.0, -1.0),
            vec3(-1.0,  0.0,  0.0),
            vec3( 0.0, -1.0,  0.0),
            vec3( 0.0,  0.0, -1.0),
            vec3( 1.0,  1.0,  0.0).normalize(),
            vec3( 0.0,  1.0,  1.0).normalize(),
            vec3( 1.0,  0.0,  1.0).normalize(),
            vec3(-1.0,  1.0,  0.0).normalize(),
            vec3( 0.0, -1.0,  1.0).normalize(),
            vec3( 1.0,  0.0, -1.0).normalize(),
            vec3( 321.0,  12.0,  543.0).normalize(),
            vec3( 432.0,  23.0,  43.0).normalize(),
            vec3( -431.0,  -20.0,  21.0).normalize(),
            vec3(-1.0,  21.0,  -30.0).normalize(),
            vec3( -30.0, -1.0,  1.0).normalize(),
            vec3( 1.0,  10.0, -1.0).normalize(),
        ];

        for normal in normals {
            let (tangent0, tangent1) = normal.any_orthonormal_pair();

            assert_kinda_eq!(tangent0, encode_decode(normal, tangent0));
            assert_kinda_eq!(tangent1, encode_decode(normal, tangent1));
        }
    }
}