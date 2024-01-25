use ash::vk;
use glam::{vec2, vec3a, vec4, Mat4, Vec2, Vec3A, Vec3Swizzles, Vec4, Vec4Swizzles};
use gpu_allocator::MemoryLocation;

use crate::{
    graphics::{self, AccessKind, ComputePass, ShaderStage},
    math::{self, Aabb},
    scene::SceneGraphData,
    Camera,
};

use super::debug_renderer::DebugRenderer;

#[derive(Debug, Clone, Copy)]
pub struct ClusterSettings {
    pub px_size_power: u32,
    pub screen_resolution: [u32; 2],
    pub z_slice_count: u32,
    pub far_plane: f32,
    pub luminance_cutoff: f32,
}

impl Default for ClusterSettings {
    fn default() -> Self {
        Self {
            px_size_power: 3, // 2^3 = 8
            screen_resolution: [0; 2],
            z_slice_count: 32,
            far_plane: 200.0,
            luminance_cutoff: 0.25,
        }
    }
}

impl ClusterSettings {
    pub fn tile_px_size(&self) -> u32 {
        u32::pow(2, self.px_size_power)
    }

    pub fn set_resolution(&mut self, resolution: [u32; 2]) {
        self.screen_resolution = resolution;
    }

    pub fn tile_counts(&self) -> [usize; 2] {
        self.screen_resolution.map(|n| n.div_ceil(self.tile_px_size()) as usize)
    }

    pub fn linear_cluster_count(&self) -> usize {
        let tile_counts = self.tile_counts();
        tile_counts[0] * tile_counts[1] * self.z_slice_count as usize
    }

    pub fn linear_max_allocated_cluster_count(&self) -> usize {
        let tile_counts = self.tile_counts();
        tile_counts[0] * tile_counts[1] * MAX_ALLOCATED_DEPTH_SLICES.max(self.z_slice_count as usize)
    }

    pub fn cluster_counts(&self) -> [usize; 3] {
        let tile_counts = self.tile_counts();
        [tile_counts[0], tile_counts[1], self.z_slice_count as usize]
    }

    pub fn cluster_grid_info(&self, near: f32) -> (f32, f32) {
        let far = self.far_plane;
        let num_slices = self.z_slice_count as f32;
        let log_f_n = f32::log2(far / near);

        let z_scale = num_slices / log_f_n;
        let z_bias = -((num_slices * near.log2()) / log_f_n);

        (z_scale, z_bias)
    }

    pub fn edit(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("tile pixel size");
            ui.add(
                egui::Slider::new(&mut self.px_size_power, 3..=6)
                    .custom_formatter(|n, _| u32::pow(2, n as u32).to_string()),
            );
        });
        ui.horizontal(|ui| {
            ui.label("z slice count");
            ui.add(egui::DragValue::new(&mut self.z_slice_count).clamp_range(1..=32));
        });
        ui.horizontal(|ui| {
            ui.label("z far plane");
            ui.add(egui::DragValue::new(&mut self.far_plane).clamp_range(10.0..=1000.0));
        });
        ui.horizontal(|ui| {
            ui.label("luminance cutoff");
            ui.add(egui::DragValue::new(&mut self.luminance_cutoff).clamp_range(0.01..=0.5).speed(0.01));
        });
    }
}

const MAX_ALLOCATED_DEPTH_SLICES: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct ClusterDebugSettings {
    pub show_cluster_volumes: bool,
    pub show_all_cluster_volumes: bool,
    pub selected_cluster_id: [u32; 3],
}

impl Default for ClusterDebugSettings {
    fn default() -> Self {
        Self {
            show_cluster_volumes: false,
            show_all_cluster_volumes: false,
            selected_cluster_id: [0; 3],
        }
    }
}

impl ClusterDebugSettings {
    pub fn edit(&mut self, settings: &ClusterSettings, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.show_cluster_volumes, "show cluster volumes");
        // ui.checkbox(&mut self.show_all_cluster_volumes, "show all cluster volumes");
        ui.label("selected cluster id");
        ui.indent("selected_cluster_id", |ui| {
            let cluster_count = settings.cluster_counts();
            for i in 0..3 {
                ui.add(egui::Slider::new(
                    &mut self.selected_cluster_id[i],
                    0..=cluster_count[i] as u32 - 1,
                ));
            }
        });
    }
}

fn screen_to_view(screen_to_view_matrix: &Mat4, screen_size: Vec2, screen_pos: Vec4) -> Vec3A {
    let tex_coord = screen_pos.xy() / screen_size;
    let ndc = vec2(tex_coord.x, 1.0 - tex_coord.y) * 2.0 - 1.0;
    let clip = vec4(ndc.x, ndc.y, screen_pos.z, screen_pos.w);
    let mut view = screen_to_view_matrix.mul_vec4(clip);
    view /= view.w;
    view.into()
}

fn line_intersection_to_z_plane(a: Vec3A, b: Vec3A, z_distance: f32) -> Vec3A {
    let normal = vec3a(0.0, 0.0, -1.0);
    let ab = b - a;
    let t = (z_distance - Vec3A::dot(normal, a)) / Vec3A::dot(normal, ab);
    a + t * ab
}

#[rustfmt::skip]
fn compute_cluster_aabb(
	matrix: &Mat4, // inverse projection matrix
	screen_size: Vec2,
	tile_size_px: f32,
    cluster_count: Vec3A,
	z_near: f32,
	z_far: f32,
    cluster_id: Vec3A,
) -> Aabb {
    let eye_pos = Vec3A::splat(0.0);

    // screen space bounds
    let min_ss = cluster_id.xy() * tile_size_px;
    let max_ss = Vec2::min(min_ss + tile_size_px, screen_size);

    //Pass min and max to view space
    let min_vs = screen_to_view(matrix, screen_size, vec4(min_ss.x, min_ss.y, 1.0, 1.0));
    let max_vs = screen_to_view(matrix, screen_size, vec4(max_ss.x, max_ss.y, 1.0, 1.0));

    //Near and far values of the cluster in view space
    //We use equation (2) directly to obtain the tile values
    let cluster_near  = z_near * f32::powf(z_far / z_near,  cluster_id.z        / cluster_count.z);
    let cluster_far   = z_near * f32::powf(z_far / z_near, (cluster_id.z + 1.0) / cluster_count.z);

    //Finding the 4 intersection points made from each point to the cluster near/far plane
    let min_point_near = line_intersection_to_z_plane(eye_pos, min_vs, cluster_near);
    let min_point_far  = line_intersection_to_z_plane(eye_pos, min_vs, cluster_far );
    let max_point_near = line_intersection_to_z_plane(eye_pos, max_vs, cluster_near);
    let max_point_far  = line_intersection_to_z_plane(eye_pos, max_vs, cluster_far );

    let min = min_point_near.min(min_point_far).min(max_point_near).min(max_point_far);
    let max = min_point_near.max(min_point_far).max(max_point_near).max(max_point_far);

    Aabb { min, max }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct ClusterCullInfo {
    world_to_view_matrix: Mat4,
	screen_to_view_matrix: Mat4,
    
    cluster_count: [u32; 3],
	tile_size_px: u32,

    screen_size: [u32; 2],
	z_near: f32,
	z_far: f32,
    
    unique_cluster_buffer: u32,
    cluster_offset_image: u32,
    light_index_buffer: u32,
    depth_bounds_buffer: u32,

    global_light_count: u32,
    global_light_list: u32,
    _padding: [u32; 2],
}

fn compute_cluster_volume_corners(
    matrix: &Mat4, // inverse projection matrix
    screen_size: Vec2,
    tile_size_px: f32,
    cluster_count: Vec3A,
    z_near: f32,
    z_far: f32,
    cluster_id: Vec3A,
) -> [Vec4; 8] {
    let eye_pos = Vec3A::splat(0.0);

    // screen space bounds
    let min_ss = cluster_id.xy() * tile_size_px;
    let max_ss = Vec2::min(min_ss + tile_size_px, screen_size);

    //Pass min and max to view space
    let min_vs = screen_to_view(matrix, screen_size, vec4(min_ss.x, min_ss.y, 1.0, 1.0));
    let max_vs = screen_to_view(matrix, screen_size, vec4(max_ss.x, max_ss.y, 1.0, 1.0));

    //Near and far values of the cluster in view space
    //We use equation (2) directly to obtain the tile values
    let cluster_near = z_near * f32::powf(z_far / z_near, cluster_id.z / cluster_count.z);
    let cluster_far = z_near * f32::powf(z_far / z_near, (cluster_id.z + 1.0) / cluster_count.z);

    [
        vec3a(0.0, 0.0, 0.0),
        vec3a(1.0, 0.0, 0.0),
        vec3a(1.0, 1.0, 0.0),
        vec3a(0.0, 1.0, 0.0),
        vec3a(0.0, 0.0, 1.0),
        vec3a(1.0, 0.0, 1.0),
        vec3a(1.0, 1.0, 1.0),
        vec3a(0.0, 1.0, 1.0),
    ]
    .map(|s| {
        let corner = math::lerp_element_wise(min_vs.extend(1.0), max_vs.extend(1.0), vec4(s.x, s.y, 0.0, 1.0));
        let cluster_z = math::lerp(cluster_near, cluster_far, s.z);
        line_intersection_to_z_plane(eye_pos, corner.into(), cluster_z).extend(1.0)
    })
}

pub fn debug_cluster_volumes(
    settings: &ClusterSettings,
    debug_settings: &ClusterDebugSettings,
    camera: &Camera,
    debug_renderer: &mut DebugRenderer,
) {
    if !debug_settings.show_cluster_volumes {
        return;
    }

    let cluster_counts = settings.cluster_counts();
    let inverse_projection_matrix = camera.compute_projection_matrix().inverse();
    let view_to_world_matrix = camera.compute_view_matrix().inverse();

    if debug_settings.show_all_cluster_volumes {
        for x in 0..cluster_counts[0] {
            for y in 0..cluster_counts[1] {
                for z in 0..cluster_counts[2] {
                    let volume_corners = compute_cluster_volume_corners(
                        &inverse_projection_matrix,
                        Vec2::from_array(settings.screen_resolution.map(|x| x as f32)),
                        settings.tile_px_size() as f32,
                        Vec3A::from_array(cluster_counts.map(|x| x as f32)),
                        camera.z_near(),
                        settings.far_plane,
                        Vec3A::new(x as f32, y as f32, z as f32),
                    )
                    .map(|c| {
                        let c = view_to_world_matrix.mul_vec4(c);
                        c / c.w
                    });
                    debug_renderer.draw_cube_with_corners(&volume_corners, vec4(1.0, 1.0, 1.0, 1.0));
                }
            }
        }

        return;
    }

    let selected_cluster_id = [
        debug_settings.selected_cluster_id[0].min(cluster_counts[0] as u32 - 1),
        debug_settings.selected_cluster_id[1].min(cluster_counts[1] as u32 - 1),
        debug_settings.selected_cluster_id[2].min(cluster_counts[2] as u32 - 1),
    ];

    let aabb = compute_cluster_aabb(
        &inverse_projection_matrix,
        Vec2::from_array(settings.screen_resolution.map(|x| x as f32)),
        settings.tile_px_size() as f32,
        Vec3A::from_array(settings.cluster_counts().map(|x| x as f32)),
        camera.z_near(),
        settings.far_plane,
        Vec3A::from_array(selected_cluster_id.map(|x| x as f32)),
    );
    let aabb_corners = math::aabb_to_to_cube_corners(aabb, Some(&view_to_world_matrix));
    let volume_corners = compute_cluster_volume_corners(
        &inverse_projection_matrix,
        Vec2::from_array(settings.screen_resolution.map(|x| x as f32)),
        settings.tile_px_size() as f32,
        Vec3A::from_array(cluster_counts.map(|x| x as f32)),
        camera.z_near(),
        settings.far_plane,
        Vec3A::from_array(selected_cluster_id.map(|x| x as f32)),
    )
    .map(|c| {
        let c = view_to_world_matrix.mul_vec4(c);
        c / c.w
    });
    debug_renderer.draw_cube_with_corners(&aabb_corners, vec4(0.0, 1.0, 0.0, 1.0));
    debug_renderer.draw_cube_with_corners(&volume_corners, vec4(1.0, 1.0, 1.0, 1.0));
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuClusterInfoBuffer {
    cluster_count: [u32; 3],
    tile_size_px: u32,
    screen_size: [u32; 2],
    z_slice_count: u32,
    z_scale: f32,
    z_bias: f32,
    luminance_cutoff: f32,
    light_offset_image: u32,
    light_index_list: u32,
    tile_depth_slice_mask_buffer: u32,
}

impl GpuClusterInfoBuffer {
    pub fn new(
        context: &graphics::Context,
        settings: &ClusterSettings,
        z_near: f32,
        offset_image: graphics::GraphImageHandle,
        index_list: graphics::GraphBufferHandle,
        tile_depth_slice_mask_buffer: graphics::GraphBufferHandle,
    ) -> Self {
        let (z_scale, z_bias) = settings.cluster_grid_info(z_near);
        Self {
            cluster_count: settings.cluster_counts().map(|x| x as u32),
            tile_size_px: settings.tile_px_size(),
            screen_size: settings.screen_resolution,
            z_scale,
            z_bias,
            z_slice_count: settings.z_slice_count,
            luminance_cutoff: settings.luminance_cutoff,
            light_offset_image: context.get_resource_descriptor_index(offset_image).unwrap(),
            light_index_list: context.get_resource_descriptor_index(index_list).unwrap(),
            tile_depth_slice_mask_buffer: context.get_resource_descriptor_index(tile_depth_slice_mask_buffer).unwrap(),
        }
    }
}

pub struct GraphClusterInfo {
    pub light_offset_image: graphics::GraphImageHandle,
    pub light_index_list: graphics::GraphBufferHandle,
    pub info_buffer: graphics::GraphBufferHandle,
}

pub fn compute_clusters(
    context: &mut graphics::Context,
    settings: &ClusterSettings,
    camera: &Camera,
    depth_buffer: graphics::GraphImageHandle,
    scene: SceneGraphData,
) -> GraphClusterInfo {
    let (active_cluster_mask, depth_bounds) = mark_active_clusters(context, &settings, depth_buffer, &camera);
    let unique_cluster_buffer = compact_active_clusters(context, &settings, active_cluster_mask);

    let (light_offset_image, light_index_list) = cluster_light_assignment(
        context,
        settings,
        camera,
        scene,
        unique_cluster_buffer,
        depth_bounds,
    );

    let data = GpuClusterInfoBuffer::new(
        context,
        settings,
        camera.z_near(),
        light_offset_image,
        light_index_list,
        active_cluster_mask,
    );

    let info_buffer = context.transient_storage_data("cluster_info_buffer", bytemuck::bytes_of(&data));

    GraphClusterInfo {
        light_offset_image,
        light_index_list,
        info_buffer,
    }
}

pub fn mark_active_clusters(
    context: &mut graphics::Context,
    settings: &ClusterSettings,
    depth_buffer: graphics::GraphImageHandle,
    camera: &Camera,
) -> (graphics::GraphBufferHandle, graphics::GraphBufferHandle) {
    let pipeline = context.create_compute_pipeline(
        "mark_active_clusters_pipeline",
        ShaderStage::spv("shaders/light_cluster/mark_active.comp.spv"),
    );

    let tile_count = settings.tile_counts();

    let tile_depth_slice_mask = context.create_transient_buffer(
        "tile_depth_slice_mask",
        graphics::BufferDesc {
            size: tile_count[0] * tile_count[1] * 4,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        },
    );

    let cluster_depth_bounds_buffer = context.create_transient_buffer(
        "cluster_depth_bounds_buffer",
        graphics::BufferDesc {
            size: settings.linear_cluster_count() * 8,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        },
    );

    let tile_size_px = settings.tile_px_size();
    let z_near = camera.z_near();
    let z_far = settings.far_plane;
    let (z_scale, z_bias) = settings.cluster_grid_info(z_near);

    let cluster_count = settings.cluster_counts().map(|x| x as u32);
    let cluster_linear_count = settings.linear_cluster_count();
    let screen_resolution = settings.screen_resolution;

    context
        .add_pass("clear_cluster_depth_slice_buffers")
        .with_dependency(tile_depth_slice_mask, AccessKind::ComputeShaderWrite)
        .with_dependency(cluster_depth_bounds_buffer, AccessKind::ComputeShaderWrite)
        .record_custom(move |cmd, graph| {
            cmd.fill_buffer(
                graph.get_buffer(tile_depth_slice_mask),
                0,
                (tile_count[0] * tile_count[1] * 4) as u64,
                0,
            );
            cmd.fill_buffer(
                graph.get_buffer(cluster_depth_bounds_buffer),
                0,
                cluster_linear_count as u64 * 8,
                0,
            );
        });

    let depth_buffer_desc = context.get_image_desc(depth_buffer);
    let depth_buffer_size = [depth_buffer_desc.dimensions[0], depth_buffer_desc.dimensions[1]];
    let depth_buffer_sample_count = depth_buffer_desc.samples.sample_count();

    ComputePass::new(context, "mark_active_tiles", pipeline)
        .push_data(cluster_count)
        .push_data(tile_size_px)
        .push_data(depth_buffer_size)
        .push_data(z_near)
        .push_data(z_far)
        .push_data(z_scale)
        .push_data(z_bias)
        .read_image(depth_buffer)
        .push_data(depth_buffer_sample_count)
        .write_buffer(tile_depth_slice_mask)
        .write_buffer(cluster_depth_bounds_buffer)
        .dispatch([screen_resolution[0].div_ceil(8), screen_resolution[1].div_ceil(8), 1]);

    (tile_depth_slice_mask, cluster_depth_bounds_buffer)
}

pub fn compact_active_clusters(
    context: &mut graphics::Context,
    settings: &ClusterSettings,
    active_cluster_mask: graphics::GraphBufferHandle,
) -> graphics::GraphBufferHandle {
    let pipeline = context.create_compute_pipeline(
        "compact_active_clusters_pipeline",
        ShaderStage::spv("shaders/light_cluster/active_cluster_compaction.comp.spv"),
    );

    let unique_cluster_buffer = context.create_transient_buffer(
        "unique_cluster_buffer",
        graphics::BufferDesc {
            size: 16 + settings.linear_max_allocated_cluster_count() * 4,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        },
    );

    let cluster_count = settings.cluster_counts().map(|x| x as u32);

    context
        .add_pass("clear_compact_cluster_buffer")
        .with_dependency(unique_cluster_buffer, AccessKind::ComputeShaderWrite)
        .record_custom(move |cmd, graph| {
            let compact_cluster_index_list = graph.get_buffer(unique_cluster_buffer);
            cmd.fill_buffer(compact_cluster_index_list, 0, 16, 0);
        });

    ComputePass::new(context, "compact_active_clusters", pipeline)
        .push_data(cluster_count)
        .read_buffer(active_cluster_mask)
        .write_buffer(unique_cluster_buffer)
        .dispatch(cluster_count.map(|x| x.div_ceil(4)));

    unique_cluster_buffer
}

pub fn cluster_light_assignment(
    context: &mut graphics::Context,
    settings: &ClusterSettings,
    camera: &Camera,
    scene: SceneGraphData,
    unique_cluster_buffer: graphics::GraphBufferHandle,
    depth_bounds_buffer: graphics::GraphBufferHandle,
) -> (graphics::GraphImageHandle, graphics::GraphBufferHandle) {
    let pipeline = context.create_compute_pipeline(
        "cluster_light_assingment_pipeline",
        ShaderStage::spv("shaders/light_cluster/light_culling.comp.spv"),
    );

    let light_offset_image = context.create_transient_image(
        "cluster_offset_image",
        graphics::ImageDesc {
            ty: graphics::ImageType::Single3D,
            format: vk::Format::R32G32_UINT,
            dimensions: settings.cluster_counts().map(|x| x as u32),
            mip_levels: 1,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            ..Default::default()
        },
    );

    let light_index_buffer = context.create_transient_buffer(
        "light_index_buffer",
        graphics::BufferDesc {
            size: 4 + settings.linear_max_allocated_cluster_count() as usize * 32 * 4,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: MemoryLocation::GpuOnly,
        },
    );

    let cluster_cull_data = ClusterCullInfo {
        world_to_view_matrix: camera.compute_view_matrix(),
        screen_to_view_matrix: camera.compute_projection_matrix().inverse(),
        cluster_count: settings.cluster_counts().map(|x| x as u32),
        tile_size_px: settings.tile_px_size(),
        screen_size: settings.screen_resolution,
        z_near: camera.z_near(),
        z_far: settings.far_plane,
        unique_cluster_buffer: context.get_resource_descriptor_index(unique_cluster_buffer).unwrap(),
        cluster_offset_image: context.get_resource_descriptor_index(light_offset_image).unwrap(),
        light_index_buffer: context.get_resource_descriptor_index(light_index_buffer).unwrap(),
        depth_bounds_buffer: context.get_resource_descriptor_index(depth_bounds_buffer).unwrap(),
        global_light_count: scene.light_count as u32,
        global_light_list: context.get_resource_descriptor_index(scene.light_data_buffer).unwrap(),
        _padding: [0; 2],
    };

    let cluster_cull_info_buffer = context.transient_storage_data(
        "cluster_cull_info_buffer",
        bytemuck::bytes_of(&cluster_cull_data),
    );

    context
        .add_pass("clear_light_index_buffer")
        .with_dependency(light_index_buffer, AccessKind::ComputeShaderWrite)
        .record_custom(move |cmd, graph| {
            let compact_cluster_index_list = graph.get_buffer(light_index_buffer);
            cmd.fill_buffer(compact_cluster_index_list, 0, 4, 0);
        });

    ComputePass::new(context, "cluster_light_assingment", pipeline)
        .with_dependency(depth_bounds_buffer, AccessKind::ComputeShaderRead)
        .with_dependency(light_offset_image, AccessKind::ComputeShaderWrite)
        .with_dependency(light_index_buffer, AccessKind::ComputeShaderWrite)
        .read_buffer(cluster_cull_info_buffer)
        .dispatch_indirect(unique_cluster_buffer, 0);

    (light_offset_image, light_index_buffer)
}
