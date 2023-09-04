use std::borrow::Cow;

use ash::vk;

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4, Quat, Mat4};

use crate::{
    graphics,
    math,
    assets::AssetGraphData,
    scene::{SceneGraphData, GpuDrawCommand},
    App,
    Camera,
    Projection,
    MAX_DRAW_COUNT,
    MAX_SHADOW_CASCADE_COUNT,
};

use super::{draw_gen::{FrustumPlaneMask, create_draw_commands}, debug_line_renderer::DebugLineRenderer};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuDirectionalLight {
    projection_matrices: [Mat4; MAX_SHADOW_CASCADE_COUNT],
    shadow_maps: [u32; MAX_SHADOW_CASCADE_COUNT],
    cascade_distances: Vec4,
    color: Vec3,
    intensity: f32,
    direction: Vec3,
    blend_seam: f32,
    
    penumbra_filter_max_size: f32,
    min_filter_radius: f32,
    max_filter_radius: f32,
    _padding1: u32
}

#[derive(Clone, Copy)]
pub struct DirectionalLightGraphData {
    pub shadow_maps: [graphics::GraphImageHandle; MAX_SHADOW_CASCADE_COUNT],
    pub buffer: graphics::GraphBufferHandle
}

#[derive(Debug, Clone, Copy)]
pub struct ShadowSettings {
    pub shadow_resolution: u32,

    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,

    // directional
    pub cascade_split_lambda: f32,
    pub max_shadow_distance: f32,
    pub split_blend_ratio: f32,

    pub penumbra_filter_max_size: f32,
    pub min_filter_radius: f32,
    pub max_filter_radius: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            shadow_resolution: 512,

            depth_bias_constant_factor: 6.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 6.0,
            
            cascade_split_lambda: 0.91,
            max_shadow_distance: 100.0,
            split_blend_ratio: 0.5,

            penumbra_filter_max_size: 6.0,
            min_filter_radius: 1.0,
            max_filter_radius: 6.0,
        }
    }
}

impl ShadowSettings {
    pub fn edit(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("shadow_resolution");
            egui::ComboBox::from_id_source("shadow_resolution")
                .selected_text(format!("{}", self.shadow_resolution))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.shadow_resolution, 512, "512");
                    ui.selectable_value(&mut self.shadow_resolution, 1024, "1024");
                    ui.selectable_value(&mut self.shadow_resolution, 2048, "2048");
                    ui.selectable_value(&mut self.shadow_resolution, 4096, "4096");
                });
        });


        ui.horizontal(|ui| {
            ui.label("depth_bias_constant_factor");
            ui.add(egui::DragValue::new(&mut self.depth_bias_constant_factor).speed(0.01));
        });

        ui.horizontal(|ui| {
            ui.label("depth_bias_clamp");
            ui.add(egui::DragValue::new(&mut self.depth_bias_clamp).speed(0.01));
        });
        ui.horizontal(|ui| {
            ui.label("depth_bias_slope_factor");
            ui.add(egui::DragValue::new(&mut self.depth_bias_slope_factor).speed(0.01));
        });

        ui.horizontal(|ui| {
            ui.label("min_filter_radius");
            ui.add(egui::DragValue::new(&mut self.min_filter_radius).speed(0.1).clamp_range(0.0..=32.0));
        });

        ui.horizontal(|ui| {
            ui.label("max_filter_radius");
            ui.add(egui::DragValue::new(&mut self.max_filter_radius).speed(0.1).clamp_range(0.0..=32.0));
        });

        ui.horizontal(|ui| {
            ui.label("penumbra_filter_max_size");
            ui.add(egui::DragValue::new(&mut self.penumbra_filter_max_size).speed(0.1).clamp_range(0.0..=32.0));
        });

        ui.horizontal(|ui| {
            ui.label("max_shadow_distance");
            ui.add(egui::DragValue::new(&mut self.max_shadow_distance).speed(0.5).clamp_range(0.0..=1000.0));
        });

        ui.horizontal(|ui| {
            ui.label("split_blend_ratio");
            ui.add(egui::DragValue::new(&mut self.split_blend_ratio).speed(0.005).clamp_range(0.0..=1.0));
        });

        ui.horizontal(|ui| {
            ui.label("lambda");
            ui.add(egui::Slider::new(&mut self.cascade_split_lambda, 0.0..=1.0));
        });
    }
}

pub fn render_shadow_map(
    context: &mut graphics::Context,
    settings: &ShadowSettings,
    name: Cow<'static, str>,

    resolution: u32,
    view_projection: Mat4,
    draw_commands: graphics::GraphBufferHandle,

    assets: AssetGraphData,
    scene: SceneGraphData,
) -> graphics::GraphImageHandle {
    let pass_name = format!("shadow_pass_for_{name}");
    let shadow_map = context.create_transient(name, graphics::ImageDesc {
        ty: graphics::ImageType::Single2D,
        format: App::DEPTH_FORMAT,
        dimensions: [resolution, resolution, 1],
        mip_levels: 1,
        samples: graphics::MultisampleCount::None,
        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        aspect: vk::ImageAspectFlags::DEPTH,
        subresource_desc: graphics::ImageSubresourceViewDesc::default(),
        ..Default::default()
    });

    let settings = settings.clone();

    let pipeline = context.create_raster_pipeline(
        "shadowmap_renderer_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/shadow.vert.spv"))
            .fragment_shader(graphics::ShaderSource::spv("shaders/shadow.frag.spv"))
            .rasterizer(graphics::RasterizerDesc {
                depth_clamp: true,
                ..Default::default()
            })
            .depth_bias_dynamic()
            .depth_state(Some(graphics::DepthState {
                format: App::DEPTH_FORMAT,
                test: graphics::PipelineState::Static(true),
                write: true,
                compare: vk::CompareOp::LESS,
            }))
    );

    context.add_pass(pass_name)
        .with_dependency(shadow_map, graphics::AccessKind::DepthAttachmentWrite)
        .with_dependency(draw_commands, graphics::AccessKind::IndirectBuffer)
        .render(move |cmd, graph| {
            let shadow_map = graph.get_image(shadow_map);
            
            let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
            let index_buffer = graph.get_buffer(assets.index_buffer);
            let entity_buffer = graph.get_buffer(scene.entity_buffer);
            let draw_commands_buffer = graph.get_buffer(draw_commands);
            let materials_buffer = graph.get_buffer(assets.materials_buffer);

            let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
                .image_view(shadow_map.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                })
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::builder()
                .render_area(shadow_map.full_rect())
                .layer_count(1)
                .depth_attachment(&depth_attachemnt);

            cmd.begin_rendering(&rendering_info);
            
            cmd.bind_raster_pipeline(pipeline);
            cmd.bind_index_buffer(&index_buffer, 0);
            cmd.set_depth_bias(
                settings.depth_bias_constant_factor,
                settings.depth_bias_clamp,
                settings.depth_bias_slope_factor
            );

            cmd.build_constants()
                .mat4(&view_projection)
                .buffer(vertex_buffer)
                .buffer(entity_buffer)
                .buffer(draw_commands_buffer)
                .buffer(materials_buffer);

            cmd.draw_indexed_indirect_count(
                draw_commands_buffer,
                4,
                draw_commands_buffer,
                0,
                MAX_DRAW_COUNT as u32,
                std::mem::size_of::<GpuDrawCommand>() as u32,
            );

            cmd.end_rendering();
        });

    shadow_map
}

pub fn render_directional_light(
    context: &mut graphics::Context,
    settings: &ShadowSettings,
    name: Cow<'static, str>,
    
    direction: Quat,
    light_color: Vec3,
    intensity: f32,

    camera: &Camera,
    aspect_ratio: f32,

    assets: AssetGraphData,
    scene: SceneGraphData,

    selected_cascade: usize,
    show_cascade_view_frustum: bool,
    show_cascade_light_frustum: bool,
    debug_line_renderer: &mut DebugLineRenderer,
) -> DirectionalLightGraphData {
    let light_direction = -direction.mul_vec3(vec3(0.0, 0.0, -1.0));
    let inv_light_direction = direction.inverse();
    
    let mut directional_light_data = GpuDirectionalLight {
        projection_matrices: bytemuck::Zeroable::zeroed(),
        shadow_maps: bytemuck::Zeroable::zeroed(),
        cascade_distances: Vec4::ZERO,
        color: light_color,
        intensity,
        direction: light_direction,
        blend_seam: settings.split_blend_ratio,

        penumbra_filter_max_size: settings.penumbra_filter_max_size,
        max_filter_radius: settings.max_filter_radius,
        min_filter_radius: settings.min_filter_radius,
        _padding1: 0,
    };

    let mut shadow_maps = [graphics::GraphHandle::uninit(); MAX_SHADOW_CASCADE_COUNT];
    
    let lambda = settings.cascade_split_lambda;

    for cascade_index in 0..MAX_SHADOW_CASCADE_COUNT {
        let Projection::Perspective { fov, near_clip } = camera.projection else { todo!() };
        let far_clip = settings.max_shadow_distance;
        
        let near_split_ratio = cascade_index as f32 / MAX_SHADOW_CASCADE_COUNT as f32;
        let far_split_ratio = (cascade_index+1) as f32 / MAX_SHADOW_CASCADE_COUNT as f32;
        
        let near = math::frustum_split(near_clip, far_clip, lambda, near_split_ratio);
        let far = math::frustum_split(near_clip, far_clip, lambda, far_split_ratio);

        directional_light_data.cascade_distances[cascade_index] = far;

        let light_matrix = Mat4::from_quat(inv_light_direction);
        let view_to_world_matrix = camera.transform.compute_matrix();
        let view_to_light_matrix = light_matrix * view_to_world_matrix;

        let subfrustum_corners_light_space = math::perspective_corners(fov, aspect_ratio, near, far)
            .map(|corner| {
                let v = view_to_light_matrix * corner;
                v / v.w
            });

        let mut subfrustum_center_light_space = Vec4::ZERO;
        let mut min_coords_light_space = Vec3A::from(subfrustum_corners_light_space[0]);
        let mut max_coords_light_space = Vec3A::from(subfrustum_corners_light_space[0]);
        for corner_light_space in subfrustum_corners_light_space.iter().copied() {
            subfrustum_center_light_space += corner_light_space;
            min_coords_light_space = Vec3A::min(min_coords_light_space, corner_light_space.into());
            max_coords_light_space = Vec3A::max(max_coords_light_space, corner_light_space.into());
        }
        subfrustum_center_light_space /= 8.0;
        
        let mut radius_sqr = 0.0;
        for corner_light_space in subfrustum_corners_light_space.iter().copied() {
            let corner = Vec3A::from(corner_light_space);
            radius_sqr = f32::max(radius_sqr, corner.distance_squared(subfrustum_center_light_space.into()));
        }
        let radius = radius_sqr.sqrt();

        // offsets the whole projection to minimize the area not visible by the camera
        // this one isn't perfect but close enough
        let forward_offset = {
            let min_coords = min_coords_light_space - Vec3A::from(subfrustum_center_light_space);
            let max_coords = max_coords_light_space - Vec3A::from(subfrustum_center_light_space);

            let forward_sign = -view_to_light_matrix.transform_vector3a(vec3a(0.0, 0.0, -1.0));
            let forward_a = (forward_sign + 1.0) / 2.0;

            let offset = math::lerp_element_wise(min_coords.extend(1.0), max_coords.extend(1.0), forward_a.extend(1.0));
            let offset = Vec3A::from(offset) - Vec3A::splat(radius) * forward_sign;

            offset
        };

        // if cascade_index == selected_cascade {
        //     let min_max_corners = math::NDC_BOUNDS.map(|corner| {
        //         let mut a = (corner + Vec4::ONE) / 2.0;
        //         a.w = 1.0;
        //         let min = min_coords_light_space.extend(1.0);
        //         let max = max_coords_light_space.extend(1.0);
        //         light_matrix.inverse() * math::lerp_element_wise(min, max, a)
        //     });
        //     debug_line_renderer.draw_frustum(&min_max_corners, vec4(0.0, 0.0, 1.0, 1.0));
        // }

        let texel_size_vs = radius * 2.0 / settings.shadow_resolution as f32;
        
        // modifications:
        //  - aligned to view space texel sizes
        //  - offset in the direction of the camera
        let subfrustum_center_modified_light_space =
            ((Vec3A::from(subfrustum_center_light_space) + forward_offset) / texel_size_vs).floor() * texel_size_vs;

        let mut max_extent = Vec3A::splat(radius);
        let mut min_extent = -max_extent;

        max_extent += subfrustum_center_modified_light_space;
        min_extent += subfrustum_center_modified_light_space;

        let projection_matrix = Mat4::orthographic_rh(
            min_extent.x, max_extent.x,
            min_extent.y, max_extent.y,
            // min_coords_light_space.x, max_coords_light_space.x,
            // min_coords_light_space.y, max_coords_light_space.y,
            -max_coords_light_space.z - 200.0,
            // -min_coords_light_space.z + 200.0,
            -subfrustum_center_modified_light_space.z + radius,
        );

        if show_cascade_view_frustum && selected_cascade == cascade_index {
            let subfrustum_corners_w = math::perspective_corners(fov, aspect_ratio, near, far)
                .map(|v| view_to_world_matrix * v);
            debug_line_renderer.draw_frustum(&subfrustum_corners_w, Vec4::splat(1.0));
        }
        
        if show_cascade_light_frustum && selected_cascade == cascade_index {
            let cascade_frustum_corners = math::frustum_corners_from_matrix(&(projection_matrix * light_matrix));
            debug_line_renderer.draw_frustum(&cascade_frustum_corners, vec4(1.0, 1.0, 0.0, 1.0));
        }
        
        let light_projection_matrix = projection_matrix * light_matrix;

        let shadow_map_draw_commands = create_draw_commands(
            context,
            format!("{name}_{cascade_index}_draw_commands").into(),
            &light_projection_matrix,
            FrustumPlaneMask::SIDES,
            None,
            assets,
            scene,
        );
        
        let shadow_map = render_shadow_map(
            context,
            settings,
            format!("{name}_{cascade_index}").into(),
            settings.shadow_resolution,
            light_projection_matrix,
            shadow_map_draw_commands,
            assets,
            scene,
        );
        
        directional_light_data.projection_matrices[cascade_index] = light_projection_matrix;
        directional_light_data.shadow_maps[cascade_index] = context
            .get_transient_resource_descriptor_index(shadow_map)
            .unwrap();
        shadow_maps[cascade_index] = shadow_map;
    };

    let buffer = context
        .transient_storage_data(format!("{name}_light_data"), bytemuck::bytes_of(&directional_light_data));

    DirectionalLightGraphData {
        shadow_maps,
        buffer,
    }
}