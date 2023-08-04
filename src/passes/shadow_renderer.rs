use std::borrow::Cow;

use ash::vk;

#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Vec2, Vec3, Vec3A, Vec4, Quat, Mat4};

use crate::{
    render,
    math,
    utils,
    assets::AssetGraphData,
    scene::{SceneGraphData, GpuDrawCommand},
    App,
    Camera,
    Projection,
    MAX_DRAW_COUNT,
    MAX_SHADOW_CASCADE_COUNT,
};

use super::draw_gen::{SceneDrawGen, FrustumPlaneMask};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuDirectionalLight {
    projection_matrices: [Mat4; MAX_SHADOW_CASCADE_COUNT],
    shadow_maps: [u32; MAX_SHADOW_CASCADE_COUNT],
    color: Vec3,
    intensity: f32,
    direction: Vec3,
    _padding0: u32,
    
    penumbra_filter_max_size: f32,
    min_filter_radius: f32,
    max_filter_radius: f32,
    _padding1: u32
}

#[derive(Clone, Copy)]
pub struct DirectionalLightGraphData {
    pub shadow_maps: [render::GraphImageHandle; MAX_SHADOW_CASCADE_COUNT],
    pub buffer: render::GraphBufferHandle
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

    pub penumbra_filter_max_size: f32,
    pub min_filter_radius: f32,
    pub max_filter_radius: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            shadow_resolution: 4096,

            depth_bias_constant_factor: -6.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: -6.0,
            
            cascade_split_lambda: 0.8,
            max_shadow_distance: 100.0,

            penumbra_filter_max_size: 8.0,
            min_filter_radius: 0.5,
            max_filter_radius: 8.0,
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
            ui.add(egui::DragValue::new(&mut self.min_filter_radius).speed(0.1));
        });

        ui.horizontal(|ui| {
            ui.label("max_filter_radius");
            ui.add(egui::DragValue::new(&mut self.max_filter_radius).speed(0.1));
        });

        ui.horizontal(|ui| {
            ui.label("penumbra_filter_max_size");
            ui.add(egui::DragValue::new(&mut self.penumbra_filter_max_size).speed(0.1));
        });

        ui.horizontal(|ui| {
            ui.label("max_shadow_distance");
            ui.add(egui::DragValue::new(&mut self.max_shadow_distance).speed(1.0));
        });

        ui.horizontal(|ui| {
            ui.label("lambda");
            ui.add(egui::Slider::new(&mut self.cascade_split_lambda, 0.0..=1.0));
        });
    }
}

pub struct ShadowMapRenderer {
    pipeline: render::RasterPipeline,
    pub settings: ShadowSettings,
}

impl ShadowMapRenderer {
    pub fn new(context: &render::Context) -> Self {
        let pipeline = {
            let vertex_shader = utils::load_spv("shaders/shadow.vert.spv").unwrap();
            let vertex_module = context.create_shader_module(&vertex_shader, "shadow_vertex_shader");
            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("shadowmap_renderer_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: None,
                vertex_input: render::VertexInput::default(),
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::BACK,
                    depth_bias: Some(render::DepthBias::default()),
                    depth_clamp: true,
                },
                color_attachments: &[],
                depth_state: Some(render::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: true,
                    write: true,
                    compare: vk::CompareOp::GREATER,
                }),
                multisample: render::MultisampleCount::None,
                dynamic_states: &[vk::DynamicState::DEPTH_BIAS]
            });

            context.destroy_shader_module(vertex_module);

            pipeline
        };

        Self { pipeline, settings: Default::default() }
    }

    pub fn render_shadow_map(
        &self,
        name: Cow<'static, str>,
        frame_ctx: &mut render::Context,

        resolution: u32,
        view_projection: Mat4,
        draw_commands: render::GraphBufferHandle,

        assets: AssetGraphData,
        scene: SceneGraphData,
    ) -> render::GraphImageHandle {
        let pass_name = format!("shadow_pass_for_{name}");
        let shadow_map = frame_ctx.create_transient_image(name, render::ImageDesc {
            ty: render::ImageType::Single2D,
            format: App::DEPTH_FORMAT,
            dimensions: [resolution, resolution, 1],
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::DEPTH,
        });

        let pipeline = self.pipeline;

        let settings = self.settings;

        frame_ctx.add_pass(pass_name)
            .with_dependency(shadow_map, render::AccessKind::DepthAttachmentWrite)
            .with_dependency(draw_commands, render::AccessKind::IndirectBuffer)
            .render(move |cmd, graph| {
                let shadow_map = graph.get_image(shadow_map);
                
                let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
                let index_buffer = graph.get_buffer(assets.index_buffer);
                let entity_buffer = graph.get_buffer(scene.entity_buffer);
                let draw_commands_buffer = graph.get_buffer(draw_commands);

                let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
                    .image_view(shadow_map.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
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
                    .buffer(&vertex_buffer)
                    .buffer(&entity_buffer);

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
        &self,
        context: &mut render::Context,
        name: Cow<'static, str>,
        
        direction: Quat,
        light_color: Vec3,
        intensity: f32,

        camera: &Camera,
        aspect_ratio: f32,

        draw_gen: &SceneDrawGen,

        assets: AssetGraphData,
        scene: SceneGraphData,
    ) -> DirectionalLightGraphData {
        let light_direction = -direction.mul_vec3(vec3(0.0, 0.0, -1.0));
        let inv_light_direction = direction.inverse();
        
        let mut directional_light_data = GpuDirectionalLight {
            projection_matrices: bytemuck::Zeroable::zeroed(),
            shadow_maps: bytemuck::Zeroable::zeroed(),
            color: light_color,
            intensity,
            direction: light_direction,
            _padding0: 0,

            penumbra_filter_max_size: self.settings.penumbra_filter_max_size,
            max_filter_radius: self.settings.max_filter_radius,
            min_filter_radius: self.settings.min_filter_radius,
            _padding1: 0,
        };

        let mut shadow_maps = [render::GraphHandle::uninit(); MAX_SHADOW_CASCADE_COUNT];
        
        let lambda = self.settings.cascade_split_lambda;

        for cascade_index in 0..MAX_SHADOW_CASCADE_COUNT {
            let Projection::Perspective { fov, near_clip } = camera.projection else { todo!() };
            let far_clip = self.settings.max_shadow_distance;
            
            let near_split_ratio = cascade_index as f32 / MAX_SHADOW_CASCADE_COUNT as f32;
            let far_split_ratio = (cascade_index+1) as f32 / MAX_SHADOW_CASCADE_COUNT as f32;
            
            let near = math::frustum_split(near_clip, far_clip, lambda, near_split_ratio);
            let far = math::frustum_split(near_clip, far_clip, lambda, far_split_ratio);

            let light_matrix = Mat4::from_quat(inv_light_direction);
            let view_to_world_matrix = camera.transform.compute_matrix();

            let subfrustum_corners_view_space = math::perspective_corners(fov, aspect_ratio, near, far);
            
            let mut subfrustum_center_view_space = Vec4::ZERO;
            for corner in subfrustum_corners_view_space.iter().copied() {
                subfrustum_center_view_space += corner;
            }
            subfrustum_center_view_space /= 8.0;
            
            let mut radius_sqr = 0.0;
            for corner in subfrustum_corners_view_space.iter().copied() {
                let corner = Vec3A::from(corner);
                radius_sqr = f32::max(radius_sqr, corner.distance_squared(subfrustum_center_view_space.into()));
            }
            let radius = radius_sqr.sqrt();

            let view_to_light_matrix = light_matrix * view_to_world_matrix;
            let subfrustum_center_light_space = view_to_light_matrix * subfrustum_center_view_space;
            let subfrustum_center_light_space = Vec3A::from(subfrustum_center_light_space)
                / subfrustum_center_light_space.w;

            let texel_size_vs = radius * 2.0 / self.settings.shadow_resolution as f32;
            
            // texel alignment
            let subfrustum_center_light_space = (subfrustum_center_light_space / texel_size_vs).floor() * texel_size_vs;

            let mut max_extent = Vec3A::splat(radius);
            let mut min_extent = -max_extent;

            max_extent += subfrustum_center_light_space;
            min_extent += subfrustum_center_light_space;

            let projection_matrix = Mat4::orthographic_rh(
                min_extent.x, max_extent.x,
                min_extent.y, max_extent.y,
                150.0, -150.0,
            );

            // if self.show_cascade_view_frustum && self.selected_cascade == cascade_index {
            //     let subfrustum_corners_w = subfrustum_corners_view_space.map(|v| view_to_world_matrix * v);
            //     self.debug_line_renderer.draw_frustum(&subfrustum_corners_w, Vec4::splat(1.0));
            // }
            
            // if self.show_cascade_light_frustum && self.selected_cascade == cascade_index {
            //     let cascade_frustum_corners = frustum_corners_from_matrix(&(projection_matrix * light_matrix));
            //     self.debug_line_renderer.draw_frustum(&cascade_frustum_corners, vec4(1.0, 1.0, 0.0, 1.0));
            // }
            
            let light_projection_matrix = projection_matrix * light_matrix;
            
            let shadow_map_draw_commands = draw_gen.create_draw_commands(
                context,
                format!("{name}_{cascade_index}_draw_commands").into(),
                &light_projection_matrix,
                FrustumPlaneMask::SIDES,
                assets,
                scene,
            );
            
            let shadow_map = self.render_shadow_map(
                format!("{name}_{cascade_index}").into(),
                context,
                self.settings.shadow_resolution,
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

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
    }
}