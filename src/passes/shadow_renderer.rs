use std::borrow::Cow;

use ash::vk;

use bytemuck::Zeroable;
use glam::Vec2Swizzles;
#[allow(unused_imports)]
use glam::{vec2, vec3, vec3a, vec4, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};

use crate::{
    assets::{AssetGraphData, GpuAssets},
    gltf_loader::mip_levels_from_size,
    graphics::{self, FRAME_COUNT},
    math,
    scene::{GpuMeshDrawCommand, SceneData, SceneGraphData},
    Camera, Projection, MAX_DRAW_COUNT, MAX_SHADOW_CASCADE_COUNT,
};

use super::{
    debug_renderer::DebugRenderer,
    draw_gen::{create_draw_commands, AlphaModeFlags, CullInfo, OcclusionCullInfo},
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuCascadedShadowData {
    light_projection_matrix: [Mat4; MAX_SHADOW_CASCADE_COUNT],
    shadow_map_world_sizes: [f32; MAX_SHADOW_CASCADE_COUNT],
    shadow_map_indices: [u32; MAX_SHADOW_CASCADE_COUNT],
}

#[derive(Clone, Copy)]
pub struct DirectionalLightGraphData {
    pub shadow_maps: [graphics::GraphImageHandle; MAX_SHADOW_CASCADE_COUNT],
    pub buffer: graphics::GraphBufferHandle,
}

#[derive(Debug, Clone, Copy)]
pub struct ShadowSettings {
    pub shadow_resolution: u32,
    pub blocker_search_radius: f32,

    pub depth_bias_constant_factor: f32,
    pub depth_bias_slope_factor: f32,
    pub depth_bias_normal_scale: f32,
    pub depth_bias_oriented: f32,

    // directional
    pub cascade_split_lambda: f32,
    pub max_shadow_distance: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            shadow_resolution: 2048,
            blocker_search_radius: 0.3,

            depth_bias_constant_factor: 0.0,
            depth_bias_slope_factor: 2.0,
            depth_bias_normal_scale: 0.0,
            depth_bias_oriented: 0.02,

            cascade_split_lambda: 0.80,
            max_shadow_distance: 32.0,
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
            ui.label("blocker_search_radius");
            ui.add(egui::DragValue::new(&mut self.blocker_search_radius).speed(0.01).clamp_range(0.0..=16.0));
        });

        ui.horizontal(|ui| {
            ui.label("depth_bias_constant_factor");
            ui.add(egui::DragValue::new(&mut self.depth_bias_constant_factor).speed(0.01).clamp_range(0.0..=16.0));
        });

        ui.horizontal(|ui| {
            ui.label("depth_bias_slope_factor");
            ui.add(egui::DragValue::new(&mut self.depth_bias_slope_factor).speed(0.01).clamp_range(0.0..=16.0));
        });

        ui.horizontal(|ui| {
            ui.label("depth_bias_normal_scale");
            ui.add(egui::DragValue::new(&mut self.depth_bias_normal_scale).speed(0.01).clamp_range(-16.0..=16.0));
        });

        ui.horizontal(|ui| {
            ui.label("depth_bias_oriented");
            ui.add(egui::DragValue::new(&mut self.depth_bias_oriented).speed(0.01).clamp_range(-16.0..=16.0));
        });

        ui.horizontal(|ui| {
            ui.label("max_shadow_distance");
            ui.add(egui::DragValue::new(&mut self.max_shadow_distance).speed(0.5).clamp_range(0.0..=1000.0));
        });

        ui.horizontal(|ui| {
            ui.label("lambda");
            ui.add(egui::Slider::new(&mut self.cascade_split_lambda, 0.0..=1.0));
        });
    }

    fn to_gpu_data(&self) -> GpuShadowSettings {
        GpuShadowSettings {
            blocker_search_radius: self.blocker_search_radius,
            normal_bias_scale: self.depth_bias_normal_scale,
            oriented_bias: -self.depth_bias_oriented,
            _padding: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShadowDebugSettings {
    pub frustum_culling: bool,
    pub receiver_mask_culling: bool,
    pub receiver_mask_frag_mode: bool,

    pub selected_shadow: Option<usize>,
    pub selected_cascade: usize,
    pub show_mask: bool,
    pub show_cascade_view_frustum: bool,
    pub show_cascade_light_frustum: bool,
    pub show_cascade_light_frustum_planes: bool,
    pub show_cascade_screen_space_aabb: bool,
}

impl Default for ShadowDebugSettings {
    fn default() -> Self {
        Self {
            frustum_culling: true,
            receiver_mask_culling: true,
            receiver_mask_frag_mode: false,

            selected_shadow: None,
            selected_cascade: 0,
            show_mask: false,
            show_cascade_view_frustum: false,
            show_cascade_light_frustum: false,
            show_cascade_light_frustum_planes: false,
            show_cascade_screen_space_aabb: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuShadowSettings {
    blocker_search_radius: f32,
    normal_bias_scale: f32,
    oriented_bias: f32,
    _padding: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum ShadowKind {
    Directional { orientation: Quat },
}

#[derive(Debug, Clone)]
pub struct ShadowCommand {
    pub name: Cow<'static, str>,
    pub kind: ShadowKind,
}

#[derive(Debug, Clone, Copy)]
pub enum RenderedShadow {
    Cascaded {
        shadow_maps: [graphics::GraphImageHandle; MAX_SHADOW_CASCADE_COUNT],
        shadow_mask: Option<[graphics::GraphImageHandle; MAX_SHADOW_CASCADE_COUNT]>,
    },
}

impl RenderedShadow {
    pub fn shadow_maps(&self) -> &[graphics::GraphImageHandle] {
        match self {
            RenderedShadow::Cascaded{ shadow_maps, ..} => shadow_maps,
        }
    }
}

pub struct ShadowRenderer {
    pub settings: ShadowSettings,
    pub debug_settings: ShadowDebugSettings,
    pub shadow_data_buffer: graphics::Buffer,
    shadow_commands: Vec<ShadowCommand>,
    pub rendered_shadows: Vec<RenderedShadow>,
}

impl ShadowRenderer {
    pub const MAX_SHADOW_COMMANDS: usize = 256;

    pub fn new(context: &graphics::Context, settings: ShadowSettings) -> Self {
        let shadow_data_buffer = context.create_buffer(
            "shadow_data_buffer",
            &graphics::BufferDesc {
                size: Self::MAX_SHADOW_COMMANDS * FRAME_COUNT * std::mem::size_of::<GpuCascadedShadowData>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: gpu_allocator::MemoryLocation::GpuOnly,
            },
        );

        Self {
            settings,
            debug_settings: ShadowDebugSettings::default(),
            shadow_data_buffer,
            shadow_commands: Vec::new(),
            rendered_shadows: Vec::new(),
        }
    }

    pub fn update_settings(&mut self, new_settings: &ShadowSettings) {
        self.settings = new_settings.clone();
    }

    pub fn edit_shadow_debug_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.debug_settings.frustum_culling, "frustum culling");
        ui.checkbox(&mut self.debug_settings.receiver_mask_culling, "receiver mask culling");
        ui.checkbox(
            &mut self.debug_settings.receiver_mask_frag_mode,
            "per fragment receiver mask generation",
        );

        ui.heading("Selected shadow");

        if let Some(shadow_index) = self.debug_settings.selected_shadow {
            match &self.rendered_shadows[shadow_index] {
                RenderedShadow::Cascaded {
                    shadow_maps,
                    shadow_mask: shadow_masks,
                } => {
                    ui.checkbox(
                        &mut self.debug_settings.show_cascade_view_frustum,
                        "show cascade view frustum",
                    );
                    ui.checkbox(
                        &mut self.debug_settings.show_cascade_light_frustum,
                        "show cascade light frustum",
                    );
                    ui.checkbox(
                        &mut self.debug_settings.show_cascade_light_frustum_planes,
                        "show cascade light frustum planes",
                    );
                    ui.checkbox(
                        &mut self.debug_settings.show_cascade_screen_space_aabb,
                        "show cascade screen space aabb",
                    );

                    ui.horizontal(|ui| {
                        ui.label("selected_cascade");
                        ui.add(egui::Slider::new(
                            &mut self.debug_settings.selected_cascade,
                            0..=MAX_SHADOW_CASCADE_COUNT - 1,
                        ));
                    });

                    ui.checkbox(&mut self.debug_settings.show_mask, "show mask");

                    let image = if let Some(shadow_masks) =
                        self.debug_settings.show_mask.then_some(shadow_masks.as_ref()).flatten()
                    {
                        shadow_masks[self.debug_settings.selected_cascade]
                    } else {
                        shadow_maps[self.debug_settings.selected_cascade]
                    };
                    ui.image(egui::ImageSource::Texture(egui::load::SizedTexture {
                        id: image.into(),
                        size: egui::Vec2::new(250.0, 250.0),
                    }));
                }
            }
        } else {
            ui.label("no shadow selected");
        }
    }

    pub fn gpu_shadow_settings(&self, context: &mut graphics::Context) -> graphics::GraphBufferHandle {
        let data = self.settings.to_gpu_data();
        context.transient_storage_data("shadow_settings_buffer", bytemuck::bytes_of(&data))
    }

    pub fn clear_shadow_commands(&mut self) {
        self.shadow_commands.clear();
    }

    pub fn add_shadow(&mut self, shadow_command: ShadowCommand) -> usize {
        let index = self.shadow_commands.len();
        self.shadow_commands.push(shadow_command);
        index
    }

    pub fn render_shadows(
        &mut self,
        context: &mut graphics::Context,

        camera: &Camera,
        camera_visibility_buffer: graphics::GraphBufferHandle,
        camera_depth_buffer: graphics::GraphImageHandle,

        assets: &GpuAssets,
        scene: &SceneData,
        debug_renderer: &mut DebugRenderer,
    ) {
        let mut shadow_datas = Vec::with_capacity(self.shadow_commands.len());
        let mut shadow_commands = Vec::new();

        self.rendered_shadows.clear();

        std::mem::swap(&mut shadow_commands, &mut self.shadow_commands);

        for (shadow_index, shadow_command) in shadow_commands.iter().enumerate() {
            let mut shadow_data = GpuCascadedShadowData::zeroed();
            match shadow_command.kind {
                ShadowKind::Directional { orientation } => {
                    self.render_cascaded_shadow(
                        context,
                        &mut shadow_data,
                        shadow_command.name.clone(),
                        shadow_index,
                        orientation,
                        camera,
                        camera_visibility_buffer,
                        camera_depth_buffer,
                        assets,
                        scene,
                        debug_renderer,
                    );
                }
            };

            shadow_datas.push(shadow_data);
        }

        std::mem::swap(&mut shadow_commands, &mut self.shadow_commands);

        context.queue_write_buffer(
            &self.shadow_data_buffer,
            Self::MAX_SHADOW_COMMANDS * context.frame_index() * std::mem::size_of::<GpuCascadedShadowData>(),
            bytemuck::cast_slice(shadow_datas.as_slice()),
        );
    }

    pub fn rendered_shadow_maps(&self) -> impl Iterator<Item = graphics::GraphImageHandle> + '_ {
        self.rendered_shadows.iter().map(|r| r.shadow_maps().iter().copied()).flatten()
    }

    pub fn render_shadow_map(
        &mut self,
        context: &mut graphics::Context,
        pass_name: Cow<'static, str>,

        shadow_map: graphics::GraphImageHandle,
        view_projection: Mat4,
        draw_commands: graphics::GraphBufferHandle,
        clear: bool,
        fragment_shader: bool,

        assets: AssetGraphData,
        scene: SceneGraphData,
    ) {
        let settings = self.settings;

        let mut pipeline_desc = graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/shadow.vert.spv"))
            .rasterizer(graphics::RasterizerDesc {
                depth_clamp: true,
                ..Default::default()
            })
            .depth_bias_dynamic()
            .depth_state(Some(graphics::DepthState {
                format: vk::Format::D16_UNORM,
                test: graphics::PipelineState::Static(true),
                write: true,
                compare: vk::CompareOp::GREATER,
            }));

        if fragment_shader {
            pipeline_desc = pipeline_desc.fragment_shader(graphics::ShaderSource::spv("shaders/shadow.frag.spv"));
        }

        let pipeline = context.create_raster_pipeline(
            if fragment_shader {
                "shadow_pipeline"
            } else {
                "shadow_opaque_only_pipeline"
            },
            &pipeline_desc,
        );

        context
            .add_pass(pass_name)
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
                    .load_op(if clear {
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    })
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(shadow_map.full_rect())
                    .layer_count(1)
                    .depth_attachment(&depth_attachemnt);

                cmd.begin_rendering(&rendering_info);

                cmd.bind_raster_pipeline(pipeline);
                cmd.bind_index_buffer(&index_buffer, 0);

                // negative becouse of reverse-z projection
                cmd.set_depth_bias(
                    -settings.depth_bias_constant_factor,
                    0.0,
                    -settings.depth_bias_slope_factor,
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
                    std::mem::size_of::<GpuMeshDrawCommand>() as u32,
                );

                cmd.end_rendering();
            });
    }

    pub fn render_cascaded_shadow(
        &mut self,
        context: &mut graphics::Context,
        shadow_data: &mut GpuCascadedShadowData,
        name: Cow<'static, str>,
        shadow_index: usize,

        direction: Quat,

        camera: &Camera,
        camera_visibility_buffer: graphics::GraphBufferHandle,
        camera_depth_buffer: graphics::GraphImageHandle,

        assets: &GpuAssets,
        scene: &SceneData,

        debug_renderer: &mut DebugRenderer,
    ) {
        let imported_assets = assets.import_to_graph(context);
        let imported_scene = scene.import_to_graph(context);

        let inv_light_direction = direction.inverse();

        *shadow_data = GpuCascadedShadowData {
            light_projection_matrix: bytemuck::Zeroable::zeroed(),
            shadow_map_indices: bytemuck::Zeroable::zeroed(),
            shadow_map_world_sizes: bytemuck::Zeroable::zeroed(),
        };

        let frustum_culling = self.debug_settings.frustum_culling;
        let receiver_mask_culling = self.debug_settings.receiver_mask_culling;

        let mut shadow_maps = [graphics::GraphHandle::uninit(); MAX_SHADOW_CASCADE_COUNT];
        let mut shadow_masks = [graphics::GraphHandle::uninit(); MAX_SHADOW_CASCADE_COUNT];

        let lambda = self.settings.cascade_split_lambda;

        for cascade_index in 0..MAX_SHADOW_CASCADE_COUNT {
            let Projection::Perspective { fov, near_clip } = camera.projection else {
                todo!()
            };
            let far_clip = self.settings.max_shadow_distance;

            let near_split_ratio = cascade_index as f32 / MAX_SHADOW_CASCADE_COUNT as f32;
            let far_split_ratio = (cascade_index + 1) as f32 / MAX_SHADOW_CASCADE_COUNT as f32;

            let near = math::frustum_split(near_clip, far_clip, lambda, near_split_ratio);
            let far = math::frustum_split(near_clip, far_clip, lambda, far_split_ratio);

            let light_matrix = Mat4::from_quat(inv_light_direction);
            let view_to_world_matrix = camera.transform.compute_matrix();
            let view_to_light_matrix = light_matrix * view_to_world_matrix;

            let subfrustum_corners_light_space =
                math::perspective_corners(fov, camera.aspect_ratio, near, far).map(|corner| {
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
                radius_sqr = f32::max(
                    radius_sqr,
                    corner.distance_squared(subfrustum_center_light_space.into()),
                );
            }
            let radius = radius_sqr.sqrt();
            // cascaded_shadow_data.cascade_world_sizes[cascade_index] = radius * 2.0;
            shadow_data.shadow_map_world_sizes[cascade_index] = radius * 2.0;

            // offsets the whole projection to minimize the area not visible by the camera
            // this one isn't perfect but close enough
            let forward_offset = {
                let min_coords = min_coords_light_space - Vec3A::from(subfrustum_center_light_space);
                let max_coords = max_coords_light_space - Vec3A::from(subfrustum_center_light_space);

                let forward_sign = Vec3A::from(view_to_light_matrix.z_axis);
                let forward_a = (forward_sign + 1.0) / 2.0;

                let offset =
                    math::lerp_element_wise(min_coords.extend(1.0), max_coords.extend(1.0), forward_a.extend(1.0));
                let offset = Vec3A::from(offset) - Vec3A::splat(radius) * forward_sign;

                offset
            };

            let texel_size_vs = radius * 2.0 / self.settings.shadow_resolution as f32;

            // modifications:
            //  - aligned to view space texel sizes
            //  - offset in the direction of the camera
            let subfrustum_center_modified_light_space =
                ((Vec3A::from(subfrustum_center_light_space) + forward_offset) / texel_size_vs).floor() * texel_size_vs;

            let max_extent = Vec3A::splat(radius);
            let min_extent = -max_extent;

            // max_extent += subfrustum_center_modified_light_space;
            // min_extent += subfrustum_center_modified_light_space;

            let light_matrix = Mat4::from_translation((-subfrustum_center_modified_light_space).into()) * light_matrix;

            let near_clip = min_extent.z - 80.0;
            let far_clip = max_extent.z;

            let projection_matrix = Mat4::orthographic_rh(
                min_extent.x,
                max_extent.x,
                min_extent.y,
                max_extent.y,
                far_clip, // reverse z
                near_clip,
            );

            let light_projection_matrix = projection_matrix * light_matrix;

            let show_debug_stuff = self.debug_settings.selected_shadow == Some(shadow_index)
                && self.debug_settings.selected_cascade == cascade_index;

            if show_debug_stuff {
                if self.debug_settings.show_cascade_view_frustum {
                    let subfrustum_corners_w = math::perspective_corners(fov, camera.aspect_ratio, near, far)
                        .map(|v| view_to_world_matrix * v);
                    debug_renderer.draw_cube_with_corners(&subfrustum_corners_w, Vec4::splat(1.0));
                }

                if self.debug_settings.show_cascade_light_frustum {
                    let cascade_frustum_corners =
                        math::frustum_corners_from_matrix(&(projection_matrix * light_matrix));
                    debug_renderer.draw_cube_with_corners(&cascade_frustum_corners, vec4(1.0, 1.0, 0.0, 1.0));
                }

                if self.debug_settings.show_cascade_screen_space_aabb {
                    let width = radius * 2.0;
                    let height = width;

                    // parameters used in shader
                    let width_recip_2x = width.recip() * 2.0;
                    let height_recip_2x = height.recip() * 2.0;

                    let projection_to_world_matrix = light_projection_matrix.inverse();

                    for entity in scene.entities.iter() {
                        let Some(model) = entity.model else {
                            continue;
                        };

                        let model_matrix = entity.transform.compute_matrix();
                        for submesh in assets.models[model].submeshes.iter() {
                            let bounding_sphere = assets.mesh_infos[submesh.mesh_handle].bounding_sphere;
                            let bounding_sphere_light_space =
                                math::transform_sphere(&(light_matrix * model_matrix), bounding_sphere);

                            let sphere_center = vec2(
                                width_recip_2x * bounding_sphere_light_space.x,
                                height_recip_2x * bounding_sphere_light_space.y,
                            );
                            let sphere_box_size =
                                Vec2::splat(bounding_sphere_light_space.w) * vec2(width_recip_2x, height_recip_2x);
                            let aabb = sphere_center.xyxy() + sphere_box_size.xyxy() * vec4(-1.0, -1.0, 1.0, 1.0);
                            let aabb = aabb.clamp(Vec4::splat(-1.0), Vec4::splat(1.0));

                            let closest_z = bounding_sphere_light_space.z + bounding_sphere_light_space.w;
                            let r = 1.0 / (far_clip - near_clip);
                            let depth = closest_z * r + (r * far_clip);

                            let corners = [
                                vec2(aabb.x, aabb.y),
                                vec2(aabb.x, aabb.w),
                                vec2(aabb.z, aabb.w),
                                vec2(aabb.z, aabb.y),
                            ]
                            .map(|c| {
                                let v = projection_to_world_matrix * vec4(c.x, c.y, depth, 1.0);
                                v / v.w
                            });
                            debug_renderer.draw_quad(&corners, vec4(1.0, 1.0, 1.0, 1.0));
                        }
                    }
                }
            }

            let light_frustum_planes = math::frustum_planes_from_matrix(&Mat4::orthographic_rh(
                min_extent.x,
                max_extent.x,
                min_extent.y,
                max_extent.y,
                near_clip, // HACK: non-reverse z, the last plane is buggy for some reason, this way only the near
                far_clip,  // plane is wrong, but we don't use that
            ))
            .map(math::normalize_plane);

            let light_to_world_matrix = light_matrix.inverse();
            let camera_view_projection_matrix = camera.compute_matrix();
            let camera_clip_to_light_matrix = camera_view_projection_matrix * light_to_world_matrix;

            let camera_frustum_planes = math::frustum_planes_from_matrix(&camera_clip_to_light_matrix)
                .into_iter()
                .map(math::normalize_plane)
                .take(5)
                .filter(|&plane| Vec3A::dot(plane.into(), Vec3A::Z) >= 0.0);

            let mut culling_planes = [Vec4::ZERO; 12];
            let mut culling_plane_count = 0;

            for (i, plane) in light_frustum_planes.into_iter().chain(camera_frustum_planes).enumerate() {
                // if i == 4 { continue; } // near light plane
                culling_planes[i] = plane;
                culling_plane_count += 1;
            }

            let culling_planes = &culling_planes[..culling_plane_count];

            if self.debug_settings.show_cascade_light_frustum_planes && show_debug_stuff {
                for (i, light_space_plane) in culling_planes.iter().copied().enumerate() {
                    let world_space_plane = math::transform_plane(&light_to_world_matrix, light_space_plane);
                    if i > 5 {
                        debug_renderer.draw_plane(world_space_plane, 2.0, vec4(0.0, 0.0, 1.0, 1.0));
                    } else {
                        debug_renderer.draw_plane(world_space_plane, 2.0, vec4(1.0, 1.0, 0.0, 1.0));
                    }
                }
            }

            let shadow_map_name = format!("{name}_{cascade_index}");
            let shadow_map = context.create_transient(
                shadow_map_name.clone(),
                graphics::ImageDesc {
                    ty: graphics::ImageType::Single2D,
                    format: vk::Format::D16_UNORM,
                    dimensions: [self.settings.shadow_resolution, self.settings.shadow_resolution, 1],
                    mip_levels: 1,
                    samples: graphics::MultisampleCount::None,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    aspect: vk::ImageAspectFlags::DEPTH,
                    subresource_desc: graphics::ImageSubresourceViewDesc::default(),
                    ..Default::default()
                },
            );

            let culling_planes = if frustum_culling { culling_planes } else { &[] };

            if receiver_mask_culling {
                // light projection to camera projection
                let reprojection_matrix = camera_view_projection_matrix * light_projection_matrix.inverse();

                shadow_masks[cascade_index] = shadow_pass_with_mask(
                    context,
                    &shadow_map_name,
                    shadow_map,
                    &self.settings,
                    Projection::Orthographic {
                        half_width: radius,
                        near_clip,
                        far_clip,
                    },
                    &light_matrix,
                    &light_projection_matrix,
                    culling_planes,
                    camera_visibility_buffer,
                    camera_depth_buffer,
                    &reprojection_matrix,
                    self.debug_settings.receiver_mask_frag_mode,
                    imported_assets,
                    imported_scene,
                );
            } else {
                let draw_commands = create_draw_commands(
                    context,
                    format!("{shadow_map_name}_shadow_draw_commands").into(),
                    imported_assets,
                    imported_scene,
                    &CullInfo {
                        view_matrix: light_matrix,
                        view_space_cull_planes: culling_planes,
                        occlusion_culling: OcclusionCullInfo::None,
                        alpha_mode_filter: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,
                        debug_print: false,
                    },
                    None,
                );

                self.render_shadow_map(
                    context,
                    format!("{shadow_map_name}_shadow_pass").into(),
                    shadow_map,
                    light_projection_matrix,
                    draw_commands,
                    true,
                    true,
                    imported_assets,
                    imported_scene,
                )
            }

            shadow_data.light_projection_matrix[cascade_index] = light_projection_matrix;
            shadow_data.shadow_map_indices[cascade_index] = context.get_resource_descriptor_index(shadow_map).unwrap();

            shadow_maps[cascade_index] = shadow_map;
        }

        self.rendered_shadows.push(RenderedShadow::Cascaded {
            shadow_maps,
            shadow_mask: receiver_mask_culling.then_some(shadow_masks),
        });
    }
}

const SHADOW_MASK_FORMAT: vk::Format = vk::Format::R8_UNORM;
const SHADOW_MASK_SIZE: u32 = 128;

fn shadow_pass_with_mask(
    context: &mut graphics::Context,

    name: &str,
    shadow_map: graphics::GraphImageHandle,
    settings: &ShadowSettings,
    projection: Projection,
    light_view_matrix: &Mat4,
    light_view_projection_matrix: &Mat4,
    culling_planes: &[Vec4],

    camera_visibility_buffer: graphics::GraphBufferHandle,
    camera_depth_buffer: graphics::GraphImageHandle,
    reprojection_matrix: &Mat4,
    use_frag_mask: bool,

    assets: AssetGraphData,
    scene: SceneGraphData,
) -> graphics::GraphImageHandle {
    let settings = settings.clone();

    let shadow_mask_size = if use_frag_mask { SHADOW_MASK_SIZE } else { settings.shadow_resolution };
    let mask_mip_levels = mip_levels_from_size(shadow_mask_size);
    let shadow_mask = context.create_transient_image(
        format!("{name}_shadow_mask"),
        graphics::ImageDesc {
            ty: graphics::ImageType::Single2D,
            format: SHADOW_MASK_FORMAT,
            dimensions: [shadow_mask_size, shadow_mask_size, 1],
            mip_levels: mask_mip_levels,
            samples: graphics::MultisampleCount::None,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            subresource_desc: graphics::ImageSubresourceViewDesc {
                mip_count: u32::MAX,
                mip_descriptors: graphics::ImageDescriptorFlags::STORAGE,
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let draw_commands = create_draw_commands(
        context,
        format!("{name}_mask_draw_commands").into(),
        assets,
        scene,
        &CullInfo {
            view_matrix: light_view_matrix.clone(),
            view_space_cull_planes: culling_planes,
            occlusion_culling: OcclusionCullInfo::VisibilityRead {
                visibility_buffer: camera_visibility_buffer,
            },
            alpha_mode_filter: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,
            debug_print: false,
        },
        None,
    );

    let view_projection = light_view_projection_matrix.clone();

    if use_frag_mask {
        shadow_mask_pass_frag(
            context,
            name,
            draw_commands,
            shadow_mask,
            shadow_map,
            camera_depth_buffer,
            &settings,
            light_view_projection_matrix,
            reprojection_matrix,
            assets,
            scene,
        );
    } else {
        shadow_mask_pass_geom(
            context,
            name,
            draw_commands,
            shadow_mask,
            shadow_map,
            &settings,
            light_view_projection_matrix,
            assets,
            scene,
        );
    }

    shadow_mask_mip_reduce(context, shadow_mask);

    let draw_commands = create_draw_commands(
        context,
        format!("{name}_shadow_draw_commands").into(),
        assets,
        scene,
        &CullInfo {
            view_matrix: light_view_matrix.clone(),
            view_space_cull_planes: culling_planes,
            occlusion_culling: OcclusionCullInfo::ShadowMask {
                visibility_buffer: camera_visibility_buffer,
                shadow_mask,
                projection,
                aspect_ratio: 1.0,
            },
            alpha_mode_filter: AlphaModeFlags::OPAQUE | AlphaModeFlags::MASKED,
            debug_print: false,
        },
        None,
    );

    let pipeline = context.create_raster_pipeline(
        "shadow_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/shadow.vert.spv"))
            .fragment_shader(graphics::ShaderSource::spv("shaders/shadow.frag.spv"))
            .rasterizer(graphics::RasterizerDesc {
                depth_clamp: true,
                ..Default::default()
            })
            .depth_bias_dynamic()
            .depth_state(Some(graphics::DepthState {
                format: vk::Format::D16_UNORM,
                test: graphics::PipelineState::Static(true),
                write: true,
                compare: vk::CompareOp::GREATER,
            })),
    );

    context
        .add_pass(format!("{name}_shadow_pass"))
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
                .load_op(vk::AttachmentLoadOp::LOAD)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                })
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::builder()
                .render_area(shadow_map.full_rect())
                .layer_count(1)
                .depth_attachment(&depth_attachemnt);

            cmd.begin_rendering(&rendering_info);

            cmd.bind_raster_pipeline(pipeline);
            cmd.bind_index_buffer(&index_buffer, 0);

            // negative becouse of reverse-z projection
            cmd.set_depth_bias(
                -settings.depth_bias_constant_factor,
                0.0,
                -settings.depth_bias_slope_factor,
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
                std::mem::size_of::<GpuMeshDrawCommand>() as u32,
            );

            cmd.end_rendering();
        });

    shadow_mask
}

fn shadow_mask_pass_geom(
    context: &mut graphics::Context,

    name: &str,
    draw_commands: graphics::GraphBufferHandle,
    shadow_mask: graphics::GraphImageHandle,
    shadow_map: graphics::GraphImageHandle,
    shadow_settings: &ShadowSettings,
    light_view_projection_matrix: &Mat4,

    assets: AssetGraphData,
    scene: SceneGraphData,
) {
    let shadow_settings = shadow_settings.clone();

    let pipeline = context.create_raster_pipeline(
        "shadow_mask_geom_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/shadow.vert.spv"))
            .fragment_shader(graphics::ShaderSource::spv("shaders/shadow_mask_geom.frag.spv"))
            .color_attachments(&[graphics::PipelineColorAttachment {
                format: SHADOW_MASK_FORMAT,
                ..Default::default()
            }])
            .rasterizer(graphics::RasterizerDesc {
                depth_clamp: true,
                ..Default::default()
            })
            .depth_bias_dynamic()
            .depth_state(Some(graphics::DepthState {
                format: vk::Format::D16_UNORM,
                test: graphics::PipelineState::Static(true),
                write: true,
                compare: vk::CompareOp::GREATER,
            })),
    );
    let view_projection = light_view_projection_matrix.clone();

    context
        .add_pass(format!("{name}_shadow_mask_pass"))
        .with_dependency(shadow_mask, graphics::AccessKind::ColorAttachmentWrite)
        .with_dependency(shadow_map, graphics::AccessKind::DepthAttachmentWrite)
        .with_dependency(draw_commands, graphics::AccessKind::IndirectBuffer)
        .render(move |cmd, graph| {
            let shadow_map = graph.get_image(shadow_map);
            let shadow_mask = graph.get_image(shadow_mask);

            let vertex_buffer = graph.get_buffer(assets.vertex_buffer);
            let index_buffer = graph.get_buffer(assets.index_buffer);
            let entity_buffer = graph.get_buffer(scene.entity_buffer);
            let draw_commands_buffer = graph.get_buffer(draw_commands);
            let materials_buffer = graph.get_buffer(assets.materials_buffer);

            let color_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(shadow_mask.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [1.0; 4] },
                })
                .store_op(vk::AttachmentStoreOp::STORE);

            let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
                .image_view(shadow_map.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                })
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::builder()
                .color_attachments(std::slice::from_ref(&color_attachment))
                .render_area(shadow_map.full_rect())
                .layer_count(1)
                .depth_attachment(&depth_attachemnt);

            cmd.begin_rendering(&rendering_info);

            cmd.bind_raster_pipeline(pipeline);
            cmd.bind_index_buffer(&index_buffer, 0);

            // negative becouse of reverse-z projection
            cmd.set_depth_bias(
                -shadow_settings.depth_bias_constant_factor,
                0.0,
                -shadow_settings.depth_bias_slope_factor,
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
                std::mem::size_of::<GpuMeshDrawCommand>() as u32,
            );

            cmd.end_rendering();
        });
}

fn shadow_mask_pass_frag(
    context: &mut graphics::Context,

    name: &str,
    draw_commands: graphics::GraphBufferHandle,
    shadow_mask: graphics::GraphImageHandle,
    shadow_map: graphics::GraphImageHandle,
    camera_depth_buffer: graphics::GraphImageHandle,
    shadow_settings: &ShadowSettings,
    light_view_projection_matrix: &Mat4,
    reprojection_matrix: &Mat4,

    assets: AssetGraphData,
    scene: SceneGraphData,
) {
    #[repr(C)]
    #[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    struct ShadowMaskFragData {
        view_projection_matrix: Mat4,
        reprojection_matrix: Mat4,
        vertex_buffer: u32,
        entity_buffer: u32,
        draw_commands_buffer: u32,
        materials_buffer: u32,
        camera_depth_buffer: u32,
        shadow_mask: u32,
        _padding: [u32; 2],
    }

    let shadow_mask_frag_data = context.transient_storage_data(
        format!("{name}_shadow_mask_draw_data"),
        bytemuck::bytes_of(&ShadowMaskFragData {
            view_projection_matrix: light_view_projection_matrix.clone(),
            reprojection_matrix: reprojection_matrix.clone(),
            vertex_buffer: context.get_resource_descriptor_index(assets.vertex_buffer).unwrap(),
            entity_buffer: context.get_resource_descriptor_index(scene.entity_buffer).unwrap(),
            draw_commands_buffer: context.get_resource_descriptor_index(draw_commands).unwrap(),
            materials_buffer: context.get_resource_descriptor_index(assets.materials_buffer).unwrap(),
            camera_depth_buffer: context.get_resource_descriptor_index(camera_depth_buffer).unwrap(),
            shadow_mask: context.get_resource_descriptor_index(shadow_mask).unwrap(),
            _padding: [0; 2],
        }),
    );

    let pipeline = context.create_raster_pipeline(
        "shadow_mask_frag_pipeline",
        &graphics::RasterPipelineDesc::builder()
            .vertex_shader(graphics::ShaderSource::spv("shaders/shadow_mask_frag.vert.spv"))
            .fragment_shader(graphics::ShaderSource::spv("shaders/shadow_mask_frag.frag.spv"))
            .rasterizer(graphics::RasterizerDesc {
                depth_clamp: true,
                ..Default::default()
            })
            .depth_bias_dynamic()
            .depth_state(Some(graphics::DepthState {
                format: vk::Format::D16_UNORM,
                test: graphics::PipelineState::Static(true),
                write: true,
                compare: vk::CompareOp::GREATER,
            })),
    );

    clear_shadow_mask_compute(context, shadow_mask);

    let shadow_settings = shadow_settings.clone();

    context
        .add_pass(format!("{name}_shadow_mask_pass"))
        .with_dependency(shadow_map, graphics::AccessKind::DepthAttachmentWrite)
        .with_dependency(shadow_mask, graphics::AccessKind::FragmentShaderWrite)
        .with_dependency(camera_depth_buffer, graphics::AccessKind::FragmentShaderRead)
        .with_dependency(draw_commands, graphics::AccessKind::IndirectBuffer)
        .render(move |cmd, graph| {
            let shadow_map = graph.get_image(shadow_map);
            let index_buffer = graph.get_buffer(assets.index_buffer);
            let draw_commands_buffer = graph.get_buffer(draw_commands);
            let data_buffer = graph.get_buffer(shadow_mask_frag_data);

            let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
                .image_view(shadow_map.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                })
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::builder()
                .render_area(shadow_map.full_rect())
                .layer_count(1)
                .depth_attachment(&depth_attachemnt);

            cmd.begin_rendering(&rendering_info);

            cmd.bind_raster_pipeline(pipeline);
            cmd.bind_index_buffer(&index_buffer, 0);

            // negative becouse of reverse-z projection
            cmd.set_depth_bias(
                -shadow_settings.depth_bias_constant_factor,
                0.0,
                -shadow_settings.depth_bias_slope_factor,
            );

            cmd.build_constants().buffer(data_buffer).push();

            cmd.draw_indexed_indirect_count(
                draw_commands_buffer,
                4,
                draw_commands_buffer,
                0,
                MAX_DRAW_COUNT as u32,
                std::mem::size_of::<GpuMeshDrawCommand>() as u32,
            );

            cmd.end_rendering();
        });
}

fn clear_shadow_mask_compute(context: &mut graphics::Context, mask: graphics::GraphImageHandle) {
    let clear_pipeline = context.create_compute_pipeline(
        "clear_shadow_mask_pipeline",
        graphics::ShaderSource::spv("shaders/shadow_mask_clear.comp.spv"),
    );

    context
        .add_pass("clear_shadow_mask_compute")
        .with_dependency(mask, graphics::AccessKind::ComputeShaderWrite)
        .render(move |cmd, graph| {
            let view = graph.get_image(mask).mip_view(0).unwrap();
            cmd.bind_compute_pipeline(clear_pipeline);
            cmd.build_constants().uint(view.width()).uint(view.height()).storage_image(&view);
            cmd.dispatch([view.width() / 16 + 1, view.height() / 16 + 1, 1]);
        });
}

fn shadow_mask_mip_reduce(context: &mut graphics::Context, mask: graphics::GraphImageHandle) {
    let reduce_pipeline = context.create_compute_pipeline(
        "shadow_mask_mip_reduce_pipeline",
        graphics::ShaderSource::spv("shaders/shadow_mask_mipgen.comp.spv"),
    );

    context
        .add_pass("mipgen_min_reduce")
        .with_dependency(mask, graphics::AccessKind::ComputeShaderWrite)
        .render(move |cmd, graph| {
            let mask = graph.get_image(mask);
            cmd.bind_compute_pipeline(reduce_pipeline);

            for mip_level in 0..mask.mip_view_count() - 1 {
                if mip_level != 0 {
                    cmd.barrier(
                        &[],
                        &[],
                        &[vk::MemoryBarrier2 {
                            src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            dst_access_mask: vk::AccessFlags2::SHADER_READ,
                            ..Default::default()
                        }],
                    );
                }

                let src_view = mask.mip_view(mip_level).unwrap();
                let dst_view = mask.mip_view(mip_level + 1).unwrap();

                cmd.build_constants()
                    .uint(dst_view.width())
                    .uint(dst_view.height())
                    .sampled_image(&src_view)
                    .storage_image(&dst_view);
                cmd.dispatch([dst_view.width() / 16 + 1, dst_view.height() / 16 + 1, 1]);
            }
        });
}
