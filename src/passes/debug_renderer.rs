use std::{f32::consts::PI, mem::size_of};

use ash::vk;
use glam::{vec3, Mat4, Vec3, Vec3A, Vec4};
use gpu_allocator::MemoryLocation;

use crate::{
    app::Settings,
    assets::GpuAssets,
    camera::Camera,
    collections::arena::Index,
    graphics::{
        self, ColorAttachmentDesc, DepthAttachmentDesc, DrawPass, GpuDrawIndiexedIndirectCommand, LoadOp, ResolveMode,
    },
    math, App,
};

use super::forward::TargetAttachments;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuDebugLineVertex {
    position: Vec3,
    color: [u8; 4],
}

struct MeshDrawCommand {
    transform: Mat4,
    model_index: Index,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuDebugMeshInstance {
    matrix: Mat4,
    color: Vec4,
}

#[derive(Debug, Clone, Copy)]
struct DebugMeshDrawCommand {
    instance_start: u32,
    instance_count: u32,
    mesh: Index,
}

pub struct DebugRenderer {
    line_vertices: Vec<GpuDebugLineVertex>,
    mesh_instances: Vec<GpuDebugMeshInstance>,
    mesh_draw_commands: Vec<DebugMeshDrawCommand>,
    mesh_expanded_draw_commands: Vec<GpuDrawIndiexedIndirectCommand>,

    line_vertex_buffer: graphics::Buffer,
    mesh_instance_buffer: graphics::Buffer,
    mesh_draw_commands_buffer: graphics::Buffer,
}

impl DebugRenderer {
    pub const MAX_VERTEX_COUNT: usize = 1_000_000;
    pub const MAX_MESH_INSTANCE_COUNT: usize = 1_000;
    pub const MAX_MESH_DRAW_COMMANDS: usize = 32_000;
    const CIRCLE_LINE_SEGMENTS: usize = 24;

    pub fn new(context: &mut graphics::Context) -> Self {
        let line_buffer = context.create_buffer(
            "debug_line_buffer",
            &graphics::BufferDesc {
                size: graphics::FRAME_COUNT * Self::MAX_VERTEX_COUNT * size_of::<GpuDebugLineVertex>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let mesh_instance_buffer = context.create_buffer(
            "debug_mesh_instance_buffer",
            &graphics::BufferDesc {
                size: graphics::FRAME_COUNT * Self::MAX_MESH_INSTANCE_COUNT * size_of::<GpuDebugMeshInstance>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        let mesh_draw_commands_buffer = context.create_buffer(
            "debug_mesh_draw_commands_buffer",
            &graphics::BufferDesc {
                size: graphics::FRAME_COUNT
                    * Self::MAX_MESH_DRAW_COMMANDS
                    * size_of::<GpuDrawIndiexedIndirectCommand>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                memory_location: MemoryLocation::GpuOnly,
            },
        );

        Self {
            line_vertex_buffer: line_buffer,
            mesh_draw_commands_buffer,
            mesh_instance_buffer,

            line_vertices: Vec::new(),
            mesh_instances: Vec::new(),
            mesh_draw_commands: Vec::new(),
            mesh_expanded_draw_commands: Vec::new(),
        }
    }

    pub fn draw_line(&mut self, start: Vec3, end: Vec3, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);
        self.line_vertices.extend_from_slice(&[
            GpuDebugLineVertex { position: start, color },
            GpuDebugLineVertex { position: end, color },
        ]);
    }

    pub fn draw_quad(&mut self, corners: &[Vec4; 4], color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        self.line_vertices.extend_from_slice(&[
            GpuDebugLineVertex {
                position: corners[0].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[1].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[1].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[2].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[2].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[3].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[3].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[0].truncate(),
                color,
            },
        ]);
    }

    pub fn draw_cube_with_corners(&mut self, corners: &[Vec4; 8], color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        self.line_vertices.extend_from_slice(&[
            GpuDebugLineVertex {
                position: corners[0].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[1].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[1].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[2].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[2].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[3].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[3].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[0].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[4].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[5].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[5].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[6].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[6].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[7].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[7].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[4].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[0].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[4].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[1].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[5].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[2].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[6].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[3].truncate(),
                color,
            },
            GpuDebugLineVertex {
                position: corners[7].truncate(),
                color,
            },
        ]);
    }

    pub fn draw_cross(&mut self, pos: Vec3, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        self.line_vertices.extend_from_slice(&[
            GpuDebugLineVertex {
                position: pos - vec3(1.0, 1.0, 1.0) * 0.01,
                color,
            },
            GpuDebugLineVertex {
                position: pos + vec3(1.0, 1.0, 1.0) * 0.01,
                color,
            },
            GpuDebugLineVertex {
                position: pos - vec3(-1.0, 1.0, 1.0) * 0.01,
                color,
            },
            GpuDebugLineVertex {
                position: pos + vec3(-1.0, 1.0, 1.0) * 0.01,
                color,
            },
            GpuDebugLineVertex {
                position: pos - vec3(1.0, -1.0, 1.0) * 0.01,
                color,
            },
            GpuDebugLineVertex {
                position: pos + vec3(1.0, -1.0, 1.0) * 0.01,
                color,
            },
            GpuDebugLineVertex {
                position: pos - vec3(-1.0, -1.0, 1.0) * 0.01,
                color,
            },
            GpuDebugLineVertex {
                position: pos + vec3(-1.0, -1.0, 1.0) * 0.01,
                color,
            },
        ]);
    }

    pub fn draw_sphere(&mut self, pos: Vec3, radius: f32, color: Vec4) {
        self.draw_cross(pos, color);
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        let mut circle_vertices = [GpuDebugLineVertex {
            position: Vec3::ZERO,
            color,
        }; Self::CIRCLE_LINE_SEGMENTS * 2];

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(theta.sin() * radius, theta.cos() * radius, 0.0);
        }
        self.line_vertices.extend_from_slice(&circle_vertices);

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(theta.sin() * radius, 0.0, theta.cos() * radius);
        }
        self.line_vertices.extend_from_slice(&circle_vertices);

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(0.0, theta.sin() * radius, theta.cos() * radius);
        }
        self.line_vertices.extend_from_slice(&circle_vertices);
    }

    pub fn draw_plane(&mut self, mut plane: Vec4, half_size: f32, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);
        plane = math::normalize_plane(plane);
        let normal = -Vec3A::from(plane);
        let distance = plane.w;
        let (plane_x, plane_y) = normal.any_orthonormal_pair();
        let center = normal * distance;
        self.line_vertices.extend_from_slice(&[
            GpuDebugLineVertex {
                position: (center + (plane_x + plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (plane_x + -plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (plane_x + -plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + -plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + -plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (plane_x + plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (plane_x + plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (plane_x + plane_y + normal * 0.5) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (plane_x + -plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (plane_x + -plane_y + normal * 0.5) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + -plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + -plane_y + normal * 0.5) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + plane_y) * half_size).into(),
                color,
            },
            GpuDebugLineVertex {
                position: (center + (-plane_x + plane_y + normal * 0.5) * half_size).into(),
                color,
            },
        ]);
    }

    pub fn draw_model_wireframe(&mut self, matrix: Mat4, mesh: Index, color: Vec4) {
        let instance_start = self.mesh_instances.len() as u32;
        self.mesh_instances.push(GpuDebugMeshInstance { matrix, color });
        self.mesh_draw_commands.push(DebugMeshDrawCommand {
            instance_start,
            instance_count: 1,
            mesh,
        });
    }

    pub fn render(
        &mut self,
        context: &mut graphics::Context,
        settings: &Settings,
        assets: &GpuAssets,
        target_attachments: TargetAttachments,
        camera: &Camera,
    ) {
        if self.line_vertices.is_empty() && self.mesh_draw_commands.is_empty() {
            return;
        }

        let mesh_instance_offset = Self::MAX_MESH_INSTANCE_COUNT * context.frame_index();
        let mesh_draw_commands_buffer_byte_offset =
            Self::MAX_MESH_DRAW_COMMANDS * size_of::<GpuDrawIndiexedIndirectCommand>() * context.frame_index();

        self.mesh_expanded_draw_commands.clear();
        let assets_shared = assets.shared_stuff.read();
        for draw_command in self.mesh_draw_commands.iter().copied() {
            // TODO: do the same, but with meshlets
            for submesh in assets_shared.mesh_infos[draw_command.mesh].submesh_infos.iter() {
                self.mesh_expanded_draw_commands.push(GpuDrawIndiexedIndirectCommand {
                    index_count: submesh.index_count,
                    instance_count: draw_command.instance_count,
                    first_index: submesh.index_offset,
                    vertex_offset: submesh.vertex_offset as i32,
                    first_instance: mesh_instance_offset as u32 + draw_command.instance_start,
                    ..Default::default()
                })
            }
        }
        drop(assets_shared);

        context.queue_write_buffer(
            &self.line_vertex_buffer,
            Self::MAX_VERTEX_COUNT * size_of::<GpuDebugLineVertex>() * context.frame_index(),
            bytemuck::cast_slice(&self.line_vertices),
        );
        context.queue_write_buffer(
            &self.mesh_instance_buffer,
            Self::MAX_MESH_INSTANCE_COUNT * size_of::<GpuDebugMeshInstance>() * context.frame_index(),
            bytemuck::cast_slice(&self.mesh_instances),
        );
        context.queue_write_buffer(
            &self.mesh_draw_commands_buffer,
            Self::MAX_MESH_DRAW_COMMANDS * size_of::<GpuDrawIndiexedIndirectCommand>() * context.frame_index(),
            bytemuck::cast_slice(&self.mesh_expanded_draw_commands),
        );
        context.submit_pending();

        let line_vertex_buffer = context.import(&self.line_vertex_buffer);
        let line_vertex_offset = context.frame_index() * Self::MAX_VERTEX_COUNT;
        let line_vertex_count = self.line_vertices.len();

        let mesh_instance_buffer = context.import(&self.mesh_instance_buffer);
        let mesh_draw_commands_buffer = context.import(&self.mesh_draw_commands_buffer);
        let mesh_draw_commands_count = self.mesh_expanded_draw_commands.len();

        let line_pipeline = context.create_raster_pipeline(
            "debug_line_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/debug/debug_line.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv("shaders/debug/debug_color.frag.spv"))
                .rasterizer(graphics::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::LINE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_clamp: false,
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: App::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }])
                .depth_state(Some(graphics::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: graphics::PipelineState::Dynamic,
                    write: false,
                    compare: vk::CompareOp::GREATER_OR_EQUAL,
                }))
                .multisample_state(graphics::MultisampleState {
                    sample_count: settings.msaa,
                    alpha_to_coverage: false,
                }),
        );

        let mesh_wireframe_pipeline = context.create_raster_pipeline(
            "debug_mesh_wireframe_pipeline",
            &graphics::RasterPipelineDesc::builder()
                .vertex_shader(graphics::ShaderSource::spv("shaders/debug/debug_mesh.vert.spv"))
                .fragment_shader(graphics::ShaderSource::spv("shaders/debug/debug_color.frag.spv"))
                .rasterizer(graphics::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    polygon_mode: vk::PolygonMode::LINE,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_clamp: false,
                })
                .color_attachments(&[graphics::PipelineColorAttachment {
                    format: App::COLOR_FORMAT,
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }])
                .depth_state(Some(graphics::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: graphics::PipelineState::Dynamic,
                    write: false,
                    compare: vk::CompareOp::GREATER_OR_EQUAL,
                }))
                .multisample_state(graphics::MultisampleState {
                    sample_count: settings.msaa,
                    alpha_to_coverage: false,
                }),
        );

        let view_projection_matrix = camera.compute_matrix();

        let assets = assets.import_to_graph(context);

        let mut render_pass = graphics::RenderPass::new(context, "debug_render")
            .color_attachments(&[ColorAttachmentDesc {
                target: target_attachments.color_target,
                resolve: target_attachments.color_resolve.map(|i| (i, ResolveMode::Average)),
                load_op: LoadOp::Load,
                store: true,
            }])
            .depth_attachment(DepthAttachmentDesc {
                target: target_attachments.depth_target,
                resolve: None,
                load_op: LoadOp::Load,
                store: false,
            });

        if line_vertex_count > 0 {
            DrawPass::new(&mut render_pass, line_pipeline)
                .with_depth_test(false)
                .push_data_ref(&view_projection_matrix)
                .push_data(0.1f32)
                .read_buffer(line_vertex_buffer)
                .push_data(line_vertex_offset as u32)
                .draw(0..line_vertex_count as u32, 0..1);

            DrawPass::new(&mut render_pass, line_pipeline)
                .with_depth_test(true)
                .push_data_ref(&view_projection_matrix)
                .push_data(1.0f32)
                .read_buffer(line_vertex_buffer)
                .push_data(line_vertex_offset as u32)
                .draw(0..line_vertex_count as u32, 0..1);
        }

        if mesh_draw_commands_count > 0 {
            DrawPass::new(&mut render_pass, mesh_wireframe_pipeline)
                .with_depth_test(true)
                .with_index_buffer(assets.index_buffer, 0, vk::IndexType::UINT32)
                .push_data_ref(&view_projection_matrix)
                .read_buffer(assets.vertex_buffer)
                .read_buffer(mesh_instance_buffer)
                .push_data(1.0f32)
                .draw_command(graphics::DrawCommand::DrawIndexedIndirect {
                    draw_buffer: mesh_draw_commands_buffer,
                    draw_buffer_offset: mesh_draw_commands_buffer_byte_offset as u64,
                    draw_count: mesh_draw_commands_count as u32,
                    stride: size_of::<GpuDrawIndiexedIndirectCommand>() as u32,
                });
        }

        render_pass.finish();

        self.line_vertices.clear();
        self.mesh_instances.clear();
        self.mesh_draw_commands.clear();
    }
}
