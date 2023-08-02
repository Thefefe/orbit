use std::f32::consts::PI;

use ash::vk;
use glam::{Vec3A, Vec4, vec3, Vec3, Mat4};
use gpu_allocator::MemoryLocation;

use crate::{render, App, utils};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct DebugLineVertex {
    position: Vec3,
    color: [u8; 4],
}

pub struct DebugLineRenderer {
    pipeline: render::RasterPipeline,
    line_buffer: render::Buffer,
    vertex_cursor: usize,
    frame_index: usize,
}

impl DebugLineRenderer {
    pub const MAX_VERTEX_COUNT: usize = 1_000_000;
    const CIRCLE_LINE_SEGMENTS: usize = 24;

    pub fn new(context: &render::Context) -> Self {
        let pipeline = {
            let vertex_shader = utils::load_spv("shaders/debug_line.vert.spv").unwrap();
            let fragment_shader = utils::load_spv("shaders/debug_line.frag.spv").unwrap();

            let vertex_module = context.create_shader_module(&vertex_shader, "debug_line_vertex_shader");
            let fragment_module = context.create_shader_module(&fragment_shader, "debug_line_fragment_shader");

            let entry = cstr::cstr!("main");

            let pipeline = context.create_raster_pipeline("basic_pipeline", &render::RasterPipelineDesc {
                vertex_stage: render::ShaderStage {
                    module: vertex_module,
                    entry,
                },
                fragment_stage: Some(render::ShaderStage {
                    module: fragment_module,
                    entry,
                }),
                vertex_input: render::VertexInput {
                    bindings: &[vk::VertexInputBindingDescription {
                        binding: 0,
                        stride: std::mem::size_of::<DebugLineVertex>() as u32,
                        input_rate: vk::VertexInputRate::VERTEX,
                    }],
                    attributes: &[
                        vk::VertexInputAttributeDescription {
                            location: 0,
                            binding: 0,
                            format: vk::Format::R32G32B32_SFLOAT,
                            offset: bytemuck::offset_of!(DebugLineVertex, position) as u32,
                        },
                        vk::VertexInputAttributeDescription {
                            location: 1,
                            binding: 0,
                            format: vk::Format::R8G8B8A8_UNORM,
                            offset: bytemuck::offset_of!(DebugLineVertex, color) as u32,
                        },
                    ],
                },
                rasterizer: render::RasterizerDesc {
                    primitive_topology: vk::PrimitiveTopology::LINE_LIST,
                    polygon_mode: vk::PolygonMode::FILL,
                    line_width: 1.0,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                    depth_bias: None,
                    depth_clamp: false,
                },
                color_attachments: &[render::PipelineColorAttachment {
                    format: context.swapchain.format(),
                    color_mask: vk::ColorComponentFlags::RGBA,
                    color_blend: None,
                }],
                depth_state: Some(render::DepthState {
                    format: App::DEPTH_FORMAT,
                    test: true,
                    write: false,
                    compare: vk::CompareOp::GREATER_OR_EQUAL,
                }),
                multisample: App::MULTISAMPLING,
                dynamic_states: &[vk::DynamicState::DEPTH_TEST_ENABLE]
            });

            context.destroy_shader_module(vertex_module);
            context.destroy_shader_module(fragment_module);

            pipeline
        };

        let line_buffer = context.create_buffer("debug_line_buffer", &render::BufferDesc {
            size: render::FRAME_COUNT * Self::MAX_VERTEX_COUNT * std::mem::size_of::<DebugLineVertex>(),
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_location: MemoryLocation::CpuToGpu,
        });

        Self { pipeline, line_buffer, vertex_cursor: 0, frame_index: 0 }
    }

    fn remainin_vertex_space(&self) -> usize {
        Self::MAX_VERTEX_COUNT * (self.frame_index + 1) - self.vertex_cursor
    }

    pub fn add_vertices(&mut self, vertices: &[DebugLineVertex]) {
        assert!(self.remainin_vertex_space() >= vertices.len());
        unsafe {
            let dst_ptr = self.line_buffer.mapped_ptr.unwrap()
                .as_ptr()
                .cast::<DebugLineVertex>()
                .add(Self::MAX_VERTEX_COUNT * self.frame_index + self.vertex_cursor);
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), dst_ptr, vertices.len());
        }
        self.vertex_cursor += vertices.len();
    }

    pub fn draw_line(&mut self, start: Vec3, end: Vec3, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);
        self.add_vertices(&[
            DebugLineVertex { position: start, color },
            DebugLineVertex { position: end, color },
        ]);
    }

    pub fn draw_frustum(&mut self, corners: &[Vec4; 8], color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        self.add_vertices(&[
            DebugLineVertex { position: corners[0].truncate(), color },
            DebugLineVertex { position: corners[1].truncate(), color },
            DebugLineVertex { position: corners[1].truncate(), color },
            DebugLineVertex { position: corners[2].truncate(), color },
            DebugLineVertex { position: corners[2].truncate(), color },
            DebugLineVertex { position: corners[3].truncate(), color },
            DebugLineVertex { position: corners[3].truncate(), color },
            DebugLineVertex { position: corners[0].truncate(), color },
            
            DebugLineVertex { position: corners[4].truncate(), color },
            DebugLineVertex { position: corners[5].truncate(), color },
            DebugLineVertex { position: corners[5].truncate(), color },
            DebugLineVertex { position: corners[6].truncate(), color },
            DebugLineVertex { position: corners[6].truncate(), color },
            DebugLineVertex { position: corners[7].truncate(), color },
            DebugLineVertex { position: corners[7].truncate(), color },
            DebugLineVertex { position: corners[4].truncate(), color },

            DebugLineVertex { position: corners[0].truncate(), color },
            DebugLineVertex { position: corners[4].truncate(), color },
            DebugLineVertex { position: corners[1].truncate(), color },
            DebugLineVertex { position: corners[5].truncate(), color },
            DebugLineVertex { position: corners[2].truncate(), color },
            DebugLineVertex { position: corners[6].truncate(), color },
            DebugLineVertex { position: corners[3].truncate(), color },
            DebugLineVertex { position: corners[7].truncate(), color },
        ]);
    }

    pub fn draw_cross(&mut self, pos: Vec3, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        self.add_vertices(&[
            DebugLineVertex { position: pos - vec3( 1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3( 1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos - vec3(-1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3(-1.0,  1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos - vec3( 1.0, -1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3( 1.0, -1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos - vec3(-1.0, -1.0, 1.0) * 0.01, color },
            DebugLineVertex { position: pos + vec3(-1.0, -1.0, 1.0) * 0.01, color },
        ]);
    }

    pub fn draw_sphere(&mut self, pos: Vec3, radius: f32, color: Vec4) {
        self.draw_cross(pos, color);
        let color = color.to_array().map(|f| (f * 255.0) as u8);

        let mut circle_vertices = [
            DebugLineVertex { position: Vec3::ZERO, color };
            Self::CIRCLE_LINE_SEGMENTS * 2
        ];

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(theta.sin() * radius, theta.cos() * radius, 0.0);
        }
        self.add_vertices(&circle_vertices);

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(theta.sin() * radius, 0.0, theta.cos() * radius);
        }
        self.add_vertices(&circle_vertices);

        for (vertex_index, vertex) in circle_vertices.iter_mut().enumerate() {
            let point = vertex_index / 2 + (vertex_index % 2);
            let ratio = point as f32 / Self::CIRCLE_LINE_SEGMENTS as f32;
            let theta = ratio * 2.0 * PI;
            vertex.position = pos + vec3(0.0, theta.sin() * radius, theta.cos() * radius);
        }
        self.add_vertices(&circle_vertices);
    }

    pub fn draw_plane(&mut self, mut plane: Vec4, half_size: f32, color: Vec4) {
        let color = color.to_array().map(|f| (f * 255.0) as u8);
        plane /= Vec3A::from(plane).length(); // normalize plane
        let normal = Vec3A::from(plane);
        let distance = plane.w;
        let (plane_x, plane_y) = normal.any_orthonormal_pair();
        let center = normal * distance;
        self.add_vertices(&[
            DebugLineVertex { position: (center + ( plane_x +  plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + ( plane_x + -plane_y) * half_size).into(), color },

            DebugLineVertex { position: (center + ( plane_x + -plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + (-plane_x + -plane_y) * half_size).into(), color },
            
            DebugLineVertex { position: (center + (-plane_x + -plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + (-plane_x +  plane_y) * half_size).into(), color },
            
            DebugLineVertex { position: (center + (-plane_x +  plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + ( plane_x +  plane_y) * half_size).into(), color },


            DebugLineVertex { position: (center + ( plane_x +  plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + ( plane_x +  plane_y + normal * 0.5) * half_size).into(), color },
            
            DebugLineVertex { position: (center + ( plane_x + -plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + ( plane_x + -plane_y + normal * 0.5) * half_size).into(), color },
            
            DebugLineVertex { position: (center + (-plane_x + -plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + (-plane_x + -plane_y + normal * 0.5) * half_size).into(), color },
            
            DebugLineVertex { position: (center + (-plane_x +  plane_y) * half_size).into(), color },
            DebugLineVertex { position: (center + (-plane_x +  plane_y + normal * 0.5) * half_size).into(), color },
        ]);
    }

    pub fn render(
        &mut self,
        frame_ctx: &mut render::Context,
        target_image: render::GraphImageHandle,
        resolve_image: Option<render::GraphImageHandle>,
        depth_image: render::GraphImageHandle,
        view_projection: Mat4,
    ) {
        let line_buffer = frame_ctx.import_buffer_with("debug_line_buffer", &self.line_buffer, Default::default());
        let buffer_offset = self.frame_index * Self::MAX_VERTEX_COUNT * std::mem::size_of::<DebugLineVertex>();
        let vertex_count = self.vertex_cursor as u32;
        let pipeline = self.pipeline;
        
        let mut dependencies = vec![
            (target_image, render::AccessKind::ColorAttachmentWrite),
            (depth_image, render::AccessKind::DepthAttachmentRead),
        ];

        if let Some(resolve_image) = resolve_image {
            dependencies.push((resolve_image, render::AccessKind::ColorAttachmentWrite));
        }

        frame_ctx.add_pass("debug_line_render")
            .with_dependency(target_image, render::AccessKind::ColorAttachmentWrite)
            .with_dependency(depth_image, render::AccessKind::DepthAttachmentRead)
            .with_dependencies(resolve_image.map(|h| (h, render::AccessKind::ColorAttachmentWrite)))
            .render(move |cmd, graph| {
                let target_image = graph.get_image(target_image);
                let resolve_image = resolve_image.map(|handle| graph.get_image(handle));
                let depth_image = graph.get_image(depth_image);
                let line_buffer = graph.get_buffer(line_buffer);

                let mut color_attachment = vk::RenderingAttachmentInfo::builder()
                    .image_view(target_image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);

                if let Some(resolve_image) = resolve_image {
                    color_attachment = color_attachment
                        .resolve_image_view(resolve_image.view)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE);
                }
                
                let depth_attachemnt = vk::RenderingAttachmentInfo::builder()
                    .image_view(depth_image.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::NONE);

                let rendering_info = vk::RenderingInfo::builder()
                    .render_area(target_image.full_rect())
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .depth_attachment(&depth_attachemnt);

                cmd.begin_rendering(&rendering_info);
                cmd.bind_raster_pipeline(pipeline);
                cmd.bind_vertex_buffer(0, &line_buffer, buffer_offset as u64);


                cmd.build_constants()
                    .mat4(&view_projection)
                    .float(0.1);
                cmd.set_depth_test_enable(false);
                cmd.draw(0..vertex_count as u32, 0..1);
                
                cmd.build_constants()
                    .mat4(&view_projection)
                    .float(1.0);
                cmd.set_depth_test_enable(true);
                cmd.draw(0..vertex_count as u32, 0..1);

                cmd.end_rendering();
            });

        self.frame_index = (self.frame_index + 1) % render::FRAME_COUNT;
        self.vertex_cursor = 0;
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
        context.destroy_buffer(&self.line_buffer);
    }
}