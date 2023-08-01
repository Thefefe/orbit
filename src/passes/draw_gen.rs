use std::borrow::Cow;

use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;

use crate::{render, utils, assets::AssetGraphData, scene::{SceneGraphData, GpuDrawCommand}, MAX_DRAW_COUNT};

pub struct SceneDrawGen {
    pipeline: render::ComputePipeline,
}

impl SceneDrawGen {
    pub fn new(context: &render::Context) -> Self {
        let spv = utils::load_spv("shaders/scene_draw_gen.comp.spv").unwrap();
        let module = context.create_shader_module(&spv, "scene_draw_gen_module"); 

        let pipeline = context.create_compute_pipeline("scene_draw_gen_pipeline", &render::ShaderStage {
            module,
            entry: cstr::cstr!("main"),
        });

        context.destroy_shader_module(module);

        Self { pipeline }

    }

    pub fn create_draw_commands(
        &self,
        context: &mut render::Context,
        draw_commands_name: Cow<'static, str>,
        view_projection_matrix: &Mat4,
        assets: AssetGraphData,
        scene: SceneGraphData,
    ) -> render::GraphBufferHandle {
        let pass_name = format!("{draw_commands_name}_generation");
        let draw_commands = context.create_transient_buffer(draw_commands_name, render::BufferDesc {
            size: MAX_DRAW_COUNT * std::mem::size_of::<GpuDrawCommand>(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            memory_location: MemoryLocation::GpuOnly,
        });
        let pipeline = self.pipeline;

        let view_projection_matrix = view_projection_matrix.clone();

        context.add_pass(pass_name)
            // .with_dependency(scene_submeshes, AccessKind::ComputeShaderRead)
            // .with_dependency(mesh_infos, AccessKind::ComputeShaderRead)
            .with_dependency(draw_commands, render::AccessKind::ComputeShaderWrite)
            .render(move |cmd, graph| {
                let mesh_infos = graph.get_buffer(assets.mesh_info_buffer);
                let scene_submeshes = graph.get_buffer(scene.submesh_buffer);
                let entity_buffer = graph.get_buffer(scene.entity_buffer);
                let draw_commands = graph.get_buffer(draw_commands);
                
                cmd.bind_compute_pipeline(pipeline);

                cmd.build_constants()
                    .mat4(&view_projection_matrix)
                    .buffer(&scene_submeshes)
                    .buffer(&mesh_infos)
                    .buffer(&draw_commands)
                    .buffer(&entity_buffer);

                cmd.dispatch([scene.submesh_count as u32 / 256 + 1, 1, 1]);
            });
        
        draw_commands
    }

    pub fn destroy(&self, context: &render::Context) {
        context.destroy_pipeline(&self.pipeline);
    }
}