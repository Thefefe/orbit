use crate::graphics;

use std::ffi::CStr;
use ash::vk;

#[derive(Debug, Clone, Copy)]
pub struct RasterPipeline {
    pub handle: vk::Pipeline,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderStage<'a> {
    pub module: vk::ShaderModule,
    pub entry: &'a CStr,
}

impl<'a> ShaderStage<'a> {
    fn to_vk(&self) -> vk::PipelineShaderStageCreateInfoBuilder<'a> {
        vk::PipelineShaderStageCreateInfo::builder().module(self.module).name(self.entry)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VertexInput<'a> {
    pub bindings: &'a [vk::VertexInputBindingDescription],
    pub attributes: &'a [vk::VertexInputAttributeDescription],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DepthBias {
    pub constant_factor: f32,
    pub clamp: f32,
    pub slope_factor: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RasterizerDesc {
    pub primitive_topology: vk::PrimitiveTopology,
    pub polygon_mode: vk::PolygonMode,
    pub line_width: f32,
    pub front_face: vk::FrontFace,
    pub cull_mode: vk::CullModeFlags,
    pub depth_bias: Option<DepthBias>,
    pub depth_clamp: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ColorBlendState {
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineColorAttachment {
    pub format: vk::Format,
    pub color_mask: vk::ColorComponentFlags,
    pub color_blend: Option<ColorBlendState>,
}

impl Default for PipelineColorAttachment {
    fn default() -> Self {
        Self {
            format: Default::default(),
            color_mask: vk::ColorComponentFlags::RGBA,
            color_blend: Default::default()
        }
    }
}

impl PipelineColorAttachment {
    fn color_blend_vk(&self) -> vk::PipelineColorBlendAttachmentState {
        if let Some(color_blend) = &self.color_blend {
            vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(color_blend.src_color_blend_factor)
                .dst_color_blend_factor(color_blend.dst_color_blend_factor)
                .color_blend_op(color_blend.color_blend_op)
                .src_alpha_blend_factor(color_blend.src_alpha_blend_factor)
                .dst_alpha_blend_factor(color_blend.dst_alpha_blend_factor)
                .alpha_blend_op(color_blend.alpha_blend_op)
                .color_write_mask(self.color_mask)
                .build()
        } else {
            vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(false)
                .color_write_mask(self.color_mask)
                .build()
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DepthState {
    pub format: vk::Format,
    pub test: bool,
    pub write: bool,
    pub compare: vk::CompareOp,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MultisampleCount {
    #[default] None,
    X2,
    X4,
    X8,
}

impl MultisampleCount {
    pub fn to_vk(self) -> vk::SampleCountFlags {
        match self {
            MultisampleCount::None => vk::SampleCountFlags::TYPE_1,
            MultisampleCount::X2 => vk::SampleCountFlags::TYPE_2,
            MultisampleCount::X4 => vk::SampleCountFlags::TYPE_4,
            MultisampleCount::X8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MultisampleState {
    pub sample_count: MultisampleCount,
    pub alpha_to_coverage: bool,
}

impl MultisampleState {
    pub fn to_vk(self) -> vk::PipelineMultisampleStateCreateInfo {
        vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(self.sample_count.to_vk())
            .alpha_to_coverage_enable(self.alpha_to_coverage)
            .build()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RasterPipelineDesc<'a> {
    pub vertex_stage: ShaderStage<'a>,
    pub fragment_stage: Option<ShaderStage<'a>>,
    pub vertex_input: VertexInput<'a>,
    pub rasterizer: RasterizerDesc,
    pub color_attachments: &'a [PipelineColorAttachment],
    pub depth_state: Option<DepthState>,
    pub multisample_state: MultisampleState,
    pub dynamic_states: &'a [vk::DynamicState],
}

impl RasterPipeline {
    pub fn create_impl(
        device: &graphics::Device,
        bindless_layout: vk::PipelineLayout,
        name: &str,
        desc: &RasterPipelineDesc,
    ) -> RasterPipeline {
        let mut stages = vec![
            desc.vertex_stage.to_vk().stage(vk::ShaderStageFlags::VERTEX).build(),
        ];

        if let Some(fragment_stage) = desc.fragment_stage {
            stages.push(fragment_stage.to_vk().stage(vk::ShaderStageFlags::FRAGMENT).build());
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(desc.vertex_input.bindings)
            .vertex_attribute_descriptions(desc.vertex_input.attributes);

        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::builder().topology(desc.rasterizer.primitive_topology);

        let mut rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(desc.rasterizer.polygon_mode)
            .line_width(desc.rasterizer.line_width)
            .cull_mode(desc.rasterizer.cull_mode)
            .front_face(desc.rasterizer.front_face)
            .depth_bias_enable(desc.rasterizer.depth_bias.is_some())
            .depth_clamp_enable(desc.rasterizer.depth_clamp);

        if let Some(depth_bias) = &desc.rasterizer.depth_bias {
            rasterization = rasterization
                .depth_bias_constant_factor(depth_bias.constant_factor)
                .depth_bias_clamp(depth_bias.clamp)
                .depth_bias_slope_factor(depth_bias.slope_factor);
        }

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let multisample_state = desc.multisample_state.to_vk();

        let color_blend_attachment: Vec<_> =
            desc.color_attachments.iter().map(|attachment| attachment.color_blend_vk()).collect();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment);

        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        dynamic_states.extend_from_slice(desc.dynamic_states);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(desc.depth_state.as_ref().map_or(false, |depth| depth.test))
            .depth_write_enable(desc.depth_state.as_ref().map_or(false, |depth| depth.write))
            .depth_compare_op(desc.depth_state.as_ref().map_or(vk::CompareOp::ALWAYS, |depth| depth.compare))
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_attachment_formats: Vec<_> =
            desc.color_attachments.iter().map(|attachment| attachment.format).collect();

        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(
                desc.depth_state.as_ref().map_or(vk::Format::UNDEFINED, |depth| depth.format),
            );

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            // .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly)
            .rasterization_state(&rasterization)
            .viewport_state(&viewport_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&dynamic_state)
            .layout(bindless_layout)
            .push_next(&mut rendering_info);

        let handle = unsafe {
            device.raw
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_create_info),
                    None,
                )
                .unwrap()[0]
        };

        device.set_debug_name(handle, name);

        RasterPipeline { handle }
    }
}

pub fn create_shader_module(device: &graphics::Device, spv: &[u32], name: &str) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo::builder().code(spv);
    let handle = unsafe { device.raw.create_shader_module(&create_info, None).unwrap() };
    device.set_debug_name(handle, name);
    handle
}

#[derive(Debug, Clone, Copy)]
pub struct ComputePipeline {
    pub handle: vk::Pipeline,
}

impl ComputePipeline {
    pub fn create_impl(
        device: &graphics::Device,
        bindless_layout: vk::PipelineLayout,
        name: &str,
        shader: &ShaderStage,
    ) -> ComputePipeline {
        let stage = shader.to_vk().stage(vk::ShaderStageFlags::COMPUTE).build();
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage)
            .layout(bindless_layout);

        let handle = unsafe {
            device.raw.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&create_info), None)
                .unwrap()[0]
        };

        device.set_debug_name(handle, name);

        Self { handle }
    }
}

pub trait Pipeline {
    fn handle(&self) -> vk::Pipeline;
}

impl Pipeline for RasterPipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

impl Pipeline for ComputePipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

impl graphics::Context {
    pub fn create_raster_pipeline(&self, name: &str, desc: &RasterPipelineDesc) -> RasterPipeline {
        RasterPipeline::create_impl(&self.device, self.descriptors.layout(), name, desc)
    }

    pub fn create_compute_pipeline(&self, name: &str, shader: &ShaderStage) -> ComputePipeline {
        ComputePipeline::create_impl(&self.device, self.descriptors.layout(), name, shader)
    }

    pub fn destroy_pipeline(&self, pipeline: &impl Pipeline) {
        unsafe {
            self.device.raw.destroy_pipeline(pipeline.handle(), None)
        }
    }

    pub fn create_shader_module(&self, spv: &[u32], name: &str) -> vk::ShaderModule {
        create_shader_module(&self.device, spv, name)
    }

    pub fn destroy_shader_module(&self, module: vk::ShaderModule) {
        unsafe { self.device.raw.destroy_shader_module(module, None) };
    }
}