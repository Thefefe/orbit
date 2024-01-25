use std::{borrow::Cow, hash::Hash, path::Path};

use crate::{graphics, utils};

use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineState<T> {
    Static(T),
    Dynamic,
}

impl<T: Default> Default for PipelineState<T> {
    fn default() -> Self {
        PipelineState::Static(T::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShaderSource {
    Spv(Cow<'static, Path>),
}

impl ShaderSource {
    pub fn spv(p: &'static str) -> Self {
        Self::Spv(Cow::Borrowed(p.as_ref()))
    }

    pub fn name(&self) -> Cow<str> {
        match self {
            ShaderSource::Spv(spv) => spv.file_name().map(|c| c.to_string_lossy()).unwrap(),
        }
    }
}

const MAX_SPECIALISATION_ENTRY_COUNT: usize = 4;
const MAX_SPECIALISATION_DATA_SIZE: usize = MAX_SPECIALISATION_ENTRY_COUNT * 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SpecializationMapEntry {
    pub constant_id: u32,
    pub offset: u32,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpecialisationMap {
    data: [u8; MAX_SPECIALISATION_DATA_SIZE],
    entries: [SpecializationMapEntry; MAX_SPECIALISATION_ENTRY_COUNT],
    data_size: usize,
    entry_count: usize,
}

impl SpecialisationMap {
    pub fn data(&self) -> &[u8] {
        &self.data[0..self.data_size]
    }

    pub fn entries(&self) -> &[SpecializationMapEntry] {
        &self.entries[..self.entry_count]
    }

    #[inline(always)]
    #[track_caller]
    pub fn add_bytes_with_align(&mut self, constant_id: u32, bytes: &[u8], align: usize) {
        assert!(self.data_size + self.data_size % align + bytes.len() < MAX_SPECIALISATION_DATA_SIZE);
        assert!(self.entry_count < MAX_SPECIALISATION_ENTRY_COUNT);

        self.data_size += self.data_size % align;
        let offset = self.data_size;
        self.data[offset..offset + bytes.len()].clone_from_slice(bytes);
        self.data_size += bytes.len();

        self.entries[self.entry_count] = SpecializationMapEntry {
            constant_id,
            offset: offset as u32,
            size: bytes.len(),
        };
        self.entry_count += 1;
    }
}

impl PartialEq for SpecialisationMap {
    fn eq(&self, other: &Self) -> bool {
        self.entries() == other.entries() && self.data() == other.data()
    }
}

impl Eq for SpecialisationMap {}

impl Hash for SpecialisationMap {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data().hash(state);
        self.entries().hash(state)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderStage {
    pub source: ShaderSource,
    pub specialisation: SpecialisationMap,
}

impl From<ShaderSource> for ShaderStage {
    fn from(value: ShaderSource) -> Self {
        Self {
            source: value,
            specialisation: Default::default(),
        }
    }
}

impl ShaderStage {
    pub fn new(source: ShaderSource) -> Self {
        Self {
            source,
            specialisation: Default::default(),
        }
    }

    pub fn spv(p: &'static str) -> Self {
        Self {
            source: ShaderSource::Spv(Cow::Borrowed(p.as_ref())),
            specialisation: Default::default(),
        }
    }

    pub fn has_spec_constants(&self) -> bool {
        !self.specialisation.entries().is_empty()
    }

    pub fn spec_info(&self) -> vk::SpecializationInfo {
        vk::SpecializationInfo {
            map_entry_count: self.specialisation.entry_count as u32,
            p_map_entries: self.specialisation.entries.as_ptr().cast(),
            data_size: self.specialisation.data_size,
            p_data: self.specialisation.data.as_ptr().cast(),
        }
    }

    #[track_caller]
    #[inline(always)]
    pub fn spec_u32(mut self, id: u32, v: u32) -> Self {
        self.specialisation.add_bytes_with_align(id, &v.to_ne_bytes(), 4);
        self
    }

    #[track_caller]
    #[inline(always)]
    pub fn spec_bool(mut self, id: u32, v: bool) -> Self {
        self.specialisation.add_bytes_with_align(id, &(v as u32).to_ne_bytes(), 4);
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RasterPipeline {
    pub handle: vk::Pipeline,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DepthBias {
    pub constant_factor: f32,
    pub clamp: f32,
    pub slope_factor: f32,
}

// for hashing
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct OrderedDepthBias {
    pub constant_factor: ordered_float::OrderedFloat<f32>,
    pub clamp: ordered_float::OrderedFloat<f32>,
    pub slope_factor: ordered_float::OrderedFloat<f32>,
}

impl From<DepthBias> for OrderedDepthBias {
    fn from(value: DepthBias) -> Self {
        Self {
            constant_factor: value.constant_factor.into(),
            clamp: value.clamp.into(),
            slope_factor: value.slope_factor.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RasterizerDesc {
    pub primitive_topology: vk::PrimitiveTopology,
    pub polygon_mode: vk::PolygonMode,
    pub front_face: vk::FrontFace,
    pub cull_mode: vk::CullModeFlags,
    pub depth_clamp: bool,
}

impl Default for RasterizerDesc {
    fn default() -> Self {
        Self {
            primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: vk::PolygonMode::FILL,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            cull_mode: vk::CullModeFlags::BACK,
            depth_clamp: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ColorBlendState {
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            color_blend: Default::default(),
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DepthState {
    pub format: vk::Format,
    pub test: PipelineState<bool>,
    pub write: bool,
    pub compare: vk::CompareOp,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MultisampleCount {
    #[default]
    None,
    X2,
    X4,
    X8,
}

impl MultisampleCount {
    pub const ALL: [Self; 4] = [Self::None, Self::X2, Self::X4, Self::X8];

    pub fn sample_count(self) -> u32 {
        match self {
            MultisampleCount::None => 1,
            MultisampleCount::X2 => 2,
            MultisampleCount::X4 => 4,
            MultisampleCount::X8 => 8,
        }
    }

    pub fn is_some(self) -> bool {
        self != Self::None
    }

    pub fn to_vk(self) -> vk::SampleCountFlags {
        match self {
            MultisampleCount::None => vk::SampleCountFlags::TYPE_1,
            MultisampleCount::X2 => vk::SampleCountFlags::TYPE_2,
            MultisampleCount::X4 => vk::SampleCountFlags::TYPE_4,
            MultisampleCount::X8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

impl From<vk::SampleCountFlags> for MultisampleCount {
    fn from(value: vk::SampleCountFlags) -> Self {
        match value {
            vk::SampleCountFlags::TYPE_1 => MultisampleCount::None,
            vk::SampleCountFlags::TYPE_2 => MultisampleCount::X2,
            vk::SampleCountFlags::TYPE_4 => MultisampleCount::X4,
            vk::SampleCountFlags::TYPE_8 => MultisampleCount::X8,
            _ => unimplemented!(),
        }
    }
}

impl std::fmt::Display for MultisampleCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultisampleCount::None => write!(f, "off"),
            MultisampleCount::X2 => write!(f, "2x"),
            MultisampleCount::X4 => write!(f, "4x"),
            MultisampleCount::X8 => write!(f, "8x"),
        }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct MultisampleState {
    pub sample_count: MultisampleCount,
    pub alpha_to_coverage: bool,
}

impl MultisampleState {
    pub fn to_vk(self) -> vk::PipelineMultisampleStateCreateInfo {
        vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(self.sample_count.to_vk())
            // .sample_shading_enable(self.sample_count != graphics::MultisampleCount::None)
            .alpha_to_coverage_enable(self.alpha_to_coverage)
            .build()
    }
}

pub const MAX_COLOR_ATTACHMENT_COUNT: usize = 2;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct RasterPipelineDesc {
    pub task_shader: Option<ShaderStage>,
    pub mesh_shader: Option<ShaderStage>,
    pub vertex_shader: Option<ShaderStage>,
    pub fragment_shader: Option<ShaderStage>,
    pub rasterizer: RasterizerDesc,
    pub depth_bias: PipelineState<Option<OrderedDepthBias>>,
    pub color_attachment_count: usize,
    pub color_attachments: [PipelineColorAttachment; MAX_COLOR_ATTACHMENT_COUNT],
    pub depth_state: Option<DepthState>,
    pub multisample_state: MultisampleState,
}

impl RasterPipelineDesc {
    pub fn builder() -> RasterPipelineDescBuilder {
        RasterPipelineDescBuilder { desc: Self::default() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RasterPipelineDescBuilder {
    desc: RasterPipelineDesc,
}

impl std::ops::Deref for RasterPipelineDescBuilder {
    type Target = RasterPipelineDesc;

    fn deref(&self) -> &Self::Target {
        &self.desc
    }
}

impl RasterPipelineDescBuilder {
    pub fn task_shader(mut self, stage: impl Into<ShaderStage>) -> Self {
        self.desc.task_shader = Some(stage.into());
        self
    }

    pub fn mesh_shader(mut self, stage: impl Into<ShaderStage>) -> Self {
        self.desc.mesh_shader = Some(stage.into());
        self
    }

    pub fn vertex_shader(mut self, stage: impl Into<ShaderStage>) -> Self {
        self.desc.vertex_shader = Some(stage.into());
        self
    }

    pub fn fragment_shader(mut self, stage: impl Into<ShaderStage>) -> Self {
        self.desc.fragment_shader = Some(stage.into());
        self
    }

    pub fn rasterizer(mut self, rasterizer: RasterizerDesc) -> Self {
        self.desc.rasterizer = rasterizer;
        self
    }

    pub fn depth_bias_static(mut self, depth_bias: Option<DepthBias>) -> Self {
        self.desc.depth_bias = PipelineState::Static(depth_bias.map(|depth_bias| depth_bias.into()));
        self
    }

    pub fn depth_bias_dynamic(mut self) -> Self {
        self.desc.depth_bias = PipelineState::Dynamic;
        self
    }

    pub fn color_attachments(mut self, attachments: &[PipelineColorAttachment]) -> Self {
        assert!(attachments.len() <= MAX_COLOR_ATTACHMENT_COUNT);
        self.desc.color_attachment_count = attachments.len();
        unsafe {
            std::ptr::copy_nonoverlapping(
                attachments.as_ptr(),
                self.desc.color_attachments.as_mut_ptr(),
                attachments.len(),
            );
        }
        self
    }

    pub fn depth_state(mut self, depth_state: Option<DepthState>) -> Self {
        self.desc.depth_state = depth_state;
        self
    }

    pub fn multisample_state(mut self, multisample_state: MultisampleState) -> Self {
        self.desc.multisample_state = multisample_state;
        self
    }

    pub fn build(self) -> RasterPipelineDesc {
        self.desc
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

impl graphics::Context {
    pub fn create_raster_pipeline(&mut self, name: &str, desc: &RasterPipelineDesc) -> RasterPipeline {
        if let Some(pipeline) = self.raster_pipelines.get(desc).copied() {
            return pipeline;
        }

        let mut stages = vec![];
        let mut spec_infos = [vk::SpecializationInfo::default(); 4];
        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        if desc.task_shader.is_some() || desc.mesh_shader.is_some() {
            assert!(self.device.mesh_shader_fns.is_some(), "mesh shading isn't supported");
        }

        let mut add_shader_stage = |shader_stage: &ShaderStage, stage: vk::ShaderStageFlags, spci: usize| {
            let module = self.get_shader_module(&shader_stage.source);

            let mut desc = vk::PipelineShaderStageCreateInfo::builder()
                .stage(stage)
                .module(module)
                .name(cstr::cstr!("main"));

            if shader_stage.has_spec_constants() {
                spec_infos[spci] = shader_stage.spec_info();
                desc = desc.specialization_info(&spec_infos[0]);
            }

            stages.push(desc.build());
        };

        if let Some(task_shader) = &desc.task_shader {
            assert!(
                desc.mesh_shader.is_some(),
                "can't use task shaders without mesh shaders"
            );

            add_shader_stage(&task_shader, vk::ShaderStageFlags::TASK_EXT, 0);
        }

        if let Some(mesh_shader) = &desc.mesh_shader {
            add_shader_stage(&mesh_shader, vk::ShaderStageFlags::MESH_EXT, 1);
        }

        if let Some(vertex_shader) = &desc.vertex_shader {
            add_shader_stage(&vertex_shader, vk::ShaderStageFlags::VERTEX, 2);
        }

        if let Some(fragment_shader) = &desc.fragment_shader {
            add_shader_stage(&fragment_shader, vk::ShaderStageFlags::FRAGMENT, 3);
        }

        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::builder().topology(desc.rasterizer.primitive_topology);

        let mut rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(desc.rasterizer.polygon_mode)
            .line_width(1.0)
            .cull_mode(desc.rasterizer.cull_mode)
            .front_face(desc.rasterizer.front_face)
            .depth_clamp_enable(desc.rasterizer.depth_clamp);

        match &desc.depth_bias {
            PipelineState::Static(Some(depth_bias)) => {
                rasterization = rasterization
                    .depth_bias_enable(true)
                    .depth_bias_constant_factor(depth_bias.constant_factor.0)
                    .depth_bias_clamp(depth_bias.clamp.0)
                    .depth_bias_slope_factor(depth_bias.slope_factor.0)
            }
            PipelineState::Static(None) => rasterization = rasterization.depth_bias_enable(false),
            PipelineState::Dynamic => {
                rasterization = rasterization.depth_bias_enable(true);
                dynamic_states.push(vk::DynamicState::DEPTH_BIAS)
            }
        }

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewport_count(1).scissor_count(1);

        let multisample_state = desc.multisample_state.to_vk();

        let color_blend_attachment = desc.color_attachments.map(|attachment| attachment.color_blend_vk());

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment[..desc.color_attachment_count]);

        let mut depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder();

        if let Some(depth_state) = &desc.depth_state {
            match depth_state.test {
                PipelineState::Static(test) => depth_stencil_state = depth_stencil_state.depth_test_enable(test),
                PipelineState::Dynamic => {
                    depth_stencil_state = depth_stencil_state.depth_test_enable(true);
                    dynamic_states.push(vk::DynamicState::DEPTH_TEST_ENABLE);
                }
            }

            depth_stencil_state = depth_stencil_state
                .depth_write_enable(desc.depth_state.as_ref().map_or(false, |depth| depth.write))
                .depth_compare_op(desc.depth_state.as_ref().map_or(vk::CompareOp::ALWAYS, |depth| depth.compare))
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false);
        }

        let color_attachment_formats = desc.color_attachments.map(|attachment| attachment.format);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&color_attachment_formats[..desc.color_attachment_count])
            .depth_attachment_format(desc.depth_state.as_ref().map_or(vk::Format::UNDEFINED, |depth| depth.format));

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .input_assembly_state(&input_assembly)
            .vertex_input_state(&vertex_input_state)
            .rasterization_state(&rasterization)
            .viewport_state(&viewport_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&dynamic_state)
            .layout(self.device.pipeline_layout)
            .push_next(&mut rendering_info);

        let handle = unsafe {
            self.device
                .raw
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_create_info),
                    None,
                )
                .unwrap()[0]
        };

        self.device.set_debug_name(handle, name);

        self.raster_pipelines.insert(desc.clone(), RasterPipeline { handle });

        RasterPipeline { handle }
    }

    pub fn create_compute_pipeline(&mut self, name: &str, shader: ShaderStage) -> ComputePipeline {
        if let Some(compute_pipeline) = self.compute_pipelines.get(&shader).copied() {
            return compute_pipeline;
        }

        let module = self.get_shader_module(&shader.source);
        let spec_info = shader.spec_info();

        let mut stage_desc = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(cstr::cstr!("main"));

        if shader.has_spec_constants() {
            stage_desc = stage_desc.specialization_info(&spec_info);
        }

        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_desc.build())
            .layout(self.device.pipeline_layout);

        let handle = unsafe {
            self.device
                .raw
                .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&create_info), None)
                .unwrap()[0]
        };

        self.device.set_debug_name(handle, name);

        self.compute_pipelines.insert(shader, ComputePipeline { handle });

        ComputePipeline { handle }
    }

    fn get_shader_module(&mut self, source: &ShaderSource) -> vk::ShaderModule {
        if let Some(module) = self.shader_modules.get(source).copied() {
            module
        } else {
            match source {
                ShaderSource::Spv(path) => {
                    let name = path.file_name().map(|c| c.to_string_lossy()).unwrap();
                    let spv = utils::load_spv(path).expect(&format!("couldn't open file {path:?}"));
                    let create_info = vk::ShaderModuleCreateInfo::builder().code(&spv);
                    let handle = unsafe { self.device.raw.create_shader_module(&create_info, None).unwrap() };
                    self.device.set_debug_name(handle, &name);
                    self.shader_modules.insert(source.clone(), handle);
                    handle
                }
            }
        }
    }
}
