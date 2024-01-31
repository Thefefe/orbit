use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ops::Range,
};

use ash::vk;
use parking_lot::Mutex;

use crate::{
    collections::arena,
    graphics::{self, ResourceKind},
};

pub type GraphResourceIndex = usize;
pub type GraphPassIndex = arena::Index;
pub type GraphDependencyIndex = usize;

#[derive(Debug)]
pub enum ResourceSource {
    Create {
        desc: graphics::AnyResourceDesc,
        cache: Option<graphics::AnyResource>,
    },
    Import {
        resource: graphics::AnyResource,
    },
}

impl ResourceSource {
    pub fn desc(&self) -> graphics::AnyResourceDesc {
        match self {
            ResourceSource::Create { desc, .. } => *desc,
            ResourceSource::Import { resource } => resource.desc(),
        }
    }
}

#[derive(Debug)]
pub struct GraphResourceVersion {
    pub image_layout: vk::ImageLayout,
    pub source_pass: Option<GraphPassIndex>,
    pub read_by: Vec<GraphPassIndex>,
}

#[derive(Debug)]
pub struct GraphResourceData {
    pub name: Cow<'static, str>,

    pub source: ResourceSource,
    pub descriptor_index: Option<graphics::DescriptorIndex>,

    pub initial_access: graphics::AccessKind,
    pub target_access: graphics::AccessKind,
    pub initial_queue: Option<graphics::QueueType>,
    pub target_queue: Option<graphics::QueueType>,
    pub wait_semaphore: Option<graphics::Semaphore>,
    pub finish_semaphore: Option<graphics::Semaphore>,

    pub versions: Vec<GraphResourceVersion>,
}

impl GraphResourceData {
    fn kind(&self) -> ResourceKind {
        match &self.source {
            ResourceSource::Create { desc, .. } => desc.kind(),
            ResourceSource::Import { resource, .. } => resource.kind(),
        }
    }

    fn current_version(&self) -> usize {
        self.versions.len() - 1
    }

    fn curent_layout(&self) -> vk::ImageLayout {
        // self.versions.last().unwrap().last_access.image_layout()
        self.versions.last().unwrap().image_layout
    }

    // fn last_access(&self, version: usize) -> graphics::AccessKind {
    //     assert!(version < self.versions.len());
    //     self.versions[version].last_access
    // }

    fn source_pass(&self, version: usize) -> Option<GraphPassIndex> {
        self.versions[version].source_pass
    }
}

type PassFn = Box<dyn Fn(&graphics::CommandRecorder, &graphics::CompiledRenderGraph) + Send + Sync>;

#[derive(Debug, Clone, Copy)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
    DontCare,
}

impl<T> LoadOp<T> {
    pub fn vk_load_op(self) -> vk::AttachmentLoadOp {
        match self {
            LoadOp::Load => vk::AttachmentLoadOp::LOAD,
            LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
            LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

impl LoadOp<[f32; 4]> {
    pub fn clear_value(self) -> vk::ClearValue {
        match self {
            LoadOp::Clear(float32) => vk::ClearValue {
                color: vk::ClearColorValue { float32 },
            },
            _ => vk::ClearValue::default(),
        }
    }
}

impl LoadOp<[i32; 4]> {
    pub fn clear_value(self) -> vk::ClearValue {
        match self {
            LoadOp::Clear(int32) => vk::ClearValue {
                color: vk::ClearColorValue { int32 },
            },
            _ => vk::ClearValue::default(),
        }
    }
}

impl LoadOp<[u32; 4]> {
    pub fn clear_value(self) -> vk::ClearValue {
        match self {
            LoadOp::Clear(uint32) => vk::ClearValue {
                color: vk::ClearColorValue { uint32 },
            },
            _ => vk::ClearValue::default(),
        }
    }
}

impl LoadOp<f32> {
    pub fn clear_value(self) -> vk::ClearValue {
        match self {
            LoadOp::Clear(depth) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
            },
            _ => vk::ClearValue::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ResolveMode {
    SampleZero,
    Min,
    Max,
    Average,
}

impl ResolveMode {
    pub fn vk_flags(self) -> vk::ResolveModeFlags {
        match self {
            ResolveMode::SampleZero => vk::ResolveModeFlags::SAMPLE_ZERO,
            ResolveMode::Min => vk::ResolveModeFlags::MIN,
            ResolveMode::Max => vk::ResolveModeFlags::MAX,
            ResolveMode::Average => vk::ResolveModeFlags::AVERAGE,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ColorAttachmentDesc {
    pub target: graphics::GraphImageHandle,
    pub resolve: Option<(graphics::GraphImageHandle, ResolveMode)>,
    pub load_op: LoadOp<[f32; 4]>,
    pub store: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct DepthAttachmentDesc {
    pub target: graphics::GraphImageHandle,
    pub resolve: Option<(graphics::GraphImageHandle, ResolveMode)>,
    pub load_op: LoadOp<f32>,
    pub store: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Dispatch {
    DispatchWorkgroups([u32; 3]),
    DispatchIndirect(graphics::GraphBufferHandle, u64),
}

#[derive(Debug, Clone)]
pub struct RenderPassDraw {
    pub pipeline: graphics::RasterPipeline,
    pub depth_test_enable: Option<bool>,
    pub depth_bias: Option<[f32; 3]>,
    pub index_buffer: Option<(graphics::GraphBufferHandle, u64, vk::IndexType)>,
    pub constant_data: [u8; 128],
    pub constant_size: usize,
    pub draw_command: DrawCommand,
}

#[derive(Debug, Clone)]
pub enum DrawCommand {
    Draw {
        vertex_range: Range<u32>,
        instance_range: Range<u32>,
    },
    DrawIndexed {
        index_range: Range<u32>,
        instance_range: Range<u32>,
        vertex_offset: i32,
    },
    DrawIndexedIndirect {
        draw_buffer: graphics::GraphBufferHandle,
        draw_buffer_offset: u64,
        draw_count: u32,
        stride: u32,
    },
    DrawIndexedIndirectCount {
        draw_buffer: graphics::GraphBufferHandle,
        draw_buffer_offset: u64,
        count_buffer: graphics::GraphBufferHandle,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    },
    DrawMeshTasksIndirect {
        task_buffer: graphics::GraphBufferHandle,
        task_buffer_offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    },
}

impl DrawCommand {
    pub fn draw_buffer(&self) -> Option<graphics::GraphBufferHandle> {
        match self {
            DrawCommand::DrawIndexedIndirect { draw_buffer, .. } => Some(*draw_buffer),
            DrawCommand::DrawIndexedIndirectCount { draw_buffer, .. } => Some(*draw_buffer),
            DrawCommand::DrawMeshTasksIndirect { task_buffer, .. } => Some(*task_buffer),
            _ => None,
        }
    }

    pub fn count_buffer(&self) -> Option<graphics::GraphBufferHandle> {
        match self {
            DrawCommand::DrawIndexedIndirectCount { count_buffer, .. } => Some(*count_buffer),
            _ => None,
        }
    }
}

pub enum Pass {
    CustomPass(PassFn),
    ComputeDispatch {
        pipeline: graphics::ComputePipeline,
        push_constant_data: [u8; 128],
        push_constant_size: usize,
        dispatch: Dispatch,
    },
    RenderPass {
        color_attachments: [ColorAttachmentDesc; graphics::MAX_COLOR_ATTACHMENT_COUNT],
        color_attachment_count: usize,
        depth_attachment: Option<DepthAttachmentDesc>,
        render_area: [u32; 2],
        draw_range: Range<usize>,
    },
}

impl Pass {
    pub fn record(&self, cmd: &graphics::CommandRecorder, graph: &graphics::CompiledRenderGraph) {
        match self {
            Pass::CustomPass(func) => func(cmd, graph),
            Pass::ComputeDispatch {
                pipeline,
                push_constant_data,
                push_constant_size,
                dispatch,
            } => {
                cmd.bind_compute_pipeline(*pipeline);
                cmd.push_constants(&push_constant_data[0..*push_constant_size], 0);
                match dispatch {
                    Dispatch::DispatchWorkgroups(workgroup_count) => cmd.dispatch(*workgroup_count),
                    Dispatch::DispatchIndirect(indirect_buffer, offset) => {
                        cmd.dispatch_indirect(graph.get_buffer(*indirect_buffer), *offset);
                    }
                }
            }
            Pass::RenderPass {
                color_attachments,
                color_attachment_count,
                depth_attachment,
                render_area,
                draw_range,
            } => {
                let color_attachments = color_attachments.map(|a| {
                    let mut attachment = vk::RenderingAttachmentInfo::default();
                    attachment.image_view = graph.get_image(a.target).view;
                    attachment.image_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                    attachment.load_op = a.load_op.vk_load_op();
                    attachment.clear_value = a.load_op.clear_value();

                    if let Some((image, mode)) = a.resolve {
                        attachment.resolve_image_view = graph.get_image(image).view;
                        attachment.resolve_image_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                        attachment.resolve_mode = mode.vk_flags();
                    }

                    attachment
                });

                let depth_attachment = depth_attachment.map(|a| {
                    let mut attachment = vk::RenderingAttachmentInfo::default();

                    attachment.image_view = graph.get_image(a.target).view;
                    attachment.image_layout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
                    attachment.load_op = a.load_op.vk_load_op();
                    attachment.clear_value = a.load_op.clear_value();

                    if let Some((image, mode)) = a.resolve {
                        attachment.resolve_image_view = graph.get_image(image).view;
                        attachment.resolve_image_layout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
                        attachment.resolve_mode = mode.vk_flags();
                    }

                    attachment
                });

                let mut rendering_info = vk::RenderingInfo::builder()
                    .color_attachments(&color_attachments[..*color_attachment_count])
                    .layer_count(1)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: render_area[0],
                            height: render_area[1],
                        },
                    });

                if let Some(depth_attachment) = depth_attachment.as_ref() {
                    rendering_info = rendering_info.depth_attachment(depth_attachment);
                }

                cmd.begin_rendering(&rendering_info);

                let mut current_pipeline = None;
                let mut current_index_binding = None;

                for draw in &graph.draws[draw_range.clone()] {
                    if current_pipeline != Some(draw.pipeline) {
                        current_pipeline = Some(draw.pipeline);
                        cmd.bind_raster_pipeline(draw.pipeline);
                    }

                    if draw.index_buffer != current_index_binding && draw.index_buffer.is_some() {
                        current_index_binding = draw.index_buffer;
                        if let Some((buffer, offset, index_type)) = draw.index_buffer {
                            cmd.bind_index_buffer(graph.get_buffer(buffer), offset, index_type);
                        }
                    }

                    if let Some(depth_test) = draw.depth_test_enable {
                        cmd.set_depth_test_enable(depth_test);
                    }

                    if let Some([constant_factor, clamp, slope_factor]) = draw.depth_bias {
                        cmd.set_depth_bias(constant_factor, clamp, slope_factor);
                    }

                    if draw.constant_size != 0 {
                        cmd.push_constants(&draw.constant_data[..draw.constant_size], 0);
                    }

                    match &draw.draw_command {
                        DrawCommand::Draw {
                            vertex_range,
                            instance_range,
                        } => cmd.draw(vertex_range.clone(), instance_range.clone()),
                        DrawCommand::DrawIndexed {
                            index_range,
                            instance_range,
                            vertex_offset,
                        } => cmd.draw_indexed(index_range.clone(), instance_range.clone(), *vertex_offset),
                        DrawCommand::DrawIndexedIndirect {
                            draw_buffer,
                            draw_buffer_offset,
                            draw_count,
                            stride,
                        } => cmd.draw_indexed_indirect(
                            graph.get_buffer(*draw_buffer),
                            *draw_buffer_offset,
                            *draw_count,
                            *stride,
                        ),
                        DrawCommand::DrawIndexedIndirectCount {
                            draw_buffer,
                            draw_buffer_offset,
                            count_buffer,
                            count_buffer_offset,
                            max_draw_count,
                            stride,
                        } => cmd.draw_indexed_indirect_count(
                            graph.get_buffer(*draw_buffer),
                            *draw_buffer_offset,
                            graph.get_buffer(*count_buffer),
                            *count_buffer_offset,
                            *max_draw_count,
                            *stride,
                        ),
                        DrawCommand::DrawMeshTasksIndirect {
                            task_buffer,
                            task_buffer_offset,
                            draw_count,
                            stride,
                        } => cmd.draw_mesh_tasks_indirect(
                            graph.get_buffer(*task_buffer),
                            *task_buffer_offset,
                            *draw_count,
                            *stride,
                        ),
                    }
                }

                cmd.end_rendering();
            }
        }
    }
}

pub struct PassData {
    pub name: Cow<'static, str>,
    pub func: Pass,
    dependency_range: Range<usize>,
    alive: bool,
}

impl std::fmt::Debug for PassData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassData")
            .field("name", &self.name)
            .field("dependencies", &self.dependency_range)
            .field("alive", &self.alive)
            .finish()
    }
}

#[derive(Debug)]
struct DependencyData {
    access: graphics::AccessKind,
    pass_handle: GraphPassIndex,
    resource_handle: GraphResourceIndex,
    resource_version: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GraphResourceImportDesc {
    pub initial_access: graphics::AccessKind,
    pub target_access: graphics::AccessKind,
    pub initial_queue: Option<graphics::QueueType>,
    pub target_queue: Option<graphics::QueueType>,
    pub wait_semaphore: Option<graphics::Semaphore>,
    pub finish_semaphore: Option<graphics::Semaphore>,
}

#[derive(Debug)]
pub struct RenderGraph {
    pub resources: Vec<GraphResourceData>,
    pub passes: arena::Arena<PassData>,
    pub draws: Vec<RenderPassDraw>,

    dependencies: Vec<DependencyData>,

    pub import_cache: HashMap<graphics::AnyResourceHandle, GraphResourceIndex>,
    pub dont_wait_semaphores: Mutex<HashSet<vk::Semaphore>>,
    pub dont_signal_semaphores: HashSet<vk::Semaphore>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            passes: arena::Arena::new(),
            draws: Vec::new(),
            dependencies: Vec::new(),
            import_cache: HashMap::new(),
            dont_wait_semaphores: Mutex::new(HashSet::new()),
            dont_signal_semaphores: HashSet::new(),
        }
    }

    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.draws.clear();
        self.import_cache.clear();
        self.dont_wait_semaphores.get_mut().clear();
        self.dont_signal_semaphores.clear();
    }

    pub fn add_resource(&mut self, mut resource_data: GraphResourceData) -> GraphResourceIndex {
        assert!(resource_data.versions.is_empty());

        let index = self.resources.len();
        let imported_handle = if let ResourceSource::Import { resource, .. } = &resource_data.source {
            Some(resource.as_ref().handle())
        } else {
            None
        };

        if let Some(handle) = imported_handle {
            if let Some(index) = self.import_cache.get(&handle).copied() {
                return index;
            } else {
                self.import_cache.insert(handle, index);
            }
        }

        resource_data.versions = vec![GraphResourceVersion {
            image_layout: resource_data.initial_access.image_layout(),
            source_pass: None,
            read_by: vec![],
        }];

        self.resources.push(resource_data);

        index
    }

    pub fn add_pass(
        &mut self,
        name: Cow<'static, str>,
        func: Pass,
        dependencies: &[(GraphResourceIndex, graphics::AccessKind)],
    ) -> GraphPassIndex {
        let pass_index = self.passes.insert(PassData {
            name: Cow::Borrowed(""), // temp value, real value used later in the function
            func,
            dependency_range: 0..0,
            alive: false,
        });

        let dependency_start = self.dependencies.len();

        // not necessary but can catch the occasinal bug
        // TODO: make this run only as needed, maybe add a seperate validation pass
        let mut prev_deps: HashMap<GraphResourceIndex, graphics::AccessKind> = HashMap::new();

        for &(res, acc) in dependencies {
            self.add_dependency(pass_index, res, acc);

            if let Some(other_acc) = prev_deps.get(&res) {
                let res_name = self.resources[res].name.as_ref();

                assert!(
                    acc.read_only(),
                    "read and write or multiple writes in pass '{name}' for resource {res_name}"
                );

                assert!(
                    other_acc.read_only(),
                    "read and write or multiple writes in pass '{name}' for resource {res_name}"
                );

                if self.resources[res].kind() == ResourceKind::Image {
                    let layout = acc.image_layout();
                    let other_layout = other_acc.image_layout();
                    assert_eq!(
                        layout, other_layout,
                        "incompatible image layouts in pass '{name}' for image '{res_name}'"
                    );
                }
            } else {
                prev_deps.insert(res, acc);
            }
        }

        self.passes[pass_index].name = name;
        self.passes[pass_index].dependency_range = dependency_start..self.dependencies.len();

        pass_index
    }

    fn add_dependency(
        &mut self,
        pass_handle: GraphPassIndex,
        resource_handle: GraphResourceIndex,
        access: graphics::AccessKind,
    ) -> usize {
        let resource_version = self.resources[resource_handle].current_version();

        let dependency = self.dependencies.len();
        self.dependencies.push(DependencyData {
            access,
            pass_handle,
            resource_handle,
            resource_version,
        });

        let resource = &self.resources[resource_handle];
        let needs_layout_transition =
            resource.kind() == ResourceKind::Image && resource.curent_layout() != access.image_layout();

        if access.writes() {
            self.resources[resource_handle].versions.push(GraphResourceVersion {
                image_layout: access.image_layout(),
                source_pass: Some(pass_handle),
                read_by: Vec::new(),
            });
        } else if needs_layout_transition {
            let source_pass = self.resources[resource_handle].versions.last().map(|v| v.source_pass).flatten();
            self.resources[resource_handle].versions.push(GraphResourceVersion {
                image_layout: access.image_layout(),
                source_pass,
                read_by: vec![pass_handle],
            });
        } else if access.read_only() {
            self.resources[resource_handle].versions.last_mut().unwrap().read_by.push(pass_handle);
        }

        dependency
    }
}

#[derive(Debug)]
struct TransientResourceNode {
    resource: graphics::AnyResource,
    prev_node: Option<arena::Index>,
    next_node: Option<arena::Index>,
}

#[derive(Debug, Default)]
pub struct TransientResourceCache {
    resources_nodes: arena::Arena<TransientResourceNode>,
    descriptor_lookup: HashMap<graphics::AnyResourceDesc, arena::Index>,
    name_lookup: HashMap<Cow<'static, str>, arena::Index>,
}

impl TransientResourceCache {
    pub fn new() -> Self {
        Self {
            resources_nodes: arena::Arena::new(),
            descriptor_lookup: HashMap::new(),
            name_lookup: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.resources_nodes.is_empty() && self.descriptor_lookup.is_empty()
    }

    pub fn clear(&mut self) {
        self.resources_nodes.clear();
        self.descriptor_lookup.clear();
    }

    pub fn resources(&self) -> impl Iterator<Item = &graphics::AnyResource> {
        self.resources_nodes.iter().map(|(_, node)| &node.resource)
    }

    pub fn drain_resources(&mut self) -> impl Iterator<Item = graphics::AnyResource> + '_ {
        self.descriptor_lookup.clear();
        self.resources_nodes.drain().map(|node| node.resource)
    }

    pub fn get(&mut self, name: &str, desc: &graphics::AnyResourceDesc) -> Option<(graphics::AnyResource, bool)> {
        if let Some(index) = self.name_lookup.get(name).copied() {
            if self.resources_nodes.get(index).map_or(false, |n| &n.resource.desc() == desc) {
                self.name_lookup.remove(name);
                let found_node = self.resources_nodes.remove(index).unwrap();

                if let Some(prev_node_index) = found_node.prev_node {
                    self.resources_nodes[prev_node_index].next_node = found_node.next_node;
                } else {
                    if let Some(next_node_index) = found_node.next_node {
                        *self.descriptor_lookup.get_mut(desc).unwrap() = next_node_index;
                    } else {
                        self.descriptor_lookup.remove(desc);
                    }
                }

                if let Some(next_node_index) = found_node.next_node {
                    self.resources_nodes[next_node_index].prev_node = found_node.prev_node;
                }

                return Some((found_node.resource, false));
            }
        };

        let descriptor_lookup_index = self.descriptor_lookup.get_mut(desc)?;
        let found_node = self.resources_nodes.remove(*descriptor_lookup_index).unwrap();

        if let Some(prev_node_index) = found_node.prev_node {
            self.resources_nodes[prev_node_index].next_node = found_node.next_node;
        }

        if let Some(next_node_index) = found_node.next_node {
            *descriptor_lookup_index = next_node_index;
            self.resources_nodes[next_node_index].prev_node = found_node.prev_node;
        } else {
            self.descriptor_lookup.remove(desc);
        }

        Some((found_node.resource, true))
    }

    pub fn insert(&mut self, resource: graphics::AnyResource) {
        let name = resource.clone_name();
        let desc = resource.desc();

        let resource_index = self.resources_nodes.insert(TransientResourceNode {
            resource,
            prev_node: None,
            next_node: None,
        });

        self.name_lookup.insert(name, resource_index);

        if let Some(index) = self.descriptor_lookup.get_mut(&desc) {
            self.resources_nodes[resource_index].next_node = Some(*index);
            self.resources_nodes[*index].prev_node = Some(resource_index);
            *index = resource_index;
        } else {
            self.descriptor_lookup.insert(desc, resource_index);
        }
    }
}

#[derive(Debug)]
pub struct BatchData {
    pub wait_semaphore_range: Range<usize>,
    pub begin_dependency_range: Range<usize>,
    pub begin_image_barrier_range: Range<usize>,

    pub memory_barrier: vk::MemoryBarrier2,
    pub pass_range: Range<usize>,

    pub finish_dependency_range: Range<usize>,
    pub finish_image_barrier_range: Range<usize>,
    pub signal_semaphore_range: Range<usize>,
}

unsafe impl Sync for BatchData {}

#[derive(Debug, Clone, Copy)]
pub struct BatchDependency {
    pub resource_index: usize,
    pub dst_access: graphics::AccessKind,
}

pub struct CompiledPassData {
    pub name: Cow<'static, str>,
    pub func: Pass,
}

impl From<PassData> for CompiledPassData {
    fn from(value: PassData) -> Self {
        Self {
            name: value.name,
            func: value.func,
        }
    }
}

impl std::fmt::Debug for CompiledPassData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledPassData").field("name", &self.name).finish()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphImageBarrier {
    pub image_index: usize,
    pub src_access: graphics::AccessFlags,
    pub dst_access: graphics::AccessFlags,
}

#[derive(Debug)]
pub struct CompiledGraphResource {
    pub resource: graphics::AnyResource,
    pub owned_by_graph: bool,
}

#[derive(Debug, Default)]
pub struct CompiledRenderGraph {
    pub resources: Vec<CompiledGraphResource>,
    pub passes: Vec<CompiledPassData>,
    pub draws: Vec<RenderPassDraw>,
    pub dependencies: Vec<BatchDependency>,
    pub image_barriers: Vec<GraphImageBarrier>,
    pub semaphores: Vec<(graphics::Semaphore, vk::PipelineStageFlags2)>,
    pub batches: Vec<BatchData>,
}

pub struct BatchRef<'a> {
    pub wait_semaphores: &'a [(graphics::Semaphore, vk::PipelineStageFlags2)],
    pub begin_dependencies: &'a [BatchDependency],
    pub begin_image_barriers: &'a [GraphImageBarrier],

    pub memory_barrier: vk::MemoryBarrier2,
    pub passes: &'a [CompiledPassData],

    pub finish_dependencies: &'a [BatchDependency],
    pub finish_image_barriers: &'a [GraphImageBarrier],
    pub signal_semaphores: &'a [(graphics::Semaphore, vk::PipelineStageFlags2)],
}

unsafe impl Sync for BatchRef<'_> {}

impl CompiledRenderGraph {
    pub fn iter_batches(&self) -> impl Iterator<Item = BatchRef> {
        self.batches.iter().map(|batch_data| self.get_batch_ref(batch_data))
    }

    pub fn get_batch_ref(&self, batch_data: &BatchData) -> BatchRef {
        BatchRef {
            wait_semaphores: &self.semaphores[batch_data.wait_semaphore_range.clone()],
            begin_dependencies: &self.dependencies[batch_data.begin_dependency_range.clone()],
            begin_image_barriers: &self.image_barriers[batch_data.begin_image_barrier_range.clone()],

            memory_barrier: batch_data.memory_barrier,
            passes: &self.passes[batch_data.pass_range.clone()],

            finish_dependencies: &self.dependencies[batch_data.finish_dependency_range.clone()],
            finish_image_barriers: &self.image_barriers[batch_data.finish_image_barrier_range.clone()],
            signal_semaphores: &self.semaphores[batch_data.signal_semaphore_range.clone()],
        }
    }
}

impl CompiledRenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.draws.clear();
        self.dependencies.clear();
        self.image_barriers.clear();
        self.semaphores.clear();
        self.batches.clear();
    }

    #[track_caller]
    pub fn get_buffer(&self, handle: graphics::GraphHandle<graphics::BufferRaw>) -> &graphics::BufferRaw {
        match self.resources[handle.resource_index].resource.as_ref() {
            graphics::AnyResourceRef::Buffer(buffer) => buffer,
            graphics::AnyResourceRef::Image(image) => panic!(
                "attempted to access image as buffer: {:?} [{}]",
                image.name, handle.resource_index
            ),
        }
    }

    #[track_caller]
    pub fn get_image(&self, handle: graphics::GraphHandle<graphics::ImageRaw>) -> &graphics::ImageRaw {
        let resource_ref = self.resources[handle.resource_index].resource.as_ref();

        match resource_ref {
            graphics::AnyResourceRef::Image(image) => image,
            graphics::AnyResourceRef::Buffer(buffer) => panic!(
                "attempted to access buffer as image: {:?} [{}]",
                buffer.name, handle.resource_index
            ),
        }
    }
}

impl RenderGraph {
    pub fn compile_and_flush(&mut self, device: &graphics::Device, compiled: &mut CompiledRenderGraph) {
        puffin::profile_function!();
        compiled.clear();

        std::mem::swap(&mut compiled.draws, &mut self.draws);

        let mut src_accesses: Vec<graphics::AccessFlags> =
            self.resources.iter().map(|r| r.initial_access.access_flags()).collect();

        let mut dst_accesses = vec![graphics::AccessFlags::default(); self.resources.len()];

        let mut pending_image_barriers = Vec::new();
        let mut pending_res_semaphores = Vec::new();

        let mut sorted_passes = self.take_passes_with_topology_sort();

        for pass_range in sorted_passes.ranges.iter() {
            pending_image_barriers.clear();
            pending_res_semaphores.clear();

            let mut memory_barrier = vk::MemoryBarrier2::default();
            // first pass of dependencies, order matters for limiting allocations
            // while preserving data contiguity
            let begin_dependency_start = compiled.dependencies.len();
            for slot in pass_range.clone() {
                let (_, pass) = sorted_passes.passes.get_slot(slot as u32).unwrap();

                for dependency in &self.dependencies[pass.dependency_range.clone()] {
                    let resource_data = &mut self.resources[dependency.resource_handle];

                    if dependency.resource_version == 0 && resource_data.wait_semaphore.is_some() {
                        pending_res_semaphores.push(dependency.resource_handle);
                    }

                    match resource_data.kind() {
                        ResourceKind::Buffer => {
                            dst_accesses[dependency.resource_handle].extend_buffer_access(dependency.access);
                            graphics::extend_memory_barrier(
                                &mut memory_barrier,
                                src_accesses[dependency.resource_handle],
                                dst_accesses[dependency.resource_handle],
                            );
                        }
                        ResourceKind::Image => {
                            if src_accesses[dependency.resource_handle].layout != dependency.access.image_layout() {
                                pending_image_barriers.push(dependency.resource_handle);
                            }
                            dst_accesses[dependency.resource_handle].layout = dependency.access.image_layout();
                            dst_accesses[dependency.resource_handle].extend_image_access(dependency.access);
                        }
                    }

                    // TODO: remove duplicate dependencies
                    compiled.dependencies.push(BatchDependency {
                        resource_index: dependency.resource_handle,
                        dst_access: dependency.access,
                    })
                }
            }
            let begin_dependency_end = compiled.dependencies.len();

            // finish image barriers
            pending_image_barriers.sort_unstable();
            pending_image_barriers.dedup();

            let begin_image_barriers_start = compiled.image_barriers.len();
            for &image_index in pending_image_barriers.iter() {
                compiled.image_barriers.push(GraphImageBarrier {
                    image_index,
                    src_access: src_accesses[image_index],
                    dst_access: dst_accesses[image_index],
                })
            }
            let begin_image_barriers_end = compiled.image_barriers.len();

            // wait semaphores
            let wait_semaphore_start = compiled.semaphores.len();
            for res_index in pending_res_semaphores.iter() {
                let Some(semaphore) = self.resources[*res_index].wait_semaphore.take() else {
                    continue;
                };
                compiled.semaphores.push((semaphore, dst_accesses[*res_index].stage_flags));
            }
            let wait_semaphore_end = compiled.semaphores.len();

            src_accesses.copy_from_slice(&dst_accesses);

            // second dependency pass
            let signal_semaphore_start = compiled.semaphores.len();
            let finish_dependency_start = compiled.dependencies.len();
            let finish_image_barrier_start = compiled.image_barriers.len();
            for slot in pass_range.clone() {
                let pass = sorted_passes.passes.remove_slot(slot as u32).unwrap();

                for dependency in &self.dependencies[pass.dependency_range.clone()] {
                    let resource_data = &mut self.resources[dependency.resource_handle];

                    // source of the last version of the resource
                    if dependency.access.writes() && dependency.resource_version == resource_data.current_version() - 1
                    {
                        if let Some(semaphore) = resource_data.finish_semaphore.take() {
                            if self.dont_signal_semaphores.insert(semaphore.handle) {
                                // compiled.semaphores.push((semaphore, dependency.access.stage_mask()));
                                // might not be accurate if the resource is used in a later stage than its latest stored access
                                compiled
                                    .semaphores
                                    .push((semaphore, src_accesses[dependency.resource_handle].stage_flags));
                            }
                        }

                        if resource_data.kind() == ResourceKind::Image
                            && resource_data.target_access != graphics::AccessKind::None
                            && dependency.access.image_layout() != resource_data.target_access.image_layout()
                        {
                            compiled.image_barriers.push(GraphImageBarrier {
                                image_index: dependency.resource_handle,
                                src_access: dst_accesses[dependency.resource_handle],
                                dst_access: resource_data.target_access.access_flags(),
                            });

                            compiled.dependencies.push(BatchDependency {
                                resource_index: dependency.resource_handle,
                                dst_access: resource_data.target_access,
                            });
                        }
                    }
                }

                compiled.passes.push(pass.into());
            }
            let signal_semaphore_end = compiled.semaphores.len();
            let finish_dependency_end = compiled.dependencies.len();
            let finish_image_barrier_end = compiled.image_barriers.len();

            compiled.batches.push(BatchData {
                wait_semaphore_range: wait_semaphore_start..wait_semaphore_end,
                begin_dependency_range: begin_dependency_start..begin_dependency_end,
                begin_image_barrier_range: begin_image_barriers_start..begin_image_barriers_end,

                memory_barrier,
                pass_range: pass_range.clone(),

                finish_dependency_range: finish_dependency_start..finish_dependency_end,
                finish_image_barrier_range: finish_image_barrier_start..finish_image_barrier_end,
                signal_semaphore_range: signal_semaphore_start..signal_semaphore_end,
            });
        }

        for resource_data in self.resources.drain(..) {
            match resource_data.source {
                ResourceSource::Create { desc, cache } => {
                    let resource = cache.unwrap_or_else(|| {
                        graphics::AnyResource::create_owned(
                            device,
                            resource_data.name,
                            &desc,
                            resource_data.descriptor_index,
                        )
                    });

                    compiled.resources.push(CompiledGraphResource {
                        resource,
                        owned_by_graph: true,
                    });
                }
                ResourceSource::Import { resource } => {
                    compiled.resources.push(CompiledGraphResource {
                        resource,
                        owned_by_graph: false,
                    });
                }
            }
        }

        self.clear();
    }
}

#[derive(Debug)]
struct SortedPassIndices {
    ranges: Vec<Range<usize>>,
    passes: Vec<GraphPassIndex>,
}

impl SortedPassIndices {
    fn passes(&self) -> impl Iterator<Item = &[GraphPassIndex]> {
        self.ranges.iter().map(|range| &self.passes[range.clone()])
    }
}

#[derive(Debug)]
struct SortedPasses {
    ranges: Vec<Range<usize>>,
    passes: arena::Arena<PassData>,
}

impl RenderGraph {
    fn read_passes_with_topology_sort(&self) -> SortedPassIndices {
        puffin::profile_function!();

        let mut sorted_passes = SortedPassIndices {
            ranges: Vec::with_capacity(self.passes.len()), // worst case
            passes: Vec::with_capacity(self.passes.len()),
        };

        let mut remaining_passes: Vec<arena::Index> = self.passes.iter().map(|(index, _)| index).collect();
        remaining_passes.sort_by_key(|index| index.slot);

        while !remaining_passes.is_empty() {
            let start = sorted_passes.passes.len();

            for &pass in remaining_passes.iter() {
                if self
                    .prev_passes(pass)
                    .all(|pass| remaining_passes.binary_search_by_key(&pass.slot, |index| index.slot).is_err())
                {
                    sorted_passes.passes.push(pass);
                }
            }

            let end = sorted_passes.passes.len();

            remaining_passes.retain(|index| {
                sorted_passes.passes[start..end].binary_search_by_key(&index.slot, |index| index.slot).is_err()
            });

            sorted_passes.ranges.push(start..end);
        }

        sorted_passes
    }

    fn take_passes_with_topology_sort(&mut self) -> SortedPasses {
        puffin::profile_function!();

        let mut sorted_passes = SortedPasses {
            ranges: Vec::with_capacity(self.passes.len()), // worst case
            passes: arena::Arena::with_capacity(self.passes.len() as u32),
        };

        let mut remove_list = Vec::new();
        while !self.passes.is_empty() {
            let start = sorted_passes.passes.len();

            let len = self.passes.len();
            let mut occupied = 0;
            let mut slot = 0;

            while occupied < len {
                let result = self.passes.get_slot(slot as u32);
                slot += 1;

                let Some((index, _)) = result else {
                    continue;
                };
                occupied += 1;

                if !self.prev_passes(index).any(|prev_index| self.passes.has_index(prev_index)) {
                    remove_list.push(index);
                }
            }

            for index in remove_list.iter().copied() {
                let pass_data = self.passes.remove(index).unwrap();
                sorted_passes.passes.insert(pass_data);
            }

            // just to catch some bugs if I change the render graph
            if remove_list.is_empty() {
                log::debug!("POTENTIAL INFINITE LOOP: passes left:");
                for (_, pass) in self.passes.iter() {
                    log::debug!("\t pass: {:?}", &pass.name);
                }
                break;
            }

            remove_list.clear();

            let end = sorted_passes.passes.len();

            sorted_passes.ranges.push(start..end);
        }

        sorted_passes
    }

    fn prev_passes(&self, pass: GraphPassIndex) -> impl Iterator<Item = GraphPassIndex> + '_ {
        self.passes
            .get(pass)
            .map(|pass| {
                self.dependencies[pass.dependency_range.clone()]
                    .iter()
                    .map(|dependency| {
                        let resource = &self.resources[dependency.resource_handle];

                        let prev_reads = (dependency.resource_version > 0)
                            .then(|| resource.versions[dependency.resource_version - 1].read_by.iter().copied())
                            .into_iter()
                            .flatten();

                        let source_pass = resource.source_pass(dependency.resource_version);

                        prev_reads.chain(source_pass)
                    })
                    .flatten()
            })
            .into_iter()
            .flatten()
    }
}
