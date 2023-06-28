use std::{collections::HashSet, ops::Range};

use ash::vk;

use crate::{render, collections::arena};

pub type ResourceHandle = usize;
pub type PassHandle = arena::Index;
pub type DependencyHandle = usize;

type PassFn = Box<dyn Fn(&render::CommandRecorder, &render::CompiledRenderGraph)>;

pub struct Pass {
    pub name: String,
    pub func: PassFn,
}

impl std::fmt::Debug for Pass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name.fmt(f)
    }
}

trait RenderResource {
    type View;
    type Desc;

    fn view(&self) -> Self::View;
}

impl RenderResource for render::Buffer {
    type View = render::BufferView;
    type Desc = render::BufferDesc;

    fn view(&self) -> Self::View {
        self.buffer_view
    }
}

impl RenderResource for render::Image {
    type View = render::ImageView;
    type Desc = render::ImageDesc;

    fn view(&self) -> Self::View {
        self.image_view
    }
}

#[derive(Debug, Clone)]
pub enum AnyResource {
    Buffer(render::Buffer),
    Image(render::Image),
}

#[derive(Debug, Clone, Copy)]
pub enum AnyResourceView {
    Buffer(render::BufferView),
    Image(render::ImageView),
}

#[derive(Debug, Clone, Copy)]
pub enum AnyResourceDesc {
    Buffer(render::BufferDesc),
    Image(render::ImageDesc),
}

impl RenderResource for AnyResource {
    type View = AnyResourceView;
    type Desc = AnyResourceDesc;

    fn view(&self) -> Self::View {
        match self {
            AnyResource::Buffer(buffer) => AnyResourceView::Buffer(buffer.view()),
            AnyResource::Image(image) => AnyResourceView::Image(image.view()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Buffer,
    Image,
}

impl AnyResource {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResource::Buffer(_)  => ResourceKind::Buffer,
            AnyResource::Image(_)   => ResourceKind::Image,
        }
    }
}

impl AnyResourceView {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceView::Buffer(_)  => ResourceKind::Buffer,
            AnyResourceView::Image(_)   => ResourceKind::Image,
        }
    }
}

impl AnyResourceDesc {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceDesc::Buffer(_)  => ResourceKind::Buffer,
            AnyResourceDesc::Image(_)   => ResourceKind::Image,
        }
    }
}

#[derive(Debug)]
pub enum ResourceSource {
    Import {
        view: AnyResourceView,
    },
    Create {
        desc: AnyResourceDesc,
    }
}

#[derive(Debug)]
pub struct ResourceVersion {
    last_access: render::AccessKind,
    source_pass: PassHandle,
}

#[derive(Debug)]
pub struct ResourceData {
    name: String,

    source: ResourceSource,
    resource_kind: ResourceKind,
    is_transient: bool,

    initial_access: render::AccessKind,
    target_access: render::AccessKind,
    wait_semaphore: Option<vk::Semaphore>,
    finish_semaphore: Option<vk::Semaphore>,

    versions: Vec<ResourceVersion>,
}

impl ResourceData {
    fn current_version(&self) -> usize {
        self.versions.len()
    }

    fn last_access(&self, version: usize) -> render::AccessKind {
        assert!(version <= self.versions.len());
        if version == 0 {
            self.initial_access
        } else {
            self.versions[version - 1].last_access
        }
    }

    fn source_pass(&self, version: usize) -> Option<PassHandle> {
        if version != 0 {
            Some(self.versions[version - 1].source_pass)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct PassData {
    pass: Pass,
    dependencies: Vec<DependencyHandle>,
    alive: bool,
}

#[derive(Debug)]
struct DependencyData {
    access: render::AccessKind,
    pass_handle: PassHandle,
    resource_handle: ResourceHandle,
    resource_version: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphResourceImportDesc {
    pub initial_access: render::AccessKind,
    pub target_access: render::AccessKind,
    pub wait_semaphore: Option<vk::Semaphore>,
    pub finish_semaphore: Option<vk::Semaphore>,
}

#[derive(Debug)]
pub struct RenderGraph {
    resources: Vec<ResourceData>,
    passes: arena::Arena<PassData>,
    dependencies: Vec<DependencyData>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            passes: arena::Arena::new(),
            dependencies: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.dependencies.clear();
    }

    fn add_resource(&mut self, resource_data: ResourceData) -> ResourceHandle {
        let index = self.resources.len();
        self.resources.push(resource_data);
        index
    }

    pub fn add_transient_resource(
        &mut self, 
        name: String,
        desc: AnyResourceDesc,
    ) -> ResourceHandle {
        self.add_resource(ResourceData {
            name,
            
            source: ResourceSource::Create { desc },
            resource_kind: desc.kind(),

            is_transient: true,
            initial_access: render::AccessKind::None,
            target_access: render::AccessKind::None,
            wait_semaphore: None,
            finish_semaphore: None,

            versions: vec![],
        })
    }

    pub fn import_resource(
        &mut self,
        name: String,
        view: AnyResourceView,
        desc: &GraphResourceImportDesc,
    ) -> ResourceHandle {
        self.add_resource(ResourceData {
            name,

            source: ResourceSource::Import { view },
            resource_kind: view.kind(),
            is_transient: false,

            initial_access: desc.initial_access,
            target_access: desc.target_access,
            wait_semaphore: desc.wait_semaphore,
            finish_semaphore: desc.finish_semaphore,

            versions: vec![],
        })
    }

    pub fn add_pass(&mut self, name: String, func: PassFn) -> PassHandle {
        self.passes.insert(PassData {
            pass: Pass { name, func },
            dependencies: Vec::new(),
            alive: false,
        })
    }

    pub fn add_dependency(
        &mut self,
        pass_handle: PassHandle,
        resource_handle: ResourceHandle,
        access: render::AccessKind,
    ) -> usize {
        let resource_version = self.resources[resource_handle].current_version();

        let dependency = self.dependencies.len();
        self.dependencies.push(DependencyData {
            access,
            pass_handle,
            resource_handle,
            resource_version,
        });

        self.passes[pass_handle].dependencies.push(dependency);

        if access.read_write_kind() == render::ReadWriteKind::Write {
            self.resources[resource_handle].versions.push(ResourceVersion {
                last_access: access,
                source_pass: pass_handle,
            });
        }

        dependency
    }
}

#[derive(Debug)]
pub struct BatchData {
    pub wait_semaphore_range: Range<usize>,
    pub memory_barrier: vk::MemoryBarrier2,
    pub begin_image_barrier_range: Range<usize>,

    pub pass_range: Range<usize>,

    pub finish_image_barrier_range: Range<usize>,
    pub finish_semaphore_range: Range<usize>,
}

#[derive(Debug)]
pub struct GraphResource {
    name: String,
    resource: AnyResourceView,
}

#[derive(Debug, Default)]
pub struct CompiledRenderGraph {
    pub resources: Vec<GraphResource>,
    pub passes: Vec<Pass>,
    pub image_barriers: Vec<vk::ImageMemoryBarrier2>,
    pub semaphores: Vec<vk::Semaphore>,
    pub batches: Vec<BatchData>,

    pub transient_buffers: Vec<render::Buffer>,
    pub transient_images: Vec<render::Image>,
}

pub struct Batch<'a> {
    pub wait_semaphores: &'a [vk::Semaphore],
    pub memory_barrier: vk::MemoryBarrier2,
    pub begin_image_barriers: &'a [vk::ImageMemoryBarrier2],

    pub passes: &'a [Pass],

    pub finish_image_barriers: &'a [vk::ImageMemoryBarrier2],
    pub finish_semaphores: &'a [vk::Semaphore],
}

impl CompiledRenderGraph {
    pub fn iter_batches(&self) -> impl Iterator<Item = Batch> {
        self.batches.iter().map(|batch_data| Batch {
            wait_semaphores: &self.semaphores[batch_data.wait_semaphore_range.clone()],
            memory_barrier: batch_data.memory_barrier,
            begin_image_barriers: &self.image_barriers[batch_data.begin_image_barrier_range.clone()],
            passes: &self.passes[batch_data.pass_range.clone()],
            finish_image_barriers: &self.image_barriers[batch_data.finish_image_barrier_range.clone()],
            finish_semaphores: &self.semaphores[batch_data.finish_semaphore_range.clone()],
        })
    }
}

impl CompiledRenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.image_barriers.clear();
        self.semaphores.clear();
        self.batches.clear();
    }

    pub fn get_buffer(&self, handle: render::ResourceHandle) -> Option<&render::BufferView> {
        match &self.resources[handle].resource {
            AnyResourceView::Buffer(buffer) => Some(buffer),
            _ => None,
        }
    }

    pub fn get_image(&self, handle: render::ResourceHandle) -> Option<&render::ImageView> {
        match &self.resources[handle].resource {
            AnyResourceView::Image(image) => Some(image),
            _ => None,
        }
    }
}

impl RenderGraph {
    pub fn compile_and_flush(
        &mut self,
        device: &render::Device,
        descriptors: &render::BindlessDescriptors,
        compiled: &mut CompiledRenderGraph
    ) {
        puffin::profile_function!();
        compiled.clear();

        let sorted_passes = self.topology_sort();

        for resource_data in self.resources.iter_mut() {
            // TODO: allocations can be reduces here by not having the names in both the resource and the GraphResource
            let mut name = String::new();
            std::mem::swap(&mut name, &mut resource_data.name);
            match &resource_data.source {
                ResourceSource::Import { view } => {
                    compiled.resources.push(GraphResource {
                        name,
                        resource: *view,
                    });
                },
                ResourceSource::Create { desc } => {
                    match desc {
                        AnyResourceDesc::Buffer(desc) => {
                            let buffer = render::Buffer::create_impl(
                                device,
                                descriptors,
                                resource_data.name.clone().into(),
                                desc
                            );
                            
                            compiled.resources.push(GraphResource {
                                name,
                                resource: AnyResourceView::Buffer(buffer.view()),
                            });
                            compiled.transient_buffers.push(buffer);
                        },
                        AnyResourceDesc::Image(desc) => {
                            let image = render::Image::create_impl(
                                device,
                                descriptors,
                                resource_data.name.clone().into(),
                                desc
                            );
                            
                            compiled.resources.push(GraphResource {
                                name,
                                resource: AnyResourceView::Image(image.view()),
                            });
                            compiled.transient_images.push(image);
                        },
                    }
                },
            } 
        }

        for pass_range in sorted_passes.ranges.iter() {
            let mut batch = BatchData {
                wait_semaphore_range: compiled.semaphores.len()..compiled.semaphores.len(),
                memory_barrier: vk::MemoryBarrier2::default(),
                begin_image_barrier_range: compiled.image_barriers.len()..compiled.image_barriers.len(),

                pass_range: pass_range.clone(),

                // will be set later, using the initial values is a bug
                finish_image_barrier_range: usize::MAX..usize::MAX,
                finish_semaphore_range: usize::MAX..usize::MAX,
            };

            let passes = &sorted_passes.passes[pass_range.clone()];

            for &pass in passes {
                let Some(pass) = self.passes.remove(pass) else { continue };

                // first pass of dependencies, order matters for limiting allocations
                // while preserving data contiguity
                for &dependency in pass.dependencies.iter() {
                    let dependency = &self.dependencies[dependency];
                    let resource_data = &self.resources[dependency.resource_handle];
                    let resource_kind = resource_data.resource_kind;

                    if dependency.resource_version == 0 {
                        if let Some(semaphore) = resource_data.wait_semaphore {
                            batch.wait_semaphore_range.end += 1;
                            compiled.semaphores.push(semaphore);
                        }
                    }

                    let src_access = resource_data.last_access(dependency.resource_version);
                    let dst_access = dependency.access;

                    if resource_kind != ResourceKind::Image || src_access.image_layout() == dst_access.image_layout() {
                        batch.memory_barrier.src_stage_mask |= src_access.stage_mask();
                        if src_access.read_write_kind() == render::ReadWriteKind::Write {
                            batch.memory_barrier.src_access_mask |= src_access.access_mask();
                        }

                        batch.memory_barrier.dst_stage_mask |= dst_access.stage_mask();
                        if !batch.memory_barrier.src_access_mask.is_empty() {
                            batch.memory_barrier.dst_access_mask |= dst_access.access_mask();
                        }
                    } else if let AnyResourceView::Image(image) = &compiled.resources[dependency.resource_handle].resource {
                        batch.begin_image_barrier_range.end += 1;
                        compiled.image_barriers.push(render::image_barrier(image, src_access, dst_access));
                    } else {
                        unimplemented!()
                    }
                }

                if batch.memory_barrier.src_stage_mask.is_empty() {
                    batch.memory_barrier.src_stage_mask = vk::PipelineStageFlags2::TOP_OF_PIPE;
                }

                if batch.memory_barrier.dst_stage_mask.is_empty() {
                    batch.memory_barrier.dst_stage_mask = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
                }

                batch.finish_image_barrier_range = compiled.image_barriers.len()..compiled.image_barriers.len();
                batch.finish_semaphore_range = compiled.semaphores.len()..compiled.semaphores.len();

                for &dependency in pass.dependencies.iter() {
                    let dependency = &self.dependencies[dependency];
                    let resource_data = &self.resources[dependency.resource_handle];

                    // source of the last version of the resource
                    if dependency.access.read_write_kind() == render::ReadWriteKind::Write
                        && dependency.resource_version == resource_data.current_version() - 1
                    {
                        if let Some(semaphore) = resource_data.finish_semaphore {
                            batch.finish_semaphore_range.end += 1;
                            compiled.semaphores.push(semaphore);
                        }

                        if let AnyResourceView::Image(image) = &compiled.resources[dependency.resource_handle].resource {
                            if resource_data.target_access != render::AccessKind::None
                                && dependency.access != resource_data.target_access
                            {
                                batch.finish_image_barrier_range.end += 1;
                                compiled.image_barriers.push(render::image_barrier(
                                    image,
                                    dependency.access,
                                    resource_data.target_access,
                                ))
                            }
                        }
                    }
                }

                compiled.passes.push(pass.pass);
            }

            compiled.batches.push(batch);
        }

        self.clear();
    }
}

#[derive(Debug)]
struct SortedPasses {
    ranges: Vec<Range<usize>>,
    passes: Vec<PassHandle>,
}

impl SortedPasses {
    fn passes(&self) -> impl Iterator<Item = &[PassHandle]> {
        self.ranges.iter().map(|range| &self.passes[range.clone()])
    }
}

impl RenderGraph {
    fn topology_sort(&self) -> SortedPasses {
        puffin::profile_function!();
        // TODO: maybe use a better algo,
        // though this seems to be fast enough

        let mut sorted_passes = SortedPasses {
            ranges: Vec::with_capacity(self.passes.len()), // worst case
            passes: Vec::with_capacity(self.passes.len()),
        };

        let mut remainging_passes: HashSet<arena::Index> = self.passes.iter().map(|(index, _)| index).collect();

        while !remainging_passes.is_empty() {
            let start = sorted_passes.passes.len();

            for &pass in remainging_passes.iter() {
                if self.prev_passes(pass).all(|pass| !remainging_passes.contains(&pass)) {
                    sorted_passes.passes.push(pass);
                }
            }

            let end = sorted_passes.passes.len();

            for slot in &sorted_passes.passes[start..end] {
                remainging_passes.remove(slot);
            }

            sorted_passes.ranges.push(start..end);
        }

        sorted_passes
    }

    fn prev_passes(&self, pass: PassHandle) -> impl Iterator<Item = PassHandle> + '_ {
        self.passes.get(pass).map(|pass| pass.dependencies.iter().filter_map(|&dependency_handle| {
            let handle = self.dependencies[dependency_handle].resource_handle;
            let version = self.dependencies[dependency_handle].resource_version;

            self.resources[handle].source_pass(version)
        })).into_iter().flatten()
    }
}
