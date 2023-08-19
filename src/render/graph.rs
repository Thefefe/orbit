use std::{collections::HashMap, ops::Range, borrow::Cow};

use ash::vk;

use crate::{render, collections::arena};
pub type GraphResourceIndex = usize;
pub type GraphPassIndex = arena::Index;
pub type GraphDependencyIndex = usize;

type PassFn = Box<dyn Fn(&render::CommandRecorder, &render::CompiledRenderGraph)>;

pub trait RenderResource {
    type View;
    type Desc;

    fn view(&self) -> Self::View;
    fn descriptor_index(&self) -> Option<render::DescriptorIndex>;
}

impl RenderResource for render::Buffer {
    type View = render::BufferView;
    type Desc = render::BufferDesc;

    fn view(&self) -> Self::View {
        self.buffer_view
    }

    fn descriptor_index(&self) -> Option<render::DescriptorIndex> {
        self.descriptor_index
    }
    
}

impl RenderResource for render::Image {
    type View = render::ImageView;
    type Desc = render::ImageDesc;

    fn view(&self) -> Self::View {
        self.full_view
    }

    fn descriptor_index(&self) -> Option<render::DescriptorIndex> {
        self.descriptor_index
    }
}

#[derive(Debug, Clone)]
pub enum AnyResource {
    Buffer(render::Buffer),
    Image(render::Image),
}

impl AnyResource {
    pub fn get_buffer(&self) -> Option<&render::Buffer> {
        match self {
            AnyResource::Buffer(buffer) => Some(buffer),
            AnyResource::Image(_) => None,
        }
    }

    pub fn get_image(&self) -> Option<&render::Image> {
        match self {
            AnyResource::Buffer(_) => None,
            AnyResource::Image(image) => Some(image),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AnyResourceView {
    Buffer(render::BufferView),
    Image(render::ImageView),
}

impl AnyResourceView {
    pub fn handle(&self) -> AnyResourceHandle {
        match self {
            AnyResourceView::Buffer(buffer) => AnyResourceHandle::Buffer(buffer.handle),
            AnyResourceView::Image(image) => AnyResourceHandle::Image(image.handle),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyResourceHandle {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    fn descriptor_index(&self) -> Option<render::DescriptorIndex> {
        match self {
            AnyResource::Buffer(buffer) => buffer.descriptor_index,
            AnyResource::Image(image) => image.descriptor_index,
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

    pub fn create(
        device: &render::Device,
        descriptors: &render::BindlessDescriptors,
        name: Cow<'static, str>,
        desc: &AnyResourceDesc,
        preallocated_descriptor_index: Option<render::DescriptorIndex>,
    ) -> Self {
        match desc {
            AnyResourceDesc::Buffer(desc) => {
                AnyResource::Buffer(render::Buffer::create_impl(
                    device,
                    descriptors,
                    name,
                    desc,
                    preallocated_descriptor_index
                ))
            },
            AnyResourceDesc::Image(desc) => {
                AnyResource::Image(render::Image::create_impl(
                    device,
                    descriptors,
                    name,
                    desc,
                    preallocated_descriptor_index
                ))
            },
        }
    }

    pub fn destroy(
        &self,
        device: &render::Device,
        descriptors: &render::BindlessDescriptors
    ) {
        match self {
            AnyResource::Buffer(buffer) => {
                render::Buffer::destroy_impl(device, descriptors, buffer);
            },
            AnyResource::Image(image) => {
                render::Image::destroy_impl(device, descriptors, image);
            },
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
    Create {
        desc: AnyResourceDesc,
        cache: Option<AnyResource>,
    },
    Import {
        view: AnyResourceView,
    },
}

#[derive(Debug)]
pub struct ResourceVersion {
    last_access: render::AccessKind,
    source_pass: GraphPassIndex,
}

#[derive(Debug)]
pub struct GraphResourceData {
    pub name: Cow<'static, str>,

    pub source: ResourceSource,
    pub descriptor_index: Option<render::DescriptorIndex>,

    pub initial_access: render::AccessKind,
    pub target_access: render::AccessKind,
    pub wait_semaphore: Option<render::Semaphore>,
    pub finish_semaphore: Option<render::Semaphore>,

    pub versions: Vec<ResourceVersion>,
}

impl GraphResourceData {
    fn kind(&self) -> render::ResourceKind {
        match self.source {
            ResourceSource::Create { desc, .. } => desc.kind(),
            ResourceSource::Import { view } => view.kind(),
        }
    }

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

    fn source_pass(&self, version: usize) -> Option<GraphPassIndex> {
        if version != 0 {
            Some(self.versions[version - 1].source_pass)
        } else {
            None
        }
    }
}

pub struct PassData {
    pub name: Cow<'static, str>,
    pub func: PassFn,
    dependencies: Vec<GraphDependencyIndex>,
    alive: bool,
}

impl std::fmt::Debug for PassData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassData")
            .field("name", &self.name)
            .field("dependencies", &self.dependencies)
            .field("alive", &self.alive)
            .finish()
    }
}

#[derive(Debug)]
struct DependencyData {
    access: render::AccessKind,
    pass_handle: GraphPassIndex,
    resource_handle: GraphResourceIndex,
    resource_version: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GraphResourceImportDesc {
    pub initial_access: render::AccessKind,
    pub target_access: render::AccessKind,
    pub wait_semaphore: Option<render::Semaphore>,
    pub finish_semaphore: Option<render::Semaphore>,
}

#[derive(Debug)]
pub struct RenderGraph {
    pub resources: Vec<GraphResourceData>,
    pub passes: arena::Arena<PassData>,
    dependencies: Vec<DependencyData>,
    
    pub import_cache: HashMap<AnyResourceHandle, GraphResourceIndex>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            passes: arena::Arena::new(),
            dependencies: Vec::new(),
            import_cache: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.dependencies.clear();
        self.import_cache.clear();
    }

    pub fn add_resource(&mut self, resource_data: GraphResourceData) -> GraphResourceIndex {
        let index = self.resources.len();
        let imported_handle = if let ResourceSource::Import { view } = &resource_data.source {
            Some(view.handle())
        } else {
            None
        };
        self.resources.push(resource_data);

        if let Some(handle) = imported_handle {
            self.import_cache.insert(handle, index);
        }

        index
    }

    pub fn add_pass(&mut self, name: Cow<'static, str>, func: PassFn) -> GraphPassIndex {
        self.passes.insert(PassData {
            name,
            func,
            dependencies: Vec::new(),
            alive: false,
        })
    }

    pub fn add_dependency(
        &mut self,
        pass_handle: GraphPassIndex,
        resource_handle: GraphResourceIndex,
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
pub struct CompiledGraphResource {
    pub name: Cow<'static, str>,
    pub resource_view: AnyResourceView,
}

#[derive(Debug)]
struct TransientResourceNode {
    resource: AnyResource,
    next_node: Option<arena::Index>,
}

#[derive(Debug, Default)]
pub struct TransientResourceCache {
    resources_nodes: arena::Arena<TransientResourceNode>,
    descriptor_lookup: HashMap<AnyResourceDesc, arena::Index>,
}

impl TransientResourceCache {
    pub fn new() -> Self {
        Self {
            resources_nodes: arena::Arena::new(),
            descriptor_lookup: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.resources_nodes.is_empty() && self.descriptor_lookup.is_empty()
    }

    pub fn clear(&mut self) {
        self.resources_nodes.clear();
        self.descriptor_lookup.clear();
    }

    pub fn resources(&self) -> impl Iterator<Item = &AnyResource> {
        self.resources_nodes.iter().map(|(_, node)| &node.resource)
    }

    pub fn get_by_descriptor(&mut self, desc: &AnyResourceDesc) -> Option<AnyResource> {
        let index = self.descriptor_lookup.get_mut(desc)?;
        let resource_node = self.resources_nodes.remove(*index).unwrap();
        
        if let Some(next_index) = resource_node.next_node {
            *index = next_index;
        } else {
            self.descriptor_lookup.remove(desc);
        }

        Some(resource_node.resource)
    }

    pub fn insert(&mut self, desc: AnyResourceDesc, resource: AnyResource) {
        if let Some(index) = self.descriptor_lookup.get_mut(&desc) {
            let new_index = self.resources_nodes.insert(TransientResourceNode {
                resource,
                next_node: Some(*index),
            });
            *index = new_index;
        } else {
            let index = self.resources_nodes.insert(TransientResourceNode {
                resource,
                next_node: None,
            });
            self.descriptor_lookup.insert(desc, index);
        }
    }
}

#[derive(Debug)]
pub struct BatchData {
    pub wait_semaphore_range: Range<usize>,
    pub begin_dependency_range: Range<usize>,

    pub pass_range: Range<usize>,

    pub finish_dependency_range: Range<usize>,
    pub signal_semaphore_range: Range<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchDependecy {
    pub resoure_index: usize,
    pub src_access: render::AccessKind,
    pub dst_access: render::AccessKind,
}

pub struct CompiledPassData {
    pub name: Cow<'static, str>,
    pub func: PassFn,
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
        f.debug_struct("CompiledPassData")
            .field("name", &self.name)
            .finish()
    }
}

#[derive(Debug, Default)]
pub struct CompiledRenderGraph {
    pub resources: Vec<CompiledGraphResource>,
    pub passes: Vec<CompiledPassData>,
    pub dependencies: Vec<BatchDependecy>,
    pub semaphores: Vec<(render::Semaphore, vk::PipelineStageFlags2)>,
    pub batches: Vec<BatchData>,
}

pub struct BatchRef<'a> {
    pub wait_semaphores: &'a [(render::Semaphore, vk::PipelineStageFlags2)],
    pub begin_dependencies: &'a [BatchDependecy],

    pub passes: &'a [CompiledPassData],

    pub finish_dependencies: &'a [BatchDependecy],
    pub signal_semaphores: &'a [(render::Semaphore, vk::PipelineStageFlags2)],
}

impl CompiledRenderGraph {
    pub fn iter_batches(&self) -> impl Iterator<Item = BatchRef> {
        self.batches.iter().map(|batch_data| BatchRef {
            wait_semaphores: &self.semaphores[batch_data.wait_semaphore_range.clone()], 
            begin_dependencies: &self.dependencies[batch_data.begin_dependency_range.clone()],
            passes: &self.passes[batch_data.pass_range.clone()],
            finish_dependencies: &self.dependencies[batch_data.finish_dependency_range.clone()],
            signal_semaphores: &self.semaphores[batch_data.signal_semaphore_range.clone()],
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
        self.dependencies.clear();
        self.semaphores.clear();
        self.batches.clear();
    }

    pub fn get_buffer(&self, handle: render::GraphHandle<render::Buffer>) -> &render::BufferView {
        match &self.resources[handle.resource_index].resource_view {
            AnyResourceView::Buffer(buffer) => buffer,
            _ => unreachable!(),
        }
    }

    pub fn get_image(&self, handle: render::GraphHandle<render::Image>) -> &render::ImageView {
        match &self.resources[handle.resource_index].resource_view {
            AnyResourceView::Image(image) => image,
            _ => unreachable!(),
        }
    }
}

impl RenderGraph {
    pub fn compile_and_flush(
        &mut self,
        device: &render::Device,
        descriptors: &render::BindlessDescriptors,
        compiled: &mut CompiledRenderGraph,
        to_be_used_transient_resource_cache: &mut TransientResourceCache,
    ) {
        assert!(to_be_used_transient_resource_cache.is_empty());
        puffin::profile_function!();
        compiled.clear();

        let mut sorted_passes = self.take_passes_with_topology_sort();

        for resource_data in self.resources.iter_mut() {
            // TODO: allocations can be reduces here by not having the names in both the resource and the GraphResource
            let mut name: Cow<'static, str> = Cow::Borrowed("");
            std::mem::swap(&mut name, &mut resource_data.name);
            match &mut resource_data.source {
                ResourceSource::Create { desc, cache, } => {
                    let resource = cache.take().unwrap_or_else(|| AnyResource::create(
                        device,
                        descriptors,
                        name.clone(),
                        desc,
                        resource_data.descriptor_index,
                    ));
                    let resource_view = resource.view();

                    compiled.resources.push(CompiledGraphResource { name, resource_view });
                    to_be_used_transient_resource_cache.insert(*desc, resource);
                },
                ResourceSource::Import { view } => {
                    compiled.resources.push(CompiledGraphResource {
                        name,
                        resource_view: *view,
                    });
                },  
            } 
        }

        for pass_range in sorted_passes.ranges.iter() {
            // first pass of dependencies, order matters for limiting allocations
            // while preserving data contiguity
            let wait_semaphore_start = compiled.semaphores.len();
            let begin_dependency_start = compiled.dependencies.len();
            for slot in pass_range.clone() {
                let (_, pass) = sorted_passes.passes.get_slot(slot as u32).unwrap();

                for &dependency in pass.dependencies.iter() {
                    let dependency = &self.dependencies[dependency];
                    let resource_data = &mut self.resources[dependency.resource_handle];

                    if dependency.resource_version == 0 {
                        if let Some(semaphore) = resource_data.wait_semaphore.take() {
                            // TODO: if the first access is multiple reads this may get duplicated
                            compiled.semaphores.push((semaphore, dependency.access.stage_mask()));
                        }
                    }

                    let src_access = resource_data.last_access(dependency.resource_version);
                    let dst_access = dependency.access;

                    // TODO: remove duplicate dependencies, handle seperate image
                    // layouts for same image (rare, but can happen) 
                    compiled.dependencies.push(BatchDependecy {
                        resoure_index: dependency.resource_handle,
                        src_access,
                        dst_access
                    })
                }
            }
            let wait_semaphore_end = compiled.semaphores.len();
            let begin_dependency_end = compiled.dependencies.len();

            // second dependency pass
            let signal_semaphore_start = compiled.semaphores.len();
            let finish_dependency_start = compiled.dependencies.len();
            for slot in pass_range.clone() {
                let pass = sorted_passes.passes.remove_slot(slot as u32).unwrap();
                
                for &dependency in pass.dependencies.iter() {
                    let dependency = &self.dependencies[dependency];
                    let resource_data = &mut self.resources[dependency.resource_handle];

                    // source of the last version of the resource
                    if dependency.access.read_write_kind() == render::ReadWriteKind::Write
                        && dependency.resource_version == resource_data.current_version() - 1
                    {
                        if let Some(semaphore) = resource_data.finish_semaphore.take() {
                            compiled.semaphores.push((semaphore, dependency.access.stage_mask()));
                        }

                        if resource_data.kind() == ResourceKind::Image &&
                           resource_data.target_access != render::AccessKind::None &&
                           dependency.access.image_layout() != resource_data.target_access.image_layout()
                        {
                            compiled.dependencies.push(BatchDependecy {
                                resoure_index: dependency.resource_handle,
                                src_access: dependency.access,
                                dst_access: resource_data.target_access,
                            });
                        }
                    }
                }

                compiled.passes.push(pass.into());
            }
            let signal_semaphore_end = compiled.semaphores.len();
            let finish_dependency_end = compiled.dependencies.len();

            compiled.batches.push(BatchData {
                wait_semaphore_range: wait_semaphore_start..wait_semaphore_end,
                begin_dependency_range: begin_dependency_start..begin_dependency_end,
                pass_range: pass_range.clone(),
                finish_dependency_range: finish_dependency_start..finish_dependency_end,
                signal_semaphore_range: signal_semaphore_start..signal_semaphore_end,
            });
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
                if self.prev_passes(pass).all(|pass| remaining_passes
                        .binary_search_by_key(&pass.slot, |index| index.slot).is_err())
                {
                    sorted_passes.passes.push(pass);
                }
            }

            let end = sorted_passes.passes.len();

            remaining_passes.retain(|index| sorted_passes.passes[start..end]
                .binary_search_by_key(&index.slot, |index| index.slot)
                .is_err()
            );

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

                let Some((index, _)) = result else { continue; };
                occupied += 1;

                if !self.prev_passes(index).any(|prev_index| self.passes.has_index(prev_index)) {
                    remove_list.push(index);
                }
            }

            for index in remove_list.iter().copied() {
                let pass_data = self.passes.remove(index).unwrap();
                sorted_passes.passes.insert(pass_data);
            }
            
            // temporary fix for infinite loop
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
        self.passes.get(pass).map(|pass| pass.dependencies.iter().filter_map(|&dependency_handle| {
            let handle = self.dependencies[dependency_handle].resource_handle;
            let version = self.dependencies[dependency_handle].resource_version;

            self.resources[handle].source_pass(version)
        })).into_iter().flatten()
    }
}
