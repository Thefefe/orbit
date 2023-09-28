use std::{collections::HashMap, ops::Range, borrow::Cow};
use std::hint::black_box;

use ash::vk;

use crate::{graphics, collections::arena};

use super::ReadWriteKind;
pub type GraphResourceIndex = usize;
pub type GraphPassIndex = arena::Index;
pub type GraphDependencyIndex = usize;

type PassFn = Box<dyn Fn(&graphics::CommandRecorder, &graphics::CompiledRenderGraph)>;

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

#[derive(Debug)]
pub struct GraphResourceVersion {
    pub last_access: graphics::AccessKind,
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
    pub wait_semaphore: Option<graphics::Semaphore>,
    pub finish_semaphore: Option<graphics::Semaphore>,

    pub versions: Vec<GraphResourceVersion>,
}

impl GraphResourceData {
    fn kind(&self) -> graphics::ResourceKind {
        match &self.source {
            ResourceSource::Create { desc, .. } => desc.kind(),
            ResourceSource::Import { resource, .. } => resource.kind(),
        }
    }

    fn current_version(&self) -> usize {
        self.versions.len() - 1
    }

    fn curent_layout(&self) -> vk::ImageLayout {
        self.versions.last().unwrap().last_access.image_layout()
    }

    fn last_access(&self, version: usize) -> graphics::AccessKind {
        assert!(version < self.versions.len());
        self.versions[version].last_access
    }

    fn source_pass(&self, version: usize) -> Option<GraphPassIndex> {
        self.versions[version].source_pass
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
    access: graphics::AccessKind,
    pass_handle: GraphPassIndex,
    resource_handle: GraphResourceIndex,
    resource_version: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GraphResourceImportDesc {
    pub initial_access: graphics::AccessKind,
    pub target_access: graphics::AccessKind,
    pub wait_semaphore: Option<graphics::Semaphore>,
    pub finish_semaphore: Option<graphics::Semaphore>,
}

#[derive(Debug)]
pub struct RenderGraph {
    pub resources: Vec<GraphResourceData>,
    pub passes: arena::Arena<PassData>,
    dependencies: Vec<DependencyData>,
    
    pub import_cache: HashMap<graphics::AnyResourceHandle, GraphResourceIndex>,
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
            last_access: resource_data.initial_access,
            source_pass: None,
            read_by: vec![]
        }];

        self.resources.push(resource_data);

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

        let resource_name: &str = self.resources[resource_handle].name.as_ref();

        if resource_name == "forward_draw_commands" {
            let _ = black_box(10);
        }

        self.passes[pass_handle].dependencies.push(dependency);

        let resource = &self.resources[resource_handle];
        let needs_layout_transition =
            resource.kind() == graphics::ResourceKind::Image &&
            resource.curent_layout() != access.image_layout();

        if access.read_write_kind() == graphics::ReadWriteKind::Write || needs_layout_transition {
            self.resources[resource_handle].versions.push(GraphResourceVersion {
                last_access: access,
                source_pass: Some(pass_handle),
                read_by: Vec::new(),
            });
        } else if access.read_write_kind() == graphics::ReadWriteKind::Read {
            self.resources[resource_handle].versions.last_mut().unwrap().read_by.push(pass_handle);
        }

        dependency
    }
}

#[derive(Debug)]
struct TransientResourceNode {
    resource: graphics::AnyResource,
    next_node: Option<arena::Index>,
}

#[derive(Debug, Default)]
pub struct TransientResourceCache {
    resources_nodes: arena::Arena<TransientResourceNode>,
    descriptor_lookup: HashMap<graphics::AnyResourceDesc, arena::Index>,
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

    pub fn resources(&self) -> impl Iterator<Item = &graphics::AnyResource> {
        self.resources_nodes.iter().map(|(_, node)| &node.resource)
    }

    pub fn drain_resources(&mut self) -> impl Iterator<Item = graphics::AnyResource> + '_ {
        self.descriptor_lookup.clear();
        self.resources_nodes.drain().map(|node| node.resource)
    }

    pub fn get_by_descriptor(&mut self, desc: &graphics::AnyResourceDesc) -> Option<graphics::AnyResource> {
        let index = self.descriptor_lookup.get_mut(desc)?;
        let resource_node = self.resources_nodes.remove(*index).unwrap();
        
        if let Some(next_index) = resource_node.next_node {
            *index = next_index;
        } else {
            self.descriptor_lookup.remove(desc);
        }

        Some(resource_node.resource)
    }

    pub fn insert(&mut self, desc: graphics::AnyResourceDesc, resource: graphics::AnyResource) {
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
pub struct BatchDependency {
    pub resource_index: usize,
    pub src_access: graphics::AccessKind,
    pub dst_access: graphics::AccessKind,
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

#[derive(Debug)]
pub struct CompiledGraphResource {
    pub resource: graphics::AnyResource,
    pub owned_by_graph: bool,
}

#[derive(Debug, Default)]
pub struct CompiledRenderGraph {
    pub resources: Vec<CompiledGraphResource>,
    pub passes: Vec<CompiledPassData>,
    pub dependencies: Vec<BatchDependency>,
    pub semaphores: Vec<(graphics::Semaphore, vk::PipelineStageFlags2)>,
    pub batches: Vec<BatchData>,
}

pub struct BatchRef<'a> {
    pub wait_semaphores: &'a [(graphics::Semaphore, vk::PipelineStageFlags2)],
    pub begin_dependencies: &'a [BatchDependency],

    pub passes: &'a [CompiledPassData],

    pub finish_dependencies: &'a [BatchDependency],
    pub signal_semaphores: &'a [(graphics::Semaphore, vk::PipelineStageFlags2)],
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

    #[track_caller]
    pub fn get_buffer(&self, handle: graphics::GraphHandle<graphics::BufferRaw>) -> &graphics::BufferRaw {
        match self.resources[handle.resource_index].resource.as_ref() {
            graphics::AnyResourceRef::Buffer(buffer) => buffer,
            graphics::AnyResourceRef::Image(image)
                => panic!("attempted to access image as buffer: {:?} [{}]", image.name, handle.resource_index),
        }
    }

    #[track_caller]
    pub fn get_image(&self, handle: graphics::GraphHandle<graphics::ImageRaw>) -> &graphics::ImageRaw {
        let resource_ref = self.resources[handle.resource_index].resource.as_ref();

        match resource_ref {
            graphics::AnyResourceRef::Image(image) => image,
            graphics::AnyResourceRef::Buffer(buffer)
                => panic!("attempted to access buffer as image: {:?} [{}]", buffer.name, handle.resource_index),
        }
    }
}

impl RenderGraph {
    pub fn compile_and_flush(
        &mut self,
        device: &graphics::Device,
        compiled: &mut CompiledRenderGraph,
    ) {
        puffin::profile_function!();
        compiled.clear();

        let mut sorted_passes = self.take_passes_with_topology_sort();

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

                    // TODO: remove duplicate dependencies, handle separate image
                    // layouts for same image (rare, but can happen) 
                    compiled.dependencies.push(BatchDependency {
                        resource_index: dependency.resource_handle,
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
                    if dependency.access.read_write_kind() == graphics::ReadWriteKind::Write
                        && dependency.resource_version == resource_data.current_version() - 1
                    {
                        if let Some(semaphore) = resource_data.finish_semaphore.take() {
                            compiled.semaphores.push((semaphore, dependency.access.stage_mask()));
                        }

                        if resource_data.kind() == graphics::ResourceKind::Image &&
                           resource_data.target_access != graphics::AccessKind::None &&
                           dependency.access.image_layout() != resource_data.target_access.image_layout()
                        {
                            compiled.dependencies.push(BatchDependency {
                                resource_index: dependency.resource_handle,
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

        for resource_data in self.resources.drain(..) {
            match resource_data.source {
                ResourceSource::Create { desc, cache, } => {
                    let resource = cache.unwrap_or_else(|| graphics::AnyResource::create_owned(
                        device,
                        resource_data.name,
                        &desc,
                        resource_data.descriptor_index,
                    ));

                    compiled.resources.push(CompiledGraphResource { resource, owned_by_graph: true });
                },
                ResourceSource::Import { resource } => {
                    compiled.resources.push(CompiledGraphResource { resource, owned_by_graph: false });
                },  
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
        self.passes
            .get(pass)
            .map(|pass| pass.dependencies.iter().map(|&dependency_handle| {
                let dependency = &self.dependencies[dependency_handle];
                let resource = &self.resources[dependency.resource_handle];
                let resource_version = &resource.versions[dependency.resource_version];

                let is_write = dependency.access.read_write_kind() == ReadWriteKind::Write;

                let prev_reads = is_write.then_some(resource_version.read_by.iter().clone()).into_iter().flatten().copied();
                let source_pass = resource.source_pass(dependency.resource_version);

                prev_reads.chain(source_pass)
            }).flatten())
            .into_iter()
            .flatten()

    }
}
