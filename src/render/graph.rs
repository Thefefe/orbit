use std::{collections::HashSet, ops::Range};

use ash::vk;

use crate::render;

pub type ResourceHandle = usize;
pub type PassHandle = thunderdome::Index;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Buffer,
    Image,
}

#[derive(Debug, Clone, Copy)]
pub enum AnyResourceView {
    Buffer(render::BufferView),
    Image(render::ImageView),
}

impl AnyResourceView {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceView::Buffer(_) => ResourceKind::Buffer,
            AnyResourceView::Image(_) => ResourceKind::Image,
        }
    }
}

#[derive(Debug)]
pub enum ResourceSource {
    Import,
    Pass { dependency: DependencyHandle },
}

#[derive(Debug)]
pub struct ResourceVersion {
    initial_access: render::AccessKind,
    source: ResourceSource,

    reads: Vec<DependencyHandle>,
}

#[derive(Debug)]
pub struct ResourceData {
    name: String,

    resource: AnyResourceView,

    target_access: render::AccessKind,
    wait_semaphore: Option<vk::Semaphore>,
    finish_semaphore: Option<vk::Semaphore>,

    versions: Vec<ResourceVersion>,
}

#[derive(Debug)]
pub struct PassData {
    pass: Pass,
    dependencies: Vec<DependencyHandle>,
}

#[derive(Debug)]
struct DependencyData {
    access: render::AccessKind,
    pass_handle: PassHandle,
    resource_handle: ResourceHandle,
    resource_version: usize,
}

#[derive(Debug)]
pub struct RenderGraph {
    resources: Vec<ResourceData>,
    passes: thunderdome::Arena<PassData>,
    dependencies: Vec<DependencyData>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphResourceImportDesc {
    pub initial_access: render::AccessKind,
    pub target_access: render::AccessKind,
    pub wait_semaphore: Option<vk::Semaphore>,
    pub finish_semaphore: Option<vk::Semaphore>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            passes: thunderdome::Arena::new(),
            dependencies: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.dependencies.clear();
    }

    fn add_resource(&mut self, resource_data: ResourceData) -> ResourceHandle {
        assert!(resource_data.versions.len() > 0);
        let index = self.resources.len();
        self.resources.push(resource_data);
        index
    }

    pub fn import_resource(
        &mut self,
        name: String,
        resource: AnyResourceView,
        desc: &GraphResourceImportDesc,
    ) -> ResourceHandle {
        self.add_resource(ResourceData {
            name,
            resource,

            target_access: desc.target_access,
            wait_semaphore: desc.wait_semaphore,
            finish_semaphore: desc.finish_semaphore,

            versions: vec![ResourceVersion {
                initial_access: desc.initial_access,
                source: ResourceSource::Import,
                reads: Vec::new(),
            }],
        })
    }

    pub fn add_pass(&mut self, name: String, func: PassFn) -> PassHandle {
        self.passes.insert(PassData {
            pass: Pass { name, func },
            dependencies: Vec::new(),
        })
    }

    pub fn add_dependency(
        &mut self,
        pass_handle: PassHandle,
        resource_handle: ResourceHandle,
        access: render::AccessKind,
    ) -> usize {
        let resource_version = self.resources[resource_handle].versions.len() - 1;

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
                initial_access: access,
                source: ResourceSource::Pass { dependency },
                reads: Vec::new(),
            });
        } else {
            self.resources[resource_handle].versions[resource_version].reads.push(dependency);
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

#[derive(Default)]
pub struct CompiledRenderGraph {
    pub resources: Vec<AnyResourceView>,
    pub passes: Vec<Pass>,
    pub image_barriers: Vec<vk::ImageMemoryBarrier2>,
    pub semaphores: Vec<vk::Semaphore>,
    pub batches: Vec<BatchData>,
}

impl std::fmt::Debug for CompiledRenderGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledRenderGraph")
            .field("resources", &self.resources)
            .field("passes", &self.passes)
            .field("image_barriers", &self.image_barriers)
            .field("semaphores", &self.semaphores)
            .field("batches", &self.batches)
            .finish()
    }
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
        match &self.resources[handle] {
            AnyResourceView::Buffer(buffer) => Some(buffer),
            _ => None
        }
    }

    pub fn get_image(&self, handle: render::ResourceHandle) -> Option<&render::ImageView> {
        match &self.resources[handle] {
            AnyResourceView::Image(image) => Some(image),
            _ => None
        }
    }
}

impl RenderGraph {
    pub fn compile_and_flush(&mut self, compiled: &mut CompiledRenderGraph) {
        puffin::profile_function!();
        compiled.clear();

        let sorted_passes = self.topology_sort(None);

        compiled.resources.extend(self.resources.iter().map(|res| res.resource));

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

            for &pass_slot in passes {
                let Some((_, pass)) = self.passes.remove_by_slot(pass_slot) else { continue };

                // first pass of dependencies, order matters for limiting allocations
                // while preserving data contiguity
                for &dependency in pass.dependencies.iter() {
                    let dependency = &self.dependencies[dependency];
                    let resource_data = &self.resources[dependency.resource_handle];
                    let resource_kind = resource_data.resource.kind();

                    if dependency.resource_version == 0 {
                        if let Some(semaphore) = resource_data.wait_semaphore {
                            batch.wait_semaphore_range.end += 1;
                            compiled.semaphores.push(semaphore);
                        }
                    }

                    let src_access =
                        self.resources[dependency.resource_handle].versions[dependency.resource_version].initial_access;

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
                    } else if let AnyResourceView::Image(image) = &resource_data.resource {
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

                    // 'creator' of the last version of the resource
                    if dependency.access.read_write_kind() == render::ReadWriteKind::Write
                        && dependency.resource_version == resource_data.versions.len() - 2
                    {
                        if let Some(semaphore) = resource_data.finish_semaphore {
                            batch.finish_semaphore_range.end += 1;
                            compiled.semaphores.push(semaphore);
                        }

                        if let AnyResourceView::Image(image) = &resource_data.resource {
                            if resource_data.target_access != render::AccessKind::None &&
                                dependency.access != resource_data.target_access
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
    passes: Vec<u32>,
}

impl SortedPasses {
    fn passes(&self) -> impl Iterator<Item = &[u32]> {
        self.ranges.iter().map(|range| &self.passes[range.clone()])
    }
}

impl RenderGraph {
    pub fn dead_strip(&mut self, output_passes: &[PassHandle]) -> Vec<bool> {
        let mut alive = vec![false; self.passes.len()];

        for pass in output_passes {
            self.walk_alive_check(&mut alive, pass.slot());
        }

        alive
    }

    fn walk_alive_check(&self, alive: &mut [bool], pass_slot: u32) {
        if !alive[pass_slot as usize] {
            alive[pass_slot as usize] = true;

            for pass in self.prev_passes(pass_slot) {
                self.walk_alive_check(alive, pass);
            }
        }
    }

    fn topology_sort(&self, dead_stripped: Option<&[bool]>) -> SortedPasses {
        puffin::profile_function!();
        // TODO: maybe use a better algo,
        // though this seems to be fast enough

        let mut sorted_passes = SortedPasses {
            ranges: Vec::with_capacity(self.passes.len()), // worst case
            passes: Vec::with_capacity(self.passes.len()),
        };

        let mut remainging_passes: HashSet<_> = if let Some(alive) = dead_stripped {
            assert_eq!(alive.len(), self.passes.len());
            (0..self.passes.len() as u32).filter(|&pass| alive[pass as usize]).collect()
        } else {
            (0..self.passes.len() as u32).collect()
        };

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

    fn prev_passes(&self, pass: u32) -> impl Iterator<Item = u32> + '_ {
        self.passes.get_by_slot(pass).unwrap().1.dependencies.iter().filter_map(|&dependency_handle| {
            let handle = self.dependencies[dependency_handle].resource_handle;
            let version = self.dependencies[dependency_handle].resource_version;

            if let ResourceSource::Pass { dependency } = self.resources[handle].versions[version].source {
                Some(self.dependencies[dependency].pass_handle.slot())
            } else {
                None
            }
        })
    }
}
