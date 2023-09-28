use std::{sync::{Mutex, Arc}, borrow::Cow, time::Instant, marker::PhantomData, collections::HashMap};

use ash::vk;
use winit::window::Window;

use crate::graphics;

use super::{graph::{RenderGraph, CompiledRenderGraph, self}, TransientResourceCache, ResourceKind, BatchDependency};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuGlobalData {
    screen_size: [u32; 2],
    elapsed_frames: u32,
    elapsed_time: f32,
}

pub const FRAME_COUNT: usize = 2;
const MAX_TIMESTAMP_COUNT: u32 = 128;

pub struct Frame {
    first_time_use: bool,

    in_flight_fence: vk::Fence,
    image_available_semaphore: graphics::Semaphore,
    render_finished_semaphore: graphics::Semaphore,
    command_pool: graphics::CommandPool,
    global_buffer: graphics::BufferRaw,

    // in_use_transient_resources: TransientResourceCache,
    compiled_graph: CompiledRenderGraph,
    graph_debug_info: GraphDebugInfo,
    timestamp_query_pool: vk::QueryPool,
}

struct RecordSubmitStuff {
    command_pool: graphics::CommandPool, 
    fence: vk::Fence,
}

struct FrameContext {
    acquired_image: graphics::AcquiredImage,
    acquired_image_handle: GraphImageHandle,
}

pub struct Context {
    pub window: Window,

    pub device: Arc<graphics::Device>,
    pub swapchain: graphics::Swapchain,

    pub shader_modules: HashMap<graphics::ShaderSource, vk::ShaderModule>,
    pub raster_pipelines: HashMap<graphics::RasterPipelineDesc, graphics::RasterPipeline>,
    pub compute_pipelines: HashMap<graphics::ShaderSource, graphics::ComputePipeline>,

    pub graph: RenderGraph,
    transient_resource_cache: TransientResourceCache,

    pub frames: [Frame; FRAME_COUNT],
    pub frame_index: usize,
    pub elapsed_frames: usize,
    start: Instant,

    record_submit_stuff: Mutex<RecordSubmitStuff>,

    frame_context: Option<FrameContext>,
}

impl Context {
    pub fn new(window: Window) -> Self {
        let device = graphics::Device::new(&window).expect("failed to create device");
        let device = Arc::new(device);

        let swapchain = {
            let surface_info = &device.gpu.surface_info;

            let window_size = window.inner_size();
            let extent = vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            };

            let surface_format = surface_info.choose_surface_format();

            let image_count = surface_info.choose_image_count(FRAME_COUNT as u32);

            let config = graphics::SwapchainConfig {
                extent,
                present_mode: vk::PresentModeKHR::FIFO,
                surface_format,
                frame_count: FRAME_COUNT,
                image_count,
            };

            graphics::Swapchain::new(&device, config)
        };

        let frames = std::array::from_fn(|frame_index| {
            let in_flight_fence = device.create_fence("in_flight_fence", true);
            let image_available_semaphore = device.create_semaphore("image_available_semaphore");
            let render_finished_semaphore = device.create_semaphore("render_finished_semaphore");

            let command_pool = graphics::CommandPool::new(&device, &format!("frame{frame_index}"));

            let global_buffer = graphics::BufferRaw::create_impl(&device, "global_buffer".into(), &graphics::BufferDesc {
                size: std::mem::size_of::<GpuGlobalData>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: gpu_allocator::MemoryLocation::CpuToGpu,
            }, None);

            let timestamp_query_pool = unsafe {
                let create_info = vk::QueryPoolCreateInfo::builder()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(MAX_TIMESTAMP_COUNT);

                device.raw.create_query_pool(&create_info, None).unwrap()
            };

            Frame {
                first_time_use: true,

                in_flight_fence,
                image_available_semaphore,
                render_finished_semaphore,
                command_pool,
                global_buffer,

                // in_use_transient_resources: TransientResourceCache::new(),
                compiled_graph: CompiledRenderGraph::new(),
                graph_debug_info: GraphDebugInfo::new(),
                timestamp_query_pool,
            }
        });

        let record_submit_stuff = {
            let command_pool = graphics::CommandPool::new(&device, "global");
            let fence = device.create_fence("record_submit_fence", false);

            Mutex::new(RecordSubmitStuff { command_pool, fence })  
        };

        Self {
            window,

            device,
            swapchain,
            
            shader_modules: HashMap::new(),
            raster_pipelines: HashMap::new(),
            compute_pipelines: HashMap::new(),

            graph: RenderGraph::new(),
            transient_resource_cache: TransientResourceCache::new(),
            
            frames,
            frame_index: 0,
            elapsed_frames: 0,
            start: Instant::now(),

            record_submit_stuff,

            frame_context: None,
        }
    }

    pub fn record_and_submit(&self, f: impl FnOnce(&graphics::CommandRecorder)) {
        puffin::profile_function!();
        let mut record_submit_stuff = self.record_submit_stuff.lock().unwrap();
        record_submit_stuff.command_pool.reset(&self.device);
        let buffer = record_submit_stuff.command_pool.begin_new(&self.device);
        let recorder = buffer.record(&self.device, vk::CommandBufferUsageFlags::empty());
        f(&recorder);
        drop(recorder);
        unsafe {
            self.device.raw.reset_fences(&[record_submit_stuff.fence]).unwrap();
        }
        self.device.submit(&record_submit_stuff.command_pool.buffers(), record_submit_stuff.fence);
        unsafe {
            self.device.raw.wait_for_fences(&[record_submit_stuff.fence], false, u64::MAX).unwrap();
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        
        unsafe {
            self.device.raw.device_wait_idle().unwrap();
        }

        let record_submit_stuff = self.record_submit_stuff.lock().unwrap(); 
        unsafe {
            self.device.raw.destroy_fence(record_submit_stuff.fence, None);
        }
        record_submit_stuff.command_pool.destroy(&self.device);
        
        for frame in self.frames.iter_mut() {
            unsafe {
                self.device.raw.destroy_fence(frame.in_flight_fence, None);
                self.device.raw.destroy_semaphore(frame.image_available_semaphore.handle, None);
                self.device.raw.destroy_semaphore(frame.render_finished_semaphore.handle, None);

                self.device.raw.destroy_query_pool(frame.timestamp_query_pool, None);
            }

            frame.command_pool.destroy(&self.device);

            graphics::BufferRaw::destroy_impl(&self.device, &frame.global_buffer);

            for graph::CompiledGraphResource { resource, owned_by_graph } in frame.compiled_graph.resources.drain(..) {
                if owned_by_graph || !resource.is_owned() {
                    graphics::AnyResource::destroy(&self.device, resource);
                }
            }
        }
        
        for resource in self.transient_resource_cache.drain_resources() {
            graphics::AnyResource::destroy(&self.device, resource);
        }

        for shader_module in self.shader_modules.values().copied() {
            unsafe {
                self.device.raw.destroy_shader_module(shader_module, None);
            }
        }

        for raster_pipeline in self.raster_pipelines.values().copied() {
            unsafe {
                self.device.raw.destroy_pipeline(raster_pipeline.handle, None);
            }
        }

        for compute_pipeline in self.compute_pipelines.values().copied() {
            unsafe {
                self.device.raw.destroy_pipeline(compute_pipeline.handle, None);
            }
        }

        self.swapchain.destroy(&self.device);
    }
}

impl Context {
    pub fn begin_frame(&mut self) {
        assert!(self.frame_context.is_none(), "frame already began");
        puffin::profile_function!();
        let frame = &mut self.frames[self.frame_index];

        unsafe {
            puffin::profile_scope!("fence_wait");
            self.device.raw.wait_for_fences(&[frame.in_flight_fence], false, u64::MAX).unwrap();
            self.device.raw.reset_fences(&[frame.in_flight_fence]).unwrap();
            
            if !frame.first_time_use {
                let timestamp_count = frame.graph_debug_info.timestamp_count;
                self.device.raw.get_query_pool_results::<u64>(
                    frame.timestamp_query_pool,
                    0,
                    timestamp_count,
                    &mut frame.graph_debug_info.timestamp_data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                ).unwrap();
            } else {
                frame.first_time_use = false;
            }
        }

        frame.command_pool.reset(&self.device);

        self.swapchain.resize(self.window.inner_size().into());

        let acquired_image = self
            .swapchain
            .acquire_image(&mut self.device, self.frame_index, frame.image_available_semaphore.handle)
            .unwrap();

        self.graph.clear();

        assert!(self.transient_resource_cache.is_empty());
        for graph::CompiledGraphResource { resource, owned_by_graph } in frame.compiled_graph.resources.drain(..) {
            if !owned_by_graph && resource.is_owned() { // swapchain image
                continue;
            }
            self.transient_resource_cache.insert(resource.desc(), resource);
        }

        let acquired_image_handle = self.graph.add_resource(graph::GraphResourceData {
            name: "swapchain_image".into(),
            
            source: graph::ResourceSource::Import {
                resource: acquired_image.image.clone().into(),
            },
            descriptor_index: None,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::Present,
            wait_semaphore: Some(frame.image_available_semaphore.clone()),
            finish_semaphore: Some(frame.render_finished_semaphore.clone()),
            versions: vec![],
        });

        self.frame_context = Some(FrameContext {
            acquired_image,
            acquired_image_handle: GraphHandle { resource_index:acquired_image_handle, _phantom: PhantomData },
        });        
    }
}

#[derive(Debug)]
pub struct GraphHandle<T: ?Sized> {
    pub resource_index: graphics::GraphResourceIndex,
    pub _phantom: PhantomData<T>,
}

impl<T> GraphHandle<T> {
    pub fn uninit() -> Self {
        Self { resource_index: 0, _phantom: PhantomData }
    }
}

impl<T> Clone for GraphHandle<T> {
    fn clone(&self) -> Self {
        Self {
            resource_index: self.resource_index.clone(),
            _phantom: self._phantom.clone()
        }
    }
}

impl<T> Copy for GraphHandle<T> { }

pub type GraphBufferHandle = GraphHandle<graphics::BufferRaw>;
pub type GraphImageHandle = GraphHandle<graphics::ImageRaw>;

impl From<GraphHandle<graphics::ImageRaw>> for egui::TextureId {
    fn from(value: GraphHandle<graphics::ImageRaw>) -> Self {
        Self::User(value.resource_index as u64)
    }
}

pub struct PassBuilder<'a> {
    context: &'a mut Context,
    name: Cow<'static, str>,
    dependencies: Vec<(graphics::GraphResourceIndex, graphics::AccessKind)>,
}

pub trait IntoGraphDependency {
    fn into_dependency(self) -> (graphics::GraphResourceIndex, graphics::AccessKind);
}

impl<T> IntoGraphDependency for (graphics::GraphHandle<T>, graphics::AccessKind) {
    fn into_dependency(self) -> (graphics::GraphResourceIndex, graphics::AccessKind) {
        (self.0.resource_index, self.1)
    }
}

impl<'a> PassBuilder<'a> {
    pub fn with_dependency<T>(mut self, handle: graphics::GraphHandle<T>, access: graphics::AccessKind) -> Self {
        self.dependencies.push((handle.resource_index, access));
        self
    }

    pub fn with_dependencies<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoGraphDependency
    {
        self.dependencies.extend(iter.into_iter().map(|item| item.into_dependency()));
        self
    }

    pub fn render(self, f: impl Fn(&graphics::CommandRecorder, &graphics::CompiledRenderGraph) + 'static) -> graphics::GraphPassIndex {
        let pass = self.context.graph.add_pass(self.name, Box::new(f));
        for (resource, access) in self.dependencies.iter().copied() {
            self.context.graph.add_dependency(pass, resource, access);
        }
        pass
    }
}

impl Context {
    pub fn frame_index(&self) -> usize {
        self.frame_index
    }
    
    pub fn swapchain_extent(&self) -> vk::Extent3D {
        self.frame_context.as_ref().unwrap().acquired_image.extent
    }

    pub fn swapchain_format(&self) -> vk::Format {
        self.frame_context.as_ref().unwrap().acquired_image.format
    }

    pub fn get_swapchain_image(&self) -> graphics::GraphImageHandle {
        self.frame_context.as_ref().unwrap().acquired_image_handle
    }

    pub fn get_global_buffer_index(&self) -> u32 {
        self.frames[self.frame_index].global_buffer.descriptor_index.unwrap()
    }

    pub fn get_resource_descriptor_index<T>(&self, handle: GraphHandle<T>) -> Option<graphics::DescriptorIndex> {
        self.graph.resources[handle.resource_index].descriptor_index
    }

    pub fn import<R: graphics::RenderResource>(
        &mut self,
        resource: R
    ) -> GraphHandle<R::RawResource> {
        let resource = resource.into();
        let descriptor_index = resource.as_ref().descriptor_index();
        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name: resource.as_ref().name().clone(),
            source: graph::ResourceSource::Import { resource },
            descriptor_index,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle { resource_index, _phantom: PhantomData }
    }

    pub fn import_with<R: graphics::RenderResource>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        resource: R,
        desc: graphics::GraphResourceImportDesc,
    ) -> GraphHandle<R::RawResource> {
        let name = name.into();
        let resource = resource.into();
        let descriptor_index = resource.as_ref().descriptor_index();
        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
            source: graph::ResourceSource::Import { resource },
            descriptor_index,

            initial_access: desc.initial_access,
            target_access: desc.target_access,
            wait_semaphore: desc.wait_semaphore,
            finish_semaphore: desc.finish_semaphore,
            versions: vec![],
        });

        GraphHandle { resource_index, _phantom: PhantomData }
    }

    pub fn transient_storage_data(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        data: &[u8], 
    ) -> GraphHandle<graphics::BufferRaw> {
        let name = name.into();
        let buffer_desc = graphics::BufferDesc {
            size: data.len(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: gpu_allocator::MemoryLocation::CpuToGpu,
        };
        let desc = graphics::AnyResourceDesc::Buffer(buffer_desc);
        let cache = self.transient_resource_cache
            .get_by_descriptor(&desc)
            .unwrap_or_else(|| graphics::AnyResource::create_owned(
                &self.device,
                name.clone(),
                &desc,
                None
            ));

        let graphics::AnyResourceRef::Buffer(buffer) = cache.as_ref() else { unreachable!() };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.mapped_ptr.unwrap().as_ptr(), data.len());
        }
        let descriptor_index = buffer.descriptor_index;

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
         
            source: graph::ResourceSource::Create { desc, cache: Some(cache) },
            descriptor_index,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle { resource_index, _phantom: PhantomData }
    }

    pub fn create_transient<R: graphics::OwnedRenderResource>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: R::Desc,
    ) -> GraphHandle<R> {
        let name = name.into();
        let desc = desc.into();
        let mut cache = self.transient_resource_cache.get_by_descriptor(&desc);
        let descriptor_index = match &cache {
            Some(cache) => cache.as_ref().descriptor_index(),
            None if desc.needs_descriptor_index() => Some(self.device.alloc_descriptor_index()),
            _ => None
        };

        if let Some(cache) = cache.as_mut() {
            match cache {
                graphics::AnyResource::BufferOwned(buffer) => {
                    buffer.name = name.clone();
                    self.device.set_debug_name(buffer.handle, name.as_ref());
                },
                graphics::AnyResource::ImageOwned(image)  => {
                    image.name = name.clone();
                    self.device.set_debug_name(image.handle, name.as_ref());
                },
                _ => unimplemented!()
            }
        }

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
            
            source: graph::ResourceSource::Create { desc, cache },
            descriptor_index,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle { resource_index, _phantom: PhantomData }
    }

    pub fn add_pass(&mut self, name: impl Into<Cow<'static, str>>) -> PassBuilder {
        PassBuilder { 
            context: self,
            name: name.into(),
            dependencies: Vec::new(),
        }
    }

    pub fn end_frame(&mut self) {
        puffin::profile_function!();
        let frame_context = self.frame_context.take().unwrap();
        
        unsafe {
            let screen_size = frame_context.acquired_image.extent;
            self.frames[self.frame_index].global_buffer.mapped_ptr
                .unwrap()
                .cast::<GpuGlobalData>()
                .as_ptr()
                .write(GpuGlobalData {
                    screen_size: [screen_size.width, screen_size.height],
                    elapsed_frames: self.elapsed_frames as u32,
                    elapsed_time: self.start.elapsed().as_secs_f32(),
            })
        };
        let frame = &mut self.frames[self.frame_index];

        frame.graph_debug_info.clear();
        
        self.graph.compile_and_flush(
            &self.device,
            &mut frame.compiled_graph,
        );

        frame.graph_debug_info.batch_infos.resize(frame.compiled_graph.batches.len(), GraphBatchDebugInfo::default());

        {
            puffin::profile_scope!("command_recording");

            let mut timestamp_cursor: u32 = 0;

            let mut image_barriers = Vec::new();
            let mut memory_barrier;
            for (batch_index, batch) in frame.compiled_graph.iter_batches().enumerate() {
                puffin::profile_scope!("batch_record", format!("{batch_index}"));

                memory_barrier = vk::MemoryBarrier2::default();
                image_barriers.clear();
                for BatchDependency { resource_index: resoure_index, src_access, dst_access } in batch.begin_dependencies.iter().copied() {
                    let resource = &frame.compiled_graph.resources[resoure_index].resource.as_ref();

                    if resource.kind() != ResourceKind::Image || src_access.image_layout() == dst_access.image_layout() {
                        if src_access.read_write_kind() == graphics::ReadWriteKind::Read &&
                           dst_access.read_write_kind() == graphics::ReadWriteKind::Read {
                            continue;
                        }
                        memory_barrier.src_stage_mask |= src_access.stage_mask();
                        if src_access.read_write_kind() == graphics::ReadWriteKind::Write {
                            memory_barrier.src_access_mask |= src_access.access_mask();
                        }

                        memory_barrier.dst_stage_mask |= dst_access.stage_mask();
                        if !memory_barrier.src_access_mask.is_empty() {
                            memory_barrier.dst_access_mask |= dst_access.access_mask();
                        }
                    } else if let graphics::AnyResourceRef::Image(image) = &resource {
                        image_barriers.push(graphics::image_barrier(image, src_access, dst_access));
                    }
                }

                let cmd_buffer = frame.command_pool.begin_new(&self.device);

                for (semaphore, stage) in batch.wait_semaphores {
                    cmd_buffer.wait_semaphore(semaphore.handle, *stage);
                }

                for (semaphore, stage) in batch.signal_semaphores {
                    cmd_buffer.signal_semaphore(semaphore.handle, *stage);
                }
                
                let recorder = cmd_buffer.record(&self.device, vk::CommandBufferUsageFlags::empty());

                if batch_index == 0 {
                    recorder.reset_query_pool(frame.timestamp_query_pool, 0..MAX_TIMESTAMP_COUNT);
                }

                if graphics::is_memory_barrier_not_useless(&memory_barrier) {
                    recorder.barrier(&[], &image_barriers, &[memory_barrier]);
                } else {
                    recorder.barrier(&[], &image_barriers, &[]);
                }
                
                frame.graph_debug_info.batch_infos[batch_index].timestamp_start_index = timestamp_cursor;
                frame.graph_debug_info.batch_infos[batch_index].timestamp_end_index = timestamp_cursor + 1;
                timestamp_cursor += 2;

                recorder.write_query(
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    frame.timestamp_query_pool,
                    frame.graph_debug_info.batch_infos[batch_index].timestamp_start_index,
                );

                for pass in batch.passes {
                    recorder.begin_debug_label(&pass.name, None);
                    (pass.func)(&recorder, &frame.compiled_graph);
                    recorder.end_debug_label();
                }

                recorder.write_query(
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    frame.timestamp_query_pool,
                    frame.graph_debug_info.batch_infos[batch_index].timestamp_end_index,
                );

                image_barriers.clear();
                for BatchDependency { resource_index: resoure_index, src_access, dst_access } in batch.finish_dependencies.iter().copied() {
                    if let graphics::AnyResourceRef::Image(image) = frame.compiled_graph.resources[resoure_index].resource.as_ref() {
                        image_barriers.push(graphics::image_barrier(image, src_access, dst_access));
                    }
                }
                recorder.barrier(&[], &image_barriers, &[]);
            }

            frame.graph_debug_info.timestamp_count = timestamp_cursor;
        }

        {
            puffin::profile_scope!("command_submit");
            self.device.submit(frame.command_pool.buffers(), frame.in_flight_fence);
        }
        {
            puffin::profile_scope!("queue_present");
            self.swapchain.queue_present(
                &self.device,
                frame_context.acquired_image,
                frame.render_finished_semaphore.handle,
            );
        }

        {
            puffin::profile_scope!("leftover_resource_releases");
            for resource in self.transient_resource_cache.drain_resources() {
                graphics::AnyResource::destroy(&self.device, resource);
            }
            self.transient_resource_cache.clear();
        }

        self.frame_index = (self.frame_index + 1) % FRAME_COUNT;
        self.elapsed_frames += 1;
    }
}

impl Context {
    pub fn graph_debugger(&self, egui_ctx: &egui::Context) -> bool {
        let mut open = true;
        egui::Window::new("rendergraph debugger")
            .open(&mut open)
            .show(egui_ctx, |ui| self.draw_graph_info(ui));
        open
    }

    fn draw_graph_info(&self, ui: &mut egui::Ui) {
        let timestamp_period = self.device.gpu.properties.properties10.limits.timestamp_period;
        let graph = &self.frames[self.frame_index].compiled_graph;
        let debug_info = &self.frames[self.frame_index].graph_debug_info;
        for (i, batch) in graph.iter_batches().enumerate() {
            let delta_ns = debug_info.timestamp_delta(i, timestamp_period);
            let delta_ms = delta_ns / 1_000_000.0;
            egui::CollapsingHeader::new(format!("batch {i} ({delta_ms:.2} ms)")).id_source(i).show(ui, |ui| {
                egui::CollapsingHeader::new(format!("wait_semaphores ({})", batch.wait_semaphores.len()))
                    .id_source([i, 0])
                    .show(ui, |ui| {
                        for (semaphore, stage) in batch.wait_semaphores {
                            ui.label(format!("{semaphore:?}, {stage:?}"));
                        }
                    });
                
                egui::CollapsingHeader::new(format!("begin_dependencies ({})", batch.begin_dependencies.len()))
                    .id_source([i, 1])
                    .show(ui, |ui| {
                        for (j, dependency) in batch.begin_dependencies.iter().enumerate() {
                            let resource = &graph.resources[dependency.resource_index].resource.as_ref();
                            egui::CollapsingHeader::new(resource.name().as_ref()).id_source([i, 1, j]).show(ui, |ui| {
                                ui.label(format!("src_access: {:?}", dependency.src_access));
                                ui.label(format!("dst_access: {:?}", dependency.dst_access));
                            });
                        }
                    });

                egui::CollapsingHeader::new(format!("passes ({})", batch.passes.len()))
                    .id_source([i, 2])
                    .show(ui, |ui| {
                        for pass in batch.passes {
                            ui.label(pass.name.as_ref());
                        }
                    });
                
                egui::CollapsingHeader::new(format!("finish_dependencies ({})", batch.finish_dependencies.len()))
                    .id_source([i, 3])
                    .show(ui, |ui| {
                        for (j, dependency) in batch.finish_dependencies.iter().enumerate() {
                            let resource = &graph.resources[dependency.resource_index].resource.as_ref();
                            egui::CollapsingHeader::new(resource.name().as_ref()).id_source([i, 3, j]).show(ui, |ui| {
                                ui.label(format!("src_access: {:?}", dependency.src_access));
                                ui.label(format!("dst_access: {:?}", dependency.dst_access));
                            });
                        }
                    });

                    
                egui::CollapsingHeader::new(format!("signal_semaphores ({})", batch.signal_semaphores.len()))
                    .id_source([i, 4])
                    .show(ui, |ui| {
                        for (semaphore, stage) in batch.signal_semaphores {
                            ui.label(format!("{semaphore:?}, {stage:?}"));
                        }
                    });
            });
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphBatchDebugInfo {
    pub timestamp_start_index: u32,
    pub timestamp_end_index: u32,
}

pub struct GraphDebugInfo {
    pub batch_infos: Vec<GraphBatchDebugInfo>,
    pub timestamp_data: [u64; MAX_TIMESTAMP_COUNT as usize],
    pub timestamp_count: u32,
}

impl GraphDebugInfo {
    pub fn new() -> Self {
        Self {
            batch_infos: Vec::new(),
            timestamp_data: [0; MAX_TIMESTAMP_COUNT as usize],
            timestamp_count: 0,
        }       
    }

    pub fn clear(&mut self) {
        self.batch_infos.clear();
    }
    
    pub fn timestamp_delta(&self, batch_index: usize, timestamp_period: f32) -> f32 {
        let GraphBatchDebugInfo { timestamp_start_index, timestamp_end_index, ..} = self.batch_infos[batch_index];
        let delta = self.timestamp_data[timestamp_end_index as usize] - self.timestamp_data[timestamp_start_index as usize];
        delta as f32 * timestamp_period
    }
}