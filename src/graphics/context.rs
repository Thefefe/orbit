use std::{borrow::Cow, collections::HashMap, marker::PhantomData, sync::Arc, time::Instant};

use ash::vk;
use parking_lot::Mutex;
use rayon::prelude::*;
use winit::window::Window;

use crate::graphics::{self, QueueType};

use super::{
    graph::{self, CompiledRenderGraph, RenderGraph},
    BatchDependency, ResourceKind, TransientResourceCache,
};

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
    transfer_finished_semaphore: graphics::Semaphore,
    uses_async_transfer: bool,

    command_pools: Vec<Mutex<graphics::CommandPool>>,
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

struct BufferCopy {
    dst_buffer: vk::Buffer,
    region: vk::BufferCopy,
}

struct TransferQueue {
    command_pool: graphics::CommandPool,
    fence: vk::Fence,
    staging_buffer: graphics::BufferRaw,
    staging_buffer_offset: usize,
    buffer_copies: Vec<BufferCopy>,
}

impl TransferQueue {
    const STAGING_BUFFER_SIZE: usize = 64 * 1024 * 1024;
    fn new(device: &graphics::Device) -> Self {
        let command_pool = graphics::CommandPool::new(device, "trasfer_command_pool", QueueType::AsyncTransfer);
        let fence = device.create_fence("transfer_queue_fence", true);
        let staging_buffer = graphics::BufferRaw::create_impl(
            device,
            "transfer_queue_staging_buffer".into(),
            &graphics::BufferDesc {
                size: Self::STAGING_BUFFER_SIZE,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: gpu_allocator::MemoryLocation::CpuToGpu,
            },
            None
        );

        Self {
            command_pool,
            fence,
            staging_buffer,
            staging_buffer_offset: 0,
            buffer_copies: Vec::new(),
        }
    }

    fn destroy(&self, device: &graphics::Device) {
        self.command_pool.destroy(device);
        unsafe {
            device.raw.destroy_fence(self.fence, None);
        }
        graphics::BufferRaw::destroy_impl(device, &self.staging_buffer);
    }

    fn queue_write_buffer(&mut self, buffer: &graphics::BufferView, offset: usize, data: &[u8]) {
        if data.len() == 0 { return; }
        puffin::profile_function!();

        let buffer_copy = BufferCopy {
            dst_buffer: buffer.handle,
            region: vk::BufferCopy {
                src_offset: self.staging_buffer_offset as u64,
                dst_offset: offset as u64,
                size: data.len() as u64,
            },
        };
        self.buffer_copies.push(buffer_copy);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.staging_buffer.mapped_ptr.unwrap().as_ptr().add(self.staging_buffer_offset),
                data.len()
            );
        }
        self.staging_buffer_offset += data.len();
    }

    fn submit(&mut self, device: &graphics::Device, semaphore: Option<vk::Semaphore>) {
        puffin::profile_function!();

        if self.buffer_copies.is_empty() { return; }
        unsafe {
            puffin::profile_scope!("wait_for_trasnfer_fence");
            device.raw.wait_for_fences(&[self.fence], false, u64::MAX).unwrap();
        }
        unsafe { device.raw.reset_fences(&[self.fence]).unwrap(); }
        self.command_pool.reset(device);

        let cmd = self.command_pool.begin_new(device);
        if let Some(semaphore) = semaphore {
            cmd.signal_semaphore(semaphore, vk::PipelineStageFlags2::TRANSFER);
        }
        let cmd = cmd.record(device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        for buffer_copy in self.buffer_copies.drain(..) {
            cmd.copy_buffer(
                self.staging_buffer.handle,
                buffer_copy.dst_buffer,
                std::slice::from_ref(&buffer_copy.region)
            );
        }
        drop(cmd);

        let submit_info = self.command_pool.buffers()[0].submit_info();

        device.queue_submit(QueueType::AsyncTransfer, &[submit_info], self.fence);
        self.staging_buffer_offset = 0;
    }
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

    submit_infos: Vec<vk::SubmitInfo2>,
    record_submit_stuff: Mutex<RecordSubmitStuff>,
    transfer_queue: Mutex<TransferQueue>,

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

        let frames = std::array::from_fn(|_frame_index| {
            let in_flight_fence = device.create_fence("in_flight_fence", true);
            let image_available_semaphore = device.create_semaphore("image_available_semaphore");
            let render_finished_semaphore = device.create_semaphore("render_finished_semaphore");
            let transfer_finished_semaphore = device.create_semaphore("transfer_finished_semaphore");

            // let command_pool = graphics::CommandPool::new(
            //     &device,
            //     &format!("command_pool_frame{frame_index}"),
            //     QueueType::Graphics
            // );

            let global_buffer = graphics::BufferRaw::create_impl(
                &device,
                "global_buffer".into(),
                &graphics::BufferDesc {
                    size: std::mem::size_of::<GpuGlobalData>(),
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    memory_location: gpu_allocator::MemoryLocation::CpuToGpu,
                },
                None,
            );

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
                transfer_finished_semaphore,
                uses_async_transfer: false,

                // command_pool,
                command_pools: (0..rayon::current_num_threads())
                    .map(|i| {
                        Mutex::new(graphics::CommandPool::new(
                            &device,
                            &format!("command_pool_thread_{i}"),
                            QueueType::Graphics,
                        ))
                    })
                    .collect(),
                global_buffer,

                // in_use_transient_resources: TransientResourceCache::new(),
                compiled_graph: CompiledRenderGraph::new(),
                graph_debug_info: GraphDebugInfo::new(),
                timestamp_query_pool,
            }
        });

        let record_submit_stuff = {
            let command_pool = graphics::CommandPool::new(&device, "global", QueueType::Graphics);
            let fence = device.create_fence("record_submit_fence", false);

            Mutex::new(RecordSubmitStuff { command_pool, fence })
        };

        let transfer_queue = Mutex::new(TransferQueue::new(&device));

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

            submit_infos: Vec::new(),
            record_submit_stuff,
            transfer_queue,

            frame_context: None,
        }
    }

    pub fn record_and_submit(&self, f: impl FnOnce(&graphics::CommandRecorder)) {
        puffin::profile_function!();
        let mut record_submit_stuff = self.record_submit_stuff.lock();
        record_submit_stuff.command_pool.reset(&self.device);
        let buffer = record_submit_stuff.command_pool.begin_new(&self.device);
        let recorder = buffer.record(&self.device, vk::CommandBufferUsageFlags::empty());
        f(&recorder);
        drop(recorder);
        unsafe {
            self.device.raw.reset_fences(&[record_submit_stuff.fence]).unwrap();
        }
        let submit_info = record_submit_stuff.command_pool.buffers()[0].submit_info();
        self.device.queue_submit(QueueType::Graphics, &[submit_info], record_submit_stuff.fence);
        unsafe {
            self.device.raw.wait_for_fences(&[record_submit_stuff.fence], false, u64::MAX).unwrap();
        }
    }

    pub fn queue_write_buffer(&self, buffer: &graphics::BufferView, offset: usize, data: &[u8]) {
        self.transfer_queue.lock().queue_write_buffer(buffer, offset, data);
    }

    pub fn submit_pending(&mut self) {
        self.transfer_queue.get_mut().submit(&self.device, None);
        self.graph.dont_wait_semaphores.insert(self.frames[self.frame_index].transfer_finished_semaphore.handle);
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

        let record_submit_stuff = self.record_submit_stuff.get_mut();
        unsafe {
            self.device.raw.destroy_fence(record_submit_stuff.fence, None);
        }
        record_submit_stuff.command_pool.destroy(&self.device);

        self.transfer_queue.get_mut().destroy(&self.device);

        for frame in self.frames.iter_mut() {
            unsafe {
                self.device.raw.destroy_fence(frame.in_flight_fence, None);
                self.device.raw.destroy_semaphore(frame.image_available_semaphore.handle, None);
                self.device.raw.destroy_semaphore(frame.render_finished_semaphore.handle, None);
                self.device.raw.destroy_semaphore(frame.transfer_finished_semaphore.handle, None);

                self.device.raw.destroy_query_pool(frame.timestamp_query_pool, None);
            }

            // frame.command_pool.destroy(&self.device);
            for command_pool in frame.command_pools.iter() {
                command_pool.lock().destroy(&self.device);
            }

            graphics::BufferRaw::destroy_impl(&self.device, &frame.global_buffer);

            for graph::CompiledGraphResource {
                resource,
                owned_by_graph,
            } in frame.compiled_graph.resources.drain(..)
            {
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
                self.device
                    .raw
                    .get_query_pool_results::<u64>(
                        frame.timestamp_query_pool,
                        0,
                        timestamp_count,
                        &mut frame.graph_debug_info.timestamp_data,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .unwrap();
            } else {
                frame.first_time_use = false;
            }
        }

        // frame.command_pool.reset(&self.device);
        for command_pool in frame.command_pools.iter_mut() {
            command_pool.get_mut().reset(&self.device);
        }

        self.swapchain.resize(self.window.inner_size().into());

        let acquired_image = self
            .swapchain
            .acquire_image(
                &mut self.device,
                self.frame_index,
                frame.image_available_semaphore.handle,
            )
            .unwrap();

        self.graph.clear();

        assert!(self.transient_resource_cache.is_empty());
        for graph::CompiledGraphResource {
            resource,
            owned_by_graph,
        } in frame.compiled_graph.resources.drain(..)
        {
            if !owned_by_graph && resource.is_owned() {
                // swapchain image
                continue;
            }
            self.transient_resource_cache.insert(resource);
        }

        let acquired_image_handle = self.graph.add_resource(graph::GraphResourceData {
            name: "swapchain_image".into(),

            source: graph::ResourceSource::Import {
                resource: acquired_image.image.clone().into(),
            },
            descriptor_index: None,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::Present,
            initial_queue: Some(QueueType::Graphics),
            target_queue: None,
            
            wait_semaphore: Some(frame.image_available_semaphore.clone()),
            finish_semaphore: Some(frame.render_finished_semaphore.clone()),
            versions: vec![],
        });

        self.frame_context = Some(FrameContext {
            acquired_image,
            acquired_image_handle: GraphHandle {
                resource_index: acquired_image_handle,
                _phantom: PhantomData,
            },
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
        Self {
            resource_index: 0,
            _phantom: PhantomData,
        }
    }
}

impl<T> Clone for GraphHandle<T> {
    fn clone(&self) -> Self {
        Self {
            resource_index: self.resource_index.clone(),
            _phantom: self._phantom.clone(),
        }
    }
}

impl<T> Copy for GraphHandle<T> {}

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
        I::Item: IntoGraphDependency,
    {
        self.dependencies.extend(iter.into_iter().map(|item| item.into_dependency()));
        self
    }

    pub fn render(
        self,
        f: impl Fn(&graphics::CommandRecorder, &graphics::CompiledRenderGraph) + Send + Sync + 'static,
    ) -> graphics::GraphPassIndex {
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

    pub fn import<R: graphics::RenderResource>(&mut self, resource: R) -> GraphHandle<R::RawResource> {
        let resource = resource.into();
        let descriptor_index = resource.as_ref().descriptor_index();
        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name: resource.as_ref().name().clone(),
            source: graph::ResourceSource::Import { resource },
            descriptor_index,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            initial_queue: Some(QueueType::Graphics),
            target_queue: Some(QueueType::Graphics),
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
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
            initial_queue: desc.initial_queue,
            target_queue: desc.target_queue,
            wait_semaphore: desc.wait_semaphore,
            finish_semaphore: desc.finish_semaphore,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
    }

    pub fn transient_storage_data(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        data: &[u8],
    ) -> GraphHandle<graphics::BufferRaw> {
        let name: Cow<'static, str> = name.into();
        let buffer_desc = graphics::BufferDesc {
            size: data.len(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: gpu_allocator::MemoryLocation::GpuOnly,
        };
        let desc = graphics::AnyResourceDesc::Buffer(buffer_desc);
        let (mut cache, cache_needs_rename) = self.transient_resource_cache.get(&name, &desc).unwrap_or_else(|| {
            (
                graphics::AnyResource::create_owned(&self.device, name.clone(), &desc, None),
                false,
            )
        });

        if cache_needs_rename {
            cache.rename(&self.device, name.clone());
        }

        let graphics::AnyResourceRef::Buffer(buffer) = cache.as_ref() else {
            unreachable!()
        };
        
        self.transfer_queue.get_mut().queue_write_buffer(&buffer, 0, data);
        self.graph.dont_wait_semaphores.remove(&self.frames[self.frame_index].transfer_finished_semaphore.handle);
        self.frames[self.frame_index].uses_async_transfer = true;

        let descriptor_index = buffer.descriptor_index;

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,

            source: graph::ResourceSource::Create {
                desc,
                cache: Some(cache),
            },
            descriptor_index,

            initial_queue: None,
            target_queue: None,
            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            wait_semaphore: Some(self.frames[self.frame_index].transfer_finished_semaphore.clone()),
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
    }

    pub fn create_transient<R: graphics::OwnedRenderResource>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: R::Desc,
    ) -> GraphHandle<R> {
        let name: Cow<'static, str> = name.into();
        let desc = desc.into();
        let cache = if let Some((mut cache, needs_rename)) = self.transient_resource_cache.get(&name, &desc) {
            if needs_rename {
                cache.rename(&self.device, name.clone());
            }

            Some(cache)
        } else {
            None
        };
        let descriptor_index = match &cache {
            Some(cache) => cache.as_ref().descriptor_index(),
            None if desc.needs_descriptor_index() => Some(self.device.alloc_descriptor_index()),
            _ => None,
        };

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,

            source: graph::ResourceSource::Create { desc, cache },
            descriptor_index,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            initial_queue: None,
            target_queue: None,
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn create_transient_buffer(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: graphics::BufferDesc,
    ) -> GraphBufferHandle {
        self.create_transient(name, desc)
    }

    #[inline(always)]
    pub fn create_transient_image(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: graphics::ImageDesc,
    ) -> GraphImageHandle {
        self.create_transient(name, desc)
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
            self.frames[self.frame_index]
                .global_buffer
                .mapped_ptr
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

        if frame.uses_async_transfer {
            frame.uses_async_transfer = false;
            self.transfer_queue.get_mut().submit(&self.device, Some(frame.transfer_finished_semaphore.handle));
        }

        frame.graph_debug_info.clear();

        self.graph.compile_and_flush(&self.device, &mut frame.compiled_graph);

        frame
            .graph_debug_info
            .batch_infos
            .resize(frame.compiled_graph.batches.len(), GraphBatchDebugInfo::default());
        frame.graph_debug_info.timestamp_count = frame.compiled_graph.batches.len() as u32 * 2;

        // let commands = frame.compiled_graph
        //     .iter_batches()
        //     .zip(frame.graph_debug_info.batch_infos.iter_mut())
        //     .enumerate()
        //     .map(|(batch_index, (batch_ref, batch_debug_info))| record_batch(
        //         &self.device,
        //         &frame.compiled_graph,
        //         &mut frame.command_pool,
        //         frame.timestamp_query_pool,
        //         batch_index,
        //         batch_ref,
        //         batch_debug_info
        //     ));

        let batch_cmd_indices: Vec<(usize, usize)> = {
            puffin::profile_scope!("command_recording");
            frame
                .compiled_graph
                .batches
                .par_iter()
                .zip(frame.graph_debug_info.batch_infos.par_iter_mut())
                .enumerate()
                .map_init(
                    || frame.command_pools[rayon::current_thread_index().unwrap()].lock(),
                    |command_pool, (batch_index, (batch_data, batch_debug_info))| {
                        let batch_ref = frame.compiled_graph.get_batch_ref(&batch_data);

                        let command_index = record_batch(
                            &self.device,
                            &frame.compiled_graph,
                            command_pool,
                            frame.timestamp_query_pool,
                            batch_index,
                            batch_ref,
                            batch_debug_info,
                        );

                        (rayon::current_thread_index().unwrap(), command_index)
                    },
                )
                .collect()
        };

        {
            puffin::profile_scope!("command_submit");
            self.submit_infos.clear();
            self.submit_infos.extend(
                batch_cmd_indices.iter().copied().map(|(pool_index, command_buffer_index)| {
                    frame.command_pools[pool_index].get_mut().buffers()[command_buffer_index].submit_info()
                }),
            );
            self.device.queue_submit(QueueType::Graphics, &self.submit_infos, frame.in_flight_fence);
            self.submit_infos.clear();
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

fn record_batch(
    device: &graphics::Device,
    compiled_graph: &graph::CompiledRenderGraph,
    command_pool: &mut graphics::CommandPool,
    timestamp_query_pool: vk::QueryPool,
    batch_index: usize,
    batch_ref: graph::BatchRef,
    batch_debug_info: &mut GraphBatchDebugInfo,
) -> usize {
    puffin::profile_scope!("batch_record", format!("{batch_index}"));

    let mut memory_barrier = vk::MemoryBarrier2::default();
    let mut image_barriers = Vec::new();

    for BatchDependency {
        resource_index: resoure_index,
        src_access,
        dst_access,
    } in batch_ref.begin_dependencies.iter().copied()
    {
        let resource = compiled_graph.resources[resoure_index].resource.as_ref();

        if resource.kind() != ResourceKind::Image || src_access.image_layout() == dst_access.image_layout() {
            if src_access.read_write_kind() == graphics::ReadWriteKind::Read
                && dst_access.read_write_kind() == graphics::ReadWriteKind::Read
            {
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

    let cmd_buffer = command_pool.begin_new(device);

    for (semaphore, stage) in batch_ref.wait_semaphores {
        cmd_buffer.wait_semaphore(semaphore.handle, *stage);
    }

    for (semaphore, stage) in batch_ref.signal_semaphores {
        cmd_buffer.signal_semaphore(semaphore.handle, *stage);
    }

    let recorder = cmd_buffer.record(device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    if batch_index == 0 {
        recorder.reset_query_pool(timestamp_query_pool, 0..MAX_TIMESTAMP_COUNT);
    }

    if graphics::is_memory_barrier_not_useless(&memory_barrier) {
        recorder.barrier(&[], &image_barriers, &[memory_barrier]);
    } else {
        recorder.barrier(&[], &image_barriers, &[]);
    }

    batch_debug_info.timestamp_start_index = batch_index as u32 * 2;
    batch_debug_info.timestamp_end_index = batch_index as u32 * 2 + 1;

    recorder.write_query(
        vk::PipelineStageFlags2::TOP_OF_PIPE,
        timestamp_query_pool,
        batch_debug_info.timestamp_start_index,
    );

    for pass in batch_ref.passes {
        recorder.begin_debug_label(&pass.name, None);
        (pass.func)(&recorder, compiled_graph);
        recorder.end_debug_label();
    }

    recorder.write_query(
        vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        timestamp_query_pool,
        batch_debug_info.timestamp_end_index,
    );

    image_barriers.clear();
    for BatchDependency {
        resource_index: resoure_index,
        src_access,
        dst_access,
    } in batch_ref.finish_dependencies.iter().copied()
    {
        if let graphics::AnyResourceRef::Image(image) = compiled_graph.resources[resoure_index].resource.as_ref() {
            image_barriers.push(graphics::image_barrier(image, src_access, dst_access));
        }
    }
    recorder.barrier(&[], &image_barriers, &[]);

    recorder.command_buffer_index()
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

                egui::CollapsingHeader::new(format!("passes ({})", batch.passes.len())).id_source([i, 2]).show(
                    ui,
                    |ui| {
                        for pass in batch.passes {
                            ui.label(pass.name.as_ref());
                        }
                    },
                );

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
        let GraphBatchDebugInfo {
            timestamp_start_index,
            timestamp_end_index,
            ..
        } = self.batch_infos[batch_index];
        let delta =
            self.timestamp_data[timestamp_end_index as usize] - self.timestamp_data[timestamp_start_index as usize];
        delta as f32 * timestamp_period
    }
}
