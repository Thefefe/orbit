use std::{sync::Mutex, borrow::Cow, time::Instant, marker::PhantomData};

use ash::vk;
use winit::window::Window;

use crate::render;

use super::{graph::{RenderGraph, CompiledRenderGraph, self}, TransientResourceCache, RenderResource};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuGlobalData {
    screen_size: [u32; 2],
    elapsed_frames: u32,
    elapsed_time: f32,
}

pub struct Frame {
    in_flight_fence: vk::Fence,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    command_pool: render::CommandPool,
    global_buffer: render::Buffer,

    in_use_transient_resources: TransientResourceCache,
}

pub const FRAME_COUNT: usize = 2;

struct RecordSubmitStuff {
    command_pool: render::CommandPool,
    fence: vk::Fence,
}

struct FrameContext {
    acquired_image: render::AcquiredImage,
    acquired_image_handle: GraphImageHandle,
}

pub struct Context {
    pub window: Window,

    pub device: render::Device,
    pub descriptors: render::BindlessDescriptors,
    pub swapchain: render::Swapchain,

    pub graph: RenderGraph,
    pub compiled_graph: CompiledRenderGraph,
    transient_resource_cache: TransientResourceCache,

    pub frames: [Frame; FRAME_COUNT],
    pub frame_index: usize,
    pub elapsed_frames: usize,
    start: Instant,

    record_submit_stuff: Mutex<RecordSubmitStuff>,

    frame_context: Option<FrameContext>,
}

pub struct ContextDesc {
    pub present_mode: vk::PresentModeKHR,
}

impl Context {
    pub fn new(window: Window, desc: &ContextDesc) -> Self {
        let device = render::Device::new(&window).expect("failed to create device");
        let descriptors = render::BindlessDescriptors::new(&device);

        let swapchain = {
            let surface_info = &device.gpu.surface_info;

            let window_size = window.inner_size();
            let extent = vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            };

            let present_mode = surface_info.try_select_present_mode(&[desc.present_mode]);
            if present_mode != desc.present_mode {
                log::warn!(
                    "'{:?}' isn't supported, selected: {:?}",
                    desc.present_mode,
                    present_mode
                );
            } else {
                log::info!("selected present mode: {present_mode:?}");
            }


            let surface_format = surface_info.choose_surface_format();

            let image_count = surface_info.choose_image_count(FRAME_COUNT as u32);

            let config = render::SwapchainConfig {
                extent,
                present_mode,
                surface_format,
                frame_count: FRAME_COUNT,
                image_count,
            };

            render::Swapchain::new(&device, config)
        };

        let frames = std::array::from_fn(|frame_index| {
            let in_flight_fence = device.create_fence("in_flight_fence", true);
            let image_available_semaphore = device.create_semaphore("image_available_semaphore");
            let render_finished_semaphore = device.create_semaphore("render_finished_semaphore");

            let command_pool = render::CommandPool::new(&device, &format!("frame{frame_index}"));

            let global_buffer = render::Buffer::create_impl(&device, &descriptors, "global_buffer".into(), &render::BufferDesc {
                size: std::mem::size_of::<GpuGlobalData>(),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_location: gpu_allocator::MemoryLocation::CpuToGpu,
            }, None);

            Frame {
                in_flight_fence,
                image_available_semaphore,
                render_finished_semaphore,
                command_pool,
                global_buffer,

                in_use_transient_resources: TransientResourceCache::new(),
            }
        });

        let record_submit_stuff = {
            let command_pool = render::CommandPool::new(&device, "global");
            let fence = device.create_fence("record_submit_fence", false);

            Mutex::new(RecordSubmitStuff { command_pool, fence })  
        };

        Self {
            window,

            device,
            descriptors,
            swapchain,
            
            graph: RenderGraph::new(),
            compiled_graph: CompiledRenderGraph::new(),
            transient_resource_cache: TransientResourceCache::new(),
            
            frames,
            frame_index: 0,
            elapsed_frames: 0,
            start: Instant::now(),

            record_submit_stuff,

            frame_context: None,
        }
    }

    pub fn record_and_submit(&self, f: impl FnOnce(&render::CommandRecorder)) {
        puffin::profile_function!();
        let mut record_submit_stuff = self.record_submit_stuff.lock().unwrap();
        record_submit_stuff.command_pool.reset(&self.device);
        let buffer = record_submit_stuff.command_pool.begin_new(&self.device, vk::CommandBufferUsageFlags::empty());
        let recorder = buffer.record(&self.device, &self.descriptors);
        self.descriptors.bind_descriptors(&recorder, vk::PipelineBindPoint::GRAPHICS);
        self.descriptors.bind_descriptors(&recorder, vk::PipelineBindPoint::COMPUTE);
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
        
        for frame in self.frames.iter() {
            unsafe {
                self.device.raw.destroy_fence(frame.in_flight_fence, None);
                self.device.raw.destroy_semaphore(frame.image_available_semaphore, None);
                self.device.raw.destroy_semaphore(frame.render_finished_semaphore, None);
            }

            frame.command_pool.destroy(&self.device);

            render::Buffer::destroy_impl(&self.device, &self.descriptors, &frame.global_buffer);
            
            for resource in frame.in_use_transient_resources.resources() {
                resource.destroy(&self.device, &self.descriptors);
            }
        }
        
        for resource in self.transient_resource_cache.resources() {
            resource.destroy(&self.device, &self.descriptors)
        }


        self.swapchain.destroy(&self.device);
        self.descriptors.destroy(&self.device);
        self.device.destroy();
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
        }

        frame.command_pool.reset(&self.device);

        self.swapchain.resize(self.window.inner_size().into());

        let acquired_image = self
            .swapchain
            .acquire_image(&mut self.device, self.frame_index, frame.image_available_semaphore)
            .unwrap();

        self.graph.clear();

        assert!(self.transient_resource_cache.is_empty());
        std::mem::swap(&mut self.transient_resource_cache, &mut frame.in_use_transient_resources);

        let acquired_image_handle = self.graph.add_resource(graph::GraphResourceData {
            name: "swapchain_image".into(),
            
            source: graph::ResourceSource::Import { view: render::AnyResourceView::Image(acquired_image.image_view) },
            descriptor_index: None,

            initial_access: render::AccessKind::None,
            target_access: render::AccessKind::Present,
            wait_semaphore: Some(frame.image_available_semaphore),
            finish_semaphore: Some(frame.render_finished_semaphore),
            versions: vec![],
        });

        self.frame_context = Some(FrameContext {
            acquired_image,
            acquired_image_handle: GraphHandle { resource_index:acquired_image_handle, _phantom: PhantomData },
        });        
    }
}

#[derive(Debug)]
pub struct GraphHandle<T> {
    pub resource_index: render::GraphResourceIndex,
    pub _phantom: PhantomData<T>,
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

pub type GraphBufferHandle = GraphHandle<render::Buffer>;
pub type GraphImageHandle = GraphHandle<render::Image>;

impl From<GraphHandle<render::Image>> for egui::TextureId {
    fn from(value: GraphHandle<render::Image>) -> Self {
        Self::User(value.resource_index as u64)
    }
}

pub struct PassBuilder<'a> {
    context: &'a mut Context,
    name: Cow<'static, str>,
    dependencies: Vec<(render::GraphResourceIndex, render::AccessKind)>,
}

pub trait IntoGraphDependency {
    fn into_dependency(self) -> (render::GraphResourceIndex, render::AccessKind);
}

impl<T> IntoGraphDependency for (render::GraphHandle<T>, render::AccessKind) {
    fn into_dependency(self) -> (render::GraphResourceIndex, render::AccessKind) {
        (self.0.resource_index, self.1)
    }
}

impl<'a> PassBuilder<'a> {
    pub fn with_dependency<T>(mut self, handle: render::GraphHandle<T>, access: render::AccessKind) -> Self {
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

    pub fn render(self, f: impl Fn(&render::CommandRecorder, &render::CompiledRenderGraph) + 'static) -> render::GraphPassIndex {
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

    pub fn get_swapchain_image(&self) -> render::GraphImageHandle {
        self.frame_context.as_ref().unwrap().acquired_image_handle
    }

    pub fn get_global_buffer_index(&self) -> u32 {
        self.frames[self.frame_index].global_buffer.descriptor_index.unwrap()
    }

    pub fn get_transient_resource_descriptor_index<T>(&self, handle: GraphHandle<T>) -> Option<render::DescriptorIndex> {
        self.graph.resources[handle.resource_index].descriptor_index
    }

    pub fn import_buffer(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        buffer: &render::BufferView,
        desc: &render::GraphResourceImportDesc,
    ) -> GraphHandle<render::Buffer> {
        let name = name.into();
        let view = render::AnyResourceView::Buffer(*buffer);
        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
            source: graph::ResourceSource::Import { view },
            descriptor_index: buffer.descriptor_index,

            initial_access: desc.initial_access,
            target_access: desc.target_access,
            wait_semaphore: desc.wait_semaphore,
            finish_semaphore: desc.finish_semaphore,
            versions: vec![],
        });

        GraphHandle { resource_index, _phantom: PhantomData }
    }

    pub fn import_image(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        image: &render::ImageView,
        desc: &render::GraphResourceImportDesc,
    ) -> GraphHandle<render::Image> {
        let name = name.into();
        let view = render::AnyResourceView::Image(*image);
        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
            source: graph::ResourceSource::Import { view },
            descriptor_index: image.descriptor_index,

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
    ) -> GraphHandle<render::Buffer> {
        let name = name.into();
        let buffer_desc = render::BufferDesc {
            size: data.len(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: gpu_allocator::MemoryLocation::CpuToGpu,
        };
        let desc = render::AnyResourceDesc::Buffer(buffer_desc);
        let cache = self.transient_resource_cache
            .get_by_descriptor(&desc)
            .unwrap_or_else(|| render::AnyResource::Buffer(render::Buffer::create_impl(
                &self.device,
                &self.descriptors,
                name.clone(),
                &buffer_desc,
                None
            )));
        let buffer = cache.get_buffer().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.mapped_ptr.unwrap().as_ptr(), data.len());
        }
        let descriptor_index = buffer.descriptor_index;

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
         
            source: graph::ResourceSource::Create { desc, cache: Some(cache) },
            descriptor_index,

            initial_access: render::AccessKind::None,
            target_access: render::AccessKind::None,
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle { resource_index, _phantom: PhantomData }
    }

    pub fn create_transient_buffer(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        buffer_desc: render::BufferDesc
    ) -> GraphHandle<render::Buffer> {
        let name = name.into();
        let desc = render::AnyResourceDesc::Buffer(buffer_desc);
        let cache = self.transient_resource_cache.get_by_descriptor(&desc);
        let descriptor_index = match &cache {
            Some(cache) => cache.descriptor_index(),
            None if buffer_desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) => Some(self.descriptors.alloc_buffer_index()),
            _ => None
        };

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
         
            source: graph::ResourceSource::Create { desc, cache },
            descriptor_index,

            initial_access: render::AccessKind::None,
            target_access: render::AccessKind::None,
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle { resource_index, _phantom: PhantomData }
    }
    
    pub fn create_transient_image(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        image_desc: render::ImageDesc
    ) -> GraphHandle<render::Image> {
        let name = name.into();
        let desc = render::AnyResourceDesc::Image(image_desc);
        let cache = self.transient_resource_cache.get_by_descriptor(&desc);
        let descriptor_index = match &cache {
            Some(cache) => cache.descriptor_index(),
            None if image_desc.usage.contains(vk::ImageUsageFlags::SAMPLED) => Some(self.descriptors.alloc_buffer_index()),
            _ => None
        };

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
         
            source: graph::ResourceSource::Create { desc, cache },
            descriptor_index,
            initial_access: render::AccessKind::None,
            target_access: render::AccessKind::None,
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

        self.graph.compile_and_flush(
            &self.device,
            &self.descriptors,
            &mut self.compiled_graph,
            &mut frame.in_use_transient_resources,
        );

        {
            puffin::profile_scope!("command_recording");
            for (batch_index, batch) in self.compiled_graph.iter_batches().enumerate() {
                puffin::profile_scope!("batch_record", format!("{batch_index}"));
                let cmd_buffer = frame.command_pool
                    .begin_new(&self.device, vk::CommandBufferUsageFlags::empty());

                for semaphore in batch.wait_semaphores {
                    cmd_buffer.wait_semaphore(*semaphore, batch.memory_barrier.src_stage_mask);
                }

                for semaphore in batch.finish_semaphores   {
                    cmd_buffer.signal_semaphore(*semaphore, batch.memory_barrier.dst_stage_mask);
                }
                
                let recorder = cmd_buffer.record(&self.device, &self.descriptors);
                    
                self.descriptors.bind_descriptors(&recorder, vk::PipelineBindPoint::GRAPHICS);
                self.descriptors.bind_descriptors(&recorder, vk::PipelineBindPoint::COMPUTE);

                if batch.memory_barrier.src_stage_mask != vk::PipelineStageFlags2::TOP_OF_PIPE
                || batch.memory_barrier.dst_stage_mask != vk::PipelineStageFlags2::BOTTOM_OF_PIPE
                || batch.memory_barrier.src_access_mask | batch.memory_barrier.dst_access_mask != vk::AccessFlags2::NONE
                {
                    recorder.barrier(&[], batch.begin_image_barriers, &[batch.memory_barrier]);
                } else {
                    recorder.barrier(&[], batch.begin_image_barriers, &[]);
                };

                for pass in batch.passes {
                    recorder.begin_debug_label(&pass.name, None);
                    (pass.func)(&recorder, &self.compiled_graph);
                    recorder.end_debug_label();
                }

                recorder.barrier(&[], batch.finish_image_barriers, &[]);
            }
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
                frame.render_finished_semaphore,
            );
        }

        {
            puffin::profile_scope!("leftover_resource_releases");
            for resource in self.transient_resource_cache.resources() {
                resource.destroy(&self.device, &self.descriptors)
            }
            self.transient_resource_cache.clear();
        }

        self.frame_index = (self.frame_index + 1) % FRAME_COUNT;
        self.elapsed_frames += 1;
    }
}