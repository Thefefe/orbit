use std::{sync::Mutex, borrow::Cow, time::Instant};

use ash::vk;
use winit::window::Window;

use crate::render;

use super::{graph::{RenderGraph, CompiledRenderGraph}, DescriptorHandle, TransientResourceRegistry};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuGlobalData {
    screen_size: [u32; 2],
    elapsed_frames: u32,
    elapsed_time: f32,
}

pub struct Frame {
    pub in_flight_fence: vk::Fence,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub command_pool: render::CommandPool,

    global_buffer: render::Buffer,
    
    leftover_resource_registry: TransientResourceRegistry,
}

pub const FRAME_COUNT: usize = 2;

struct RecordSubmitStuff {
    command_pool: render::CommandPool,
    fence: vk::Fence,
}

pub struct Context {
    pub window: Window,

    pub device: render::Device,
    pub descriptors: render::BindlessDescriptors,
    pub swapchain: render::Swapchain,

    pub graph: RenderGraph,
    pub compiled_graph: CompiledRenderGraph,

    pub frames: [Frame; FRAME_COUNT],
    pub frame_index: usize,
    pub elapsed_frames: usize,
    start: Instant,

    record_submit_stuff: Mutex<RecordSubmitStuff>,
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
            });

            Frame {
                in_flight_fence,
                image_available_semaphore,
                render_finished_semaphore,
                command_pool,
                
                global_buffer,
                
                leftover_resource_registry: TransientResourceRegistry::new(),
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
            
            frames,
            frame_index: 0,
            elapsed_frames: 0,
            start: Instant::now(),

            record_submit_stuff,
        }
    }

    pub fn record_and_submit(&self, f: impl FnOnce(&render::CommandRecorder)) {
        puffin::profile_function!();
        let mut record_submit_stuff = self.record_submit_stuff.lock().unwrap();
        record_submit_stuff.command_pool.reset(&self.device);
        let buffer = record_submit_stuff.command_pool.begin_new(&self.device, vk::CommandBufferUsageFlags::empty());
        f(&buffer.record(&self.device, &self.descriptors));
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

            for resource in frame.leftover_resource_registry.resources() {
                resource.destroy(&self.device, &self.descriptors)
            }
        }

        for resource in self.compiled_graph.transient_resource_registry.resources() {
            resource.destroy(&self.device, &self.descriptors);
        }

        self.swapchain.destroy(&self.device);
        self.descriptors.destroy(&self.device);
        self.device.destroy();
    }
}

pub struct FrameContext<'a> {
    pub context: &'a mut Context,
    pub acquired_image: render::AcquiredImage,
    pub acquired_image_handle: render::ResourceHandle,
}

impl std::ops::Deref for FrameContext<'_> {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        self.context
    }
}

impl Context {
    pub fn begin_frame(&mut self) -> FrameContext {
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

        let acquired_image_handle = self.graph.import_resource(
            "swapchain_image".into(),
            render::AnyResourceView::Image(acquired_image.image_view),
            &render::GraphResourceImportDesc {
                initial_access: render::AccessKind::None,
                target_access: render::AccessKind::Present,
                wait_semaphore: Some(frame.image_available_semaphore),
                finish_semaphore: Some(frame.render_finished_semaphore),
            }
        );

        FrameContext {
            context: self,
            acquired_image,
            acquired_image_handle
        }
    }
}

impl FrameContext<'_> {
    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.acquired_image.extent
    }

    pub fn swapchain_format(&self) -> vk::Format {
        self.acquired_image.format
    }

    pub fn get_swapchain_image(&self) -> render::ResourceHandle {
        self.acquired_image_handle
    }

    pub fn get_global_buffer_index(&self) -> u32 {
        self.frames[self.frame_index].global_buffer.descriptor_index.unwrap().to_raw()
    }

    pub fn import_buffer(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        buffer: &render::BufferView,
        desc: &render::GraphResourceImportDesc,
    ) -> render::ResourceHandle {
        self.context.graph.import_resource(name.into(), render::AnyResourceView::Buffer(*buffer), desc)
    }

    pub fn import_image(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        image: &render::ImageView,
        desc: &render::GraphResourceImportDesc,
    ) -> render::ResourceHandle {
        self.context.graph.import_resource(name.into(), render::AnyResourceView::Image(*image), desc)
    }

    pub fn create_transient_buffer(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: render::BufferDesc
    ) -> render::ResourceHandle {
        let name = name.into();
        self.context.graph.add_transient_resource(name, render::AnyResourceDesc::Buffer(desc))
    }
    
    pub fn create_transient_image(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: render::ImageDesc
    ) -> render::ResourceHandle {
        let name = name.into();
        self.context.graph.add_transient_resource(name, render::AnyResourceDesc::Image(desc))
    }

    pub fn add_pass(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        dependencies: &[(render::ResourceHandle, render::AccessKind)],
        f: impl Fn(&render::CommandRecorder, &render::CompiledRenderGraph) + 'static,
    ) -> render::PassHandle {
        let pass = self.context.graph.add_pass(name.into(), Box::new(f));
        for (resource, access) in dependencies.iter().copied() {
            self.context.graph.add_dependency(pass, resource, access);
        }
        pass
    }

    fn record(&mut self) {
        puffin::profile_function!();
        unsafe {
            let screen_size = self.swapchain_extent();
            self.frames[self.frame_index].global_buffer.mapped_ptr
                .unwrap()
                .cast::<GpuGlobalData>()
                .as_ptr()
                .write(GpuGlobalData {
                    screen_size: [screen_size.width, screen_size.height],
                    elapsed_frames: self.context.elapsed_frames as u32,
                    elapsed_time: self.context.start.elapsed().as_secs_f32(),
            })
        };
        let frame = &mut self.context.frames[self.context.frame_index];

        self.context.graph.compile_and_flush(
            &self.context.device,
            &self.context.descriptors,
            &mut self.context.compiled_graph,
            &mut frame.leftover_resource_registry,
        );

        {
            puffin::profile_scope!("command_recording");
            for (batch_index, batch) in self.context.compiled_graph.iter_batches().enumerate() {
                puffin::profile_scope!("batch_record", format!("{batch_index}"));
                let cmd_buffer = frame.command_pool
                    .begin_new(&self.context.device, vk::CommandBufferUsageFlags::empty());

                for semaphore in batch.wait_semaphores {
                    cmd_buffer.wait_semaphore(*semaphore, batch.memory_barrier.src_stage_mask);
                }

                for semaphore in batch.finish_semaphores   {
                    cmd_buffer.signal_semaphore(*semaphore, batch.memory_barrier.dst_stage_mask);
                }
                
                let recorder = cmd_buffer.record(&self.context.device, &self.context.descriptors);
                    
                self.context.descriptors.bind_descriptors(&recorder, vk::PipelineBindPoint::GRAPHICS);
                self.context.descriptors.bind_descriptors(&recorder, vk::PipelineBindPoint::COMPUTE);

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
                    (pass.func)(&recorder, &self.context.compiled_graph);
                    recorder.end_debug_label();
                }

                recorder.barrier(&[], batch.finish_image_barriers, &[]);
            }
        }

        {
            puffin::profile_scope!("command_submit");
            self.context.device.submit(frame.command_pool.buffers(), frame.in_flight_fence);
        }
        {
            puffin::profile_scope!("queue_present");
            self.context.swapchain.queue_present(
                &self.context.device,
                self.acquired_image,
                frame.render_finished_semaphore,
            );
        }

        {
            puffin::profile_scope!("leftover_resource_releases");
            for resource in frame.leftover_resource_registry.resources() {
                resource.destroy(&self.context.device, &self.context.descriptors)
            }
        }
        frame.leftover_resource_registry.clear();
    }
}

impl Drop for FrameContext<'_> {
    fn drop(&mut self) {
        self.record();
        self.context.frame_index = (self.context.frame_index + 1) % FRAME_COUNT;
        self.context.elapsed_frames += 1;
    }
}