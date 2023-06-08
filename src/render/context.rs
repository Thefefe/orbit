use ash::vk;
use winit::window::Window;

use crate::render;

use super::graph::{RenderGraph, CompiledRenderGraph};

pub struct Frame {
    pub in_flight_fence: vk::Fence,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub command_pool: render::CommandPool,
}

const FRAME_COUNT: usize = 2;

pub struct Context {
    pub window: Window,

    pub device: render::Device,
    pub descriptors: render::BindlessDescriptors,
    pub swapchain: render::Swapchain,

    pub graph: RenderGraph,
    pub compiled_graph: CompiledRenderGraph,

    pub frames: [Frame; FRAME_COUNT],
    pub frame_index: usize,
}

pub struct ContextDesc {
    pub present_mode: vk::PresentModeKHR,
}

impl Context {
    pub fn new(window: Window, desc: &ContextDesc) -> Self {
        let device = render::Device::new(&window).expect("failed to create device");

        let descriptors = render::BindlessDescriptors::new(&device, &[]);

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
                image_count,
            };

            log::info!("swapchain config: {config:#?}");

            render::Swapchain::new(&device, config)
        };

        let frames = std::array::from_fn(|_| {
            let in_flight_fence = device.create_fence("in_flight_fence", true);
            let image_available_semaphore = device.create_semaphore("image_available_semaphore");
            let render_finished_semaphore = device.create_semaphore("render_finished_semaphore");

            let command_pool = render::CommandPool::new(&device);

            Frame {
                in_flight_fence,
                image_available_semaphore,
                render_finished_semaphore,

                command_pool,
            }
        });

        Self {
            window,

            device,
            descriptors,
            swapchain,
            
            graph: RenderGraph::new(),
            compiled_graph: CompiledRenderGraph::new(),
            
            frames,
            frame_index: 0,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.device_wait_idle().unwrap();
        }
        
        for frame in self.frames.iter() {
            unsafe {
                self.device.raw.destroy_fence(frame.in_flight_fence, None);
                self.device.raw.destroy_semaphore(frame.image_available_semaphore, None);
                self.device.raw.destroy_semaphore(frame.render_finished_semaphore, None);
            }

            frame.command_pool.destroy(&self.device);
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

impl Context {
    pub fn begin_frame(&mut self) -> FrameContext {
        let frame = &mut self.frames[self.frame_index];

        unsafe {
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
            "swapchain_image".to_string(),
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
    pub fn get_swapchain_image(&self) -> render::ResourceHandle {
        self.acquired_image_handle
    }

    pub fn import_buffer(
        &mut self,
        name: impl Into<String>,
        buffer: &render::BufferView,
        desc: &render::GraphResourceImportDesc,
    ) -> render::ResourceHandle {
        self.context.graph.import_resource(name.into(), render::AnyResourceView::Buffer(*buffer), desc)
    }

    pub fn add_pass(
        &mut self,
        name: impl Into<String>,
        dependencies: &[(render::ResourceHandle, render::AccessKind)],
        f: impl Fn(&render::CommandRecorder, &render::CompiledRenderGraph) + 'static,
    ) -> render::PassHandle {
        let pass = self.context.graph.add_pass(name.into(), Box::new(f));
        for (resource, access) in dependencies.iter().copied() {
            self.context.graph.add_dependency(pass, resource, access);
        }
        pass
    }
}

impl Drop for FrameContext<'_> {
    fn drop(&mut self) {
        self.context.graph.flush(&mut self.context.compiled_graph);

        let frame = &mut self.context.frames[self.context.frame_index];

        for batch in self.context.compiled_graph.iter_batches() {
            let cmd_buffer = frame.command_pool
                .begin_new(&self.context.device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            for semaphore in batch.wait_semaphores {
                cmd_buffer.wait_semaphore(*semaphore, batch.memory_barrier.src_stage_mask);
            }

            for semaphore in batch.finish_semaphores   {
                cmd_buffer.signal_semaphore(*semaphore, batch.memory_barrier.dst_stage_mask);
            }

            let recorder = cmd_buffer.record(&self.context.device, &self.context.descriptors);

            self.context.descriptors.bind_descriptors(&recorder, vk::PipelineBindPoint::GRAPHICS);

            recorder.barrier(&[], batch.begin_image_barriers, &[batch.memory_barrier]);

            for pass in batch.passes {
                pass(&recorder, &self.context.compiled_graph);
            }

            recorder.barrier(&[], batch.finish_image_barriers, &[]);
        }

        self.context.device.submit(frame.command_pool.buffers(), frame.in_flight_fence);
        self.context.swapchain.queue_present(
            &self.context.device,
            self.acquired_image,
            frame.render_finished_semaphore,
        );

        self.context.frame_index = (self.context.frame_index + 1) % FRAME_COUNT;
    }
}