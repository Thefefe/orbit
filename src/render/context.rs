use ash::vk;
use winit::window::Window;

use crate::vulkan;

pub struct Frame {
    pub in_flight_fence: vk::Fence,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub command_pool: vulkan::CommandPool,
}

const FRAME_COUNT: usize = 2;

pub struct Context {
    pub window: Window,
    pub device: vulkan::Device,
    pub descriptors: vulkan::BindlessDescriptors,
    pub swapchain: vulkan::Swapchain,
    pub frames: [Frame; FRAME_COUNT],
    pub frame_index: usize,
}

pub struct ContextDesc {
    pub present_mode: vk::PresentModeKHR,
}

impl Context {
    pub fn new(window: Window, desc: &ContextDesc) -> Self {
        let device = vulkan::Device::new(&window).expect("failed to create device");

        let descriptors = vulkan::BindlessDescriptors::new(&device, &[]);

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

            let config = vulkan::SwapchainConfig {
                extent,
                present_mode,
                surface_format,
                image_count,
            };

            vulkan::Swapchain::new(&device, config)
        };

        let frames = std::array::from_fn(|_| {
            let in_flight_fence = device.create_fence("in_flight_fence", true);
            let image_available_semaphore = device.create_semaphore("image_available_semaphore");
            let render_finished_semaphore = device.create_semaphore("render_finished_semaphore");

            let command_pool = vulkan::CommandPool::new(&device);

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
    pub acquired_image: vulkan::AcquiredImage,
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
            .acquire_image(&self.device, self.frame_index, frame.image_available_semaphore)
            .unwrap();

        FrameContext {
            context: self,
            acquired_image,
        }
    }
}

impl Drop for FrameContext<'_> {
    fn drop(&mut self) {
        let frame = &mut self.context.frames[self.context.frame_index];

        self.context.device.submit(frame.command_pool.buffers(), frame.in_flight_fence);
        self.context.swapchain.queue_present(
            &self.context.device,
            self.acquired_image,
            frame.render_finished_semaphore,
        );

        self.context.frame_index = (self.context.frame_index + 1) % FRAME_COUNT;
    }
}