use crate::graphics;

use ash::vk;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapchainConfig {
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub frame_count: usize,
    pub image_count: u32,
}

struct SwapchainInner {
    handle: vk::SwapchainKHR,
    images: Vec<graphics::ImageView>,

    config: SwapchainConfig,
    last_frame_index: Option<usize>,
}

impl SwapchainInner {
    fn new(
        device: &graphics::Device,
        old_swapchain: Option<vk::SwapchainKHR>,
        config: SwapchainConfig,
        surface_info: &graphics::SurfaceInfo,
    ) -> Self {
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(device.surface)
            .min_image_count(config.image_count)
            .image_format(config.surface_format.format)
            .image_color_space(config.surface_format.color_space)
            .image_extent(config.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .present_mode(config.present_mode)
            .pre_transform(surface_info.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .clipped(false)
            .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null()));

        let handle = unsafe { device.swapchain_fns.create_swapchain(&swapchain_create_info, None) }.unwrap();

        let images = unsafe { device.swapchain_fns.get_swapchain_images(handle) }.unwrap();
        let images = images
            .into_iter()
            .map(|image| {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };

                let image_view_create_info = vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(config.surface_format.format)
                    .subresource_range(subresource_range);

                let view = unsafe { device.raw.create_image_view(&image_view_create_info, None) }.unwrap();

                let extent = vk::Extent3D {
                    width: config.extent.width,
                    height: config.extent.height,
                    depth: 1,
                };

                graphics::ImageView {
                    handle: image,
                    _descriptor_index: 0,
                    _descriptor_flags: graphics::ImageDescriptorFlags::empty(),
                    subresource_range,
                    format: config.surface_format.format,
                    view,
                    extent,
                }
            })
            .collect();

        Self {
            handle,
            images,
            config,
            last_frame_index: None,
        }
    }

    fn destroy(&self, device: &graphics::Device) {
        unsafe {
            for image in self.images.iter() {
                device.raw.destroy_image_view(image.view, None);
            }

            device.swapchain_fns.destroy_swapchain(self.handle, None);
        }
    }
}

impl graphics::SurfaceInfo {
    pub fn choose_surface_format(&self) -> vk::SurfaceFormatKHR {
        for surface_format in self.formats.iter().copied() {
            if surface_format.format == vk::Format::B8G8R8A8_SRGB
                && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return surface_format;
            }
        }

        self.formats[0]
    }

    pub fn choose_extent(&self, target: vk::Extent2D) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::MAX {
            self.capabilities.current_extent
        } else {
            log::debug!(
                "width_range: {}..{}",
                self.capabilities.min_image_extent.width,
                self.capabilities.min_image_extent.width
            );
            let width = target.width.clamp(
                self.capabilities.min_image_extent.width,
                self.capabilities.max_image_extent.width,
            );
            let height = target.height.clamp(
                self.capabilities.min_image_extent.height,
                self.capabilities.max_image_extent.height,
            );

            vk::Extent2D { width, height }
        }
    }

    fn max_image_count(&self) -> u32 {
        if self.capabilities.max_image_count == 0 {
            u32::MAX
        } else {
            self.capabilities.max_image_count
        }
    }

    pub fn choose_image_count(&self, frame_count: u32) -> u32 {
        u32::max(
            self.capabilities.min_image_count + 1,
            frame_count.min(self.max_image_count()),
        )
    }

    pub fn try_select_present_mode(&self, target_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        target_modes
            .iter()
            .copied()
            .find(|target_mode| self.present_modes.iter().any(|present_mode| present_mode == target_mode))
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }
}

#[derive(Debug, Clone)]
pub struct AcquiredImage {
    pub image: graphics::ImageRaw,
    pub image_index: u32,
    pub suboptimal: bool,
}

impl std::ops::Deref for AcquiredImage {
    type Target = graphics::ImageRaw;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

#[derive(Debug)]
pub struct SwapchainOutOfDate;

pub struct Swapchain {
    inner: SwapchainInner,
    old: VecDeque<SwapchainInner>,

    config: SwapchainConfig,
    surface_info: graphics::SurfaceInfo,
}

impl Swapchain {
    pub fn new(device: &graphics::Device, config: SwapchainConfig) -> Self {
        let surface_info = graphics::SurfaceInfo::new(device);

        Self {
            inner: SwapchainInner::new(device, None, config, &surface_info),
            old: VecDeque::new(),
            config,
            surface_info,
        }
    }

    pub fn resize(&mut self, new_size: [u32; 2]) {
        self.config.extent = vk::Extent2D {
            width: new_size[0],
            height: new_size[1],
        }
    }

    pub fn set_present_mode(&mut self, present_mode: vk::PresentModeKHR) {
        self.config.present_mode = present_mode;
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.config.extent
    }

    pub fn format(&self) -> vk::Format {
        self.config.surface_format.format
    }

    fn recreate_if_needed(&mut self, device: &graphics::Device) {
        if self.config == self.inner.config {
            return;
        }

        self.surface_info.refresh_capabilities(device);
        self.config.extent = self.surface_info.choose_extent(self.config.extent);
        let mut swapchain = SwapchainInner::new(&device, Some(self.inner.handle), self.config, &self.surface_info);

        std::mem::swap(&mut swapchain, &mut self.inner);

        self.old.push_back(swapchain);
    }

    pub fn acquire_image(
        &mut self,
        device: &graphics::Device,
        frame_index: usize, // used for old swapchain lifetime management
        acquired_semaphore: vk::Semaphore,
    ) -> Result<AcquiredImage, SwapchainOutOfDate> {
        puffin::profile_function!();
        self.recreate_if_needed(device);
        
        while let Some(swapchain) = self.old.front() {
            if swapchain.last_frame_index
                .as_ref()
                .map_or(false, |&index| frame_index > index + self.config.image_count as usize) {
                break;
            }
            swapchain.destroy(&device);
            self.old.pop_front();
        }

        let (image_index, suboptimal) = unsafe {
            match device.swapchain_fns.acquire_next_image(
                self.inner.handle,
                u64::MAX,
                acquired_semaphore,
                vk::Fence::null(),
            ) {
                Ok(ok) => ok,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(SwapchainOutOfDate),
                Err(err) => panic!("swapchain error: {err}"),
            }
        };

        self.inner.last_frame_index = Some(frame_index);

        let extent = self.config.extent;

        let image = graphics::ImageRaw {
            name: "swapchain_image".into(),
            full_view: self.inner.images[image_index as usize],
            subresource_views: Vec::new(),
            desc: graphics::ImageDesc {
                ty: graphics::ImageType::Single2D,
                format: self.config.surface_format.format,
                dimensions: [extent.width, extent.height, 1],
                mip_levels: 1,
                samples: graphics::MultisampleCount::None,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
            alloc_index: graphics::AllocIndex::null(),
        };

        Ok(AcquiredImage {
            image,
            image_index,
            suboptimal,
        })
    }

    pub fn queue_present(
        &mut self,
        device: &graphics::Device,
        image: AcquiredImage,
        finished_semaphore: vk::Semaphore,
    ) {
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(&finished_semaphore))
            .swapchains(std::slice::from_ref(&self.inner.handle))
            .image_indices(std::slice::from_ref(&image.image_index));

        unsafe {
            device.swapchain_fns.queue_present(device.queue, &present_info).unwrap();
        }
    }

    pub fn destroy(&self, device: &graphics::Device) {
        for swapchain in self.old.iter() {
            swapchain.destroy(&device);
        }

        self.inner.destroy(&device);
    }
}
