use std::{ops::Range, borrow::Cow};

use crate::render;
use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::{AllocationCreateDesc, AllocationScheme}};
use render::AccessKind;

#[derive(Debug, Clone)]
pub struct Image {
    pub name: Cow<'static, str>,
    pub image_view: render::ImageView,
    alloc_index: render::AllocIndex,
}

impl std::ops::Deref for Image {
    type Target = render::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.image_view
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageDesc {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub samples: render::MultisampleCount,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
}

impl Image {
    pub(super) fn create_impl(
        device: &render::Device,
        descriptors: &render::BindlessDescriptors,
        name: Cow<'static, str>,
        desc: &ImageDesc
    ) -> Image {
        puffin::profile_function!(&name);
        let create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(1)
            .format(desc.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .samples(desc.samples.to_vk())
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { device.raw.create_image(&create_info, None).unwrap() };

        device.set_debug_name(handle, &name);

        let requirements = unsafe { device.raw.get_image_memory_requirements(handle) };

        let (memory, vk_memory, offset, _) = device.allocate(&AllocationCreateDesc {
            name: &name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });

        unsafe {
            device.raw.bind_image_memory(handle, vk_memory, offset).unwrap();
        }

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: desc.aspect,
            base_mip_level: 0,
            level_count: desc.mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        };

        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(handle)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(desc.format)
            .subresource_range(subresource_range);

        let view = unsafe {
            device.raw.create_image_view(&image_view_create_info, None).unwrap()
        };
        device.set_debug_name(view, &format!("{}_view", name));

        let descriptor_index = if desc.usage.contains(vk::ImageUsageFlags::SAMPLED) {
            let index = descriptors.alloc_image_resource(device, view);
            Some(index)
        } else {
            None
        };

        Image {
            name,
            image_view: render::ImageView {
                handle,
                descriptor_index,
                format: desc.format,
                view,
                extent: vk::Extent2D { width: desc.width, height: desc.height },
                subresource_range,
            },
            alloc_index: memory,
        }
    }

    pub(super) fn destroy_impl(device: &render::Device, descriptors: &render::BindlessDescriptors, image: &render::Image) {
        puffin::profile_function!(&image.name);
        if let Some(descriptor_index) = image.descriptor_index {
            descriptors.free_descriptor_index(descriptor_index);
        }
        unsafe {
            puffin::profile_scope!("free_vulkan_resources");
            device.raw.destroy_image_view(image.view, None);
            device.raw.destroy_image(image.handle, None);
        }
        device.deallocate(image.alloc_index);
    }

    pub fn set_sampler_flags(&mut self, sampler_flags: render::SamplerFlags) {
        if let Some(index) = self.image_view.descriptor_index.as_mut() {
            *index = index.with_sampler(sampler_flags);
        }
    }
}

impl render::Context {
    pub fn create_image(&self, name: impl Into<Cow<'static, str>>, desc: &ImageDesc) -> Image {
        Image::create_impl(&self.device, &self.descriptors, name.into(), desc)
    }

    pub fn immediate_write_image(
        &self,
        image: &render::ImageView,
        mip_level: u32,
        layers: Range<u32>,
        prev_access: AccessKind,
        target_access: Option<AccessKind>,
        bytes: &[u8],
        subregion: Option<vk::Rect2D>
    ) {
        let scratch_buffer = self.create_buffer_init("scratch_buffer", &render::BufferDesc {
            size: bytes.len(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_location: MemoryLocation::CpuToGpu,
        }, bytes);

        let (image_offset, image_extent) = if let Some(subregion) = subregion {
            let vk::Offset2D {x, y} = subregion.offset;
            let vk::Extent2D { width, height } = subregion.extent;
            (
                vk::Offset3D { x, y, z: layers.start as i32 },
                vk::Extent3D { width, height, depth: layers.end }
            )
        } else {
            (
                vk::Offset3D { x: 0, y: 0, z: layers.start as i32 },
                vk::Extent3D { width: image.width(), height: image.height(), depth: layers.end }
            )
        };
        
        self.record_and_submit(|cmd| {
            cmd.barrier(&[], &[render::image_barrier(image, prev_access, AccessKind::TransferWrite)], &[]);

            cmd.copy_buffer_to_image(&scratch_buffer, &image, &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: image.subresource_layers(mip_level, layers),
                image_offset,
                image_extent,
            }]);

            if let Some(target_access) = target_access {
                cmd.barrier(&[], &[render::image_barrier(image, AccessKind::TransferWrite, target_access)], &[]);
            }
        });

        self.destroy_buffer(&scratch_buffer);
    }

    pub fn destroy_image(&self, image: &Image) {
        Image::destroy_impl(&self.device, &self.descriptors, image)
    }
}