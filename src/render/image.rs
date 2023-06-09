use std::ops::Range;

use crate::render;
use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::{AllocationCreateDesc, AllocationScheme}};
use render::AccessKind;

#[derive(Debug, Clone, Copy)]
pub struct Image {
    pub image_view: render::ImageView,
    alloc_index: render::AllocIndex,
}

impl std::ops::Deref for Image {
    type Target = render::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.image_view
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageDesc<'a> {
    pub name: &'a str,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
}

impl Image {
    fn create_impl(device: &render::Device, descriptors: &render::BindlessDescriptors, desc: &ImageDesc) -> Image {
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
            .tiling(desc.tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let handle = unsafe { device.raw.create_image(&create_info, None).unwrap() };

        let requirements = unsafe { device.raw.get_image_memory_requirements(handle) };

        let (memory, vk_memory, offset, _) = device.allocate(&AllocationCreateDesc {
            name: desc.name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: desc.tiling == vk::ImageTiling::LINEAR,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });

        unsafe {
            device.raw.bind_image_memory(handle, vk_memory, offset).unwrap();
        }

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: desc.aspect,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };

        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(handle)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(desc.format)
            .subresource_range(subresource_range);

        let view = unsafe {
            device.raw.create_image_view(&image_view_create_info, None).unwrap()
        };

        let descriptor_index = if desc.usage.contains(vk::ImageUsageFlags::SAMPLED) {
            Some(descriptors.alloc_image_resource(device, view))
        } else {
            None
        };

        Image {
            image_view: render::ImageView {
                handle,
                descriptor_index,
                view,
                extent: vk::Extent2D { width: desc.width, height: desc.height },
                subresource_range,
            },
            alloc_index: memory,
        }
    }

    fn destroy_impl(device: &render::Device, descriptors: &render::BindlessDescriptors, image: &render::Image) {
        unsafe {
            device.raw.destroy_image_view(image.view, None);
            device.raw.destroy_image(image.handle, None);
        }
        if let Some(descriptor_index) = image.descriptor_index {
            descriptors.free_descriptor_index(descriptor_index);
        } 
    }
}

impl render::Context {
    pub fn create_image(&self, desc: &ImageDesc) -> Image {
        Image::create_impl(&self.device, &self.descriptors, desc)
    }

    pub fn immediate_write_image(
        &self,
        image: &render::Image,
        mip_level: u32,
        layers: Range<u32>,
        prev_access: AccessKind,
        target_access: Option<AccessKind>,
        bytes: &[u8],
        subregion: Option<vk::Rect2D>
    ) {
        let scratch_buffer = self.create_buffer_init(&render::BufferDesc {
            name: "scratch_buffer",
            size: bytes.len() as u64,
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