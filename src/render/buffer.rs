use std::ptr::NonNull;

use ash::vk;
use gpu_allocator::vulkan::{AllocationScheme, AllocationCreateDesc};
use crate::render;

pub struct Buffer {
    buffer_view: render::BufferView,
    pub alloc_index: render::AllocIndex,
    pub usage: vk::BufferUsageFlags,
    pub mapped_ptr: Option<NonNull<u8>>,
}

impl std::ops::Deref for Buffer {
    type Target = render::BufferView;

    fn deref(&self) -> &Self::Target {
        &self.buffer_view
    }
}

pub struct BufferDesc<'a> {
    pub name: &'a str,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: gpu_allocator::MemoryLocation,
}

impl Buffer {
    fn create_impl(device: &render::Device, descriptors: &render::BindlessDescriptors, desc: &BufferDesc) -> Buffer {
        let create_info = vk::BufferCreateInfo::builder()
            .size(desc.size)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { device.raw.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe { device.raw.get_buffer_memory_requirements(handle) };

        let (alloc_index, memory, offset, mapped_ptr) = device.allocate(&AllocationCreateDesc {
            name: desc.name,
            requirements,
            location: desc.memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });

        unsafe {
            device.raw.bind_buffer_memory(handle, memory, offset).unwrap();
        }

        let descriptor_index = if desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            Some(descriptors.alloc_buffer_resource(device, handle))
        } else {
            None
        };

        device.set_debug_name(handle, desc.name);

        Buffer {
            buffer_view: render::BufferView {
                handle,
                descriptor_index,
                size: desc.size,
            },
            alloc_index,
            usage: desc.usage,
            mapped_ptr,
        }
    }

    fn destroy_impl(device: &render::Device, buffer: &Buffer) {
        unsafe {
            device.raw.destroy_buffer(buffer.handle, None);
        }
        device.deallocate(buffer.alloc_index);
    }
}

impl render::Context {
    pub fn create_buffer(&self, desc: &BufferDesc) -> Buffer {
        Buffer::create_impl(&self.device, &self.descriptors, desc)
    }

    pub fn create_buffer_init(&self, desc: &BufferDesc, init: &[u8]) -> Buffer {
        let buffer = Buffer::create_impl(&self.device, &self.descriptors, desc);

        if let Some(mapped_ptr) = buffer.mapped_ptr {
            let count = usize::min(buffer.size as usize, init.len());
            unsafe {
                std::ptr::copy_nonoverlapping(init.as_ptr(), mapped_ptr.as_ptr(), count);
            }
        } else {
            todo!()
        }

        buffer
    }

    pub fn destroy_buffer(&self, buffer: &Buffer) {
        Buffer::destroy_impl(&self.device, buffer)
    }
}
