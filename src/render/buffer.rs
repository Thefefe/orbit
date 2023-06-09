use std::ptr::NonNull;

use ash::vk;
use gpu_allocator::{vulkan::{AllocationScheme, AllocationCreateDesc}, MemoryLocation};
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

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc<'a> {
    pub name: &'a str,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: MemoryLocation,
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
        let mut desc = *desc;
        
        if desc.memory_location != MemoryLocation::CpuToGpu {
            desc.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        let buffer = Buffer::create_impl(&self.device, &self.descriptors, &desc);

        let init_size = usize::min(buffer.size as usize, init.len());
        if let Some(mapped_ptr) = buffer.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(init.as_ptr(), mapped_ptr.as_ptr(), init_size);
            }
        } else {
            let scratch_buffer = Buffer::create_impl(&self.device, &self.descriptors, &BufferDesc {
                name: "scratch_buffer",
                size: init_size as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: MemoryLocation::CpuToGpu,
            });

            unsafe {
                std::ptr::copy_nonoverlapping(init.as_ptr(), scratch_buffer.mapped_ptr.unwrap().as_ptr(), init_size);
            }

            self.record_and_submit(|cmd| {
                cmd.copy_buffer(&scratch_buffer, &buffer, &[
                    vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: init_size as u64,
                    }
                ]);
            });

            Buffer::destroy_impl(&self.device, &scratch_buffer);
        }

        buffer
    }

    pub fn immediate_write_buffer(&self, buffer: &render::Buffer, data: &[u8], offset: usize) {
        let copy_size = usize::min(buffer.size as usize - offset, data.len());
        if let Some(mapped_ptr) = buffer.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr.as_ptr(), copy_size);
            }
        } else {
            let scratch_buffer = Buffer::create_impl(&self.device, &self.descriptors, &BufferDesc {
                name: "scratch_buffer",
                size: copy_size as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: MemoryLocation::CpuToGpu,
            });

            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), scratch_buffer.mapped_ptr.unwrap().as_ptr(), copy_size);
            }

            self.record_and_submit(|cmd| {
                cmd.copy_buffer(&scratch_buffer, &buffer, &[
                    vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: offset as u64,
                        size: copy_size as u64,
                    }
                ]);
            });

            Buffer::destroy_impl(&self.device, &scratch_buffer);
        }
    }

    pub fn destroy_buffer(&self, buffer: &Buffer) {
        Buffer::destroy_impl(&self.device, buffer)
    }
}
