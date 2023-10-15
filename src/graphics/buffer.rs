use std::{ptr::NonNull, borrow::Cow, sync::Arc};

use ash::vk;
use gpu_allocator::{vulkan::{AllocationScheme, AllocationCreateDesc}, MemoryLocation};
use crate::graphics;

#[derive(Debug, Clone, Copy)]
pub struct BufferView {
    pub handle: vk::Buffer,
    pub device_address: u64,
    pub descriptor_index: Option<graphics::DescriptorIndex>,
    pub size: u64
}

#[derive(Debug, Clone)]
pub struct BufferRaw {
    pub name: Cow<'static, str>,
    pub(super) buffer_view: graphics::BufferView,
    pub desc: BufferDesc,
    pub alloc_index: graphics::AllocIndex,
    pub mapped_ptr: Option<NonNull<u8>>,
}

unsafe impl Sync for BufferRaw {}
unsafe impl Send for BufferRaw {}

impl std::ops::Deref for BufferRaw {
    type Target = graphics::BufferView;

    fn deref(&self) -> &Self::Target {
        &self.buffer_view
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: MemoryLocation,
}

impl BufferRaw {
    pub(super) fn create_impl(
        device: &graphics::Device,
        name: Cow<'static, str>,
        desc: &BufferDesc,
        preallocated_descriptor_index: Option<graphics::DescriptorIndex>,
    ) -> BufferRaw {
        puffin::profile_function!(&name);
        let create_info = vk::BufferCreateInfo::builder()
            .size(desc.size as u64)
            .usage(desc.usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { device.raw.create_buffer(&create_info, None).unwrap() };

        let requirements = unsafe { device.raw.get_buffer_memory_requirements(handle) };

        let (alloc_index, memory, offset, mapped_ptr) = device.allocate(&AllocationCreateDesc {
            name: &name,
            requirements,
            location: desc.memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });

        unsafe {
            device.raw.bind_buffer_memory(handle, memory, offset).unwrap();
        }

        let descriptor_index = if desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            let index = preallocated_descriptor_index
                .unwrap_or_else(|| device.alloc_descriptor_index());
            device.write_storage_buffer_resource(index, handle);
            Some(index)
        } else {
            None
        };

        device.set_debug_name(handle, &name);

        let device_address = device.get_buffer_address(handle);

        BufferRaw {
            name,
            buffer_view: graphics::BufferView {
                handle,
                device_address,
                descriptor_index,
                size: desc.size as u64,
            },
            alloc_index,
            desc: desc.clone(),
            mapped_ptr,
        }
    }

    pub(super) fn destroy_impl(device: &graphics::Device, buffer: &BufferRaw) {
        puffin::profile_function!(&buffer.name);
        if let Some(index) = buffer.descriptor_index {
            device.free_descriptor_index(index);
        }
        unsafe {
            puffin::profile_scope!("free_vulkan_resources");
            device.raw.destroy_buffer(buffer.handle, None);
        }
        device.deallocate(buffer.alloc_index);
    }
}

#[derive(Clone)]
pub struct Buffer {
    pub _buffer: Option<Arc<graphics::BufferRaw>>,
    pub _device: Arc<graphics::Device>,
}

impl std::ops::Deref for Buffer {
    type Target = graphics::BufferRaw;

    fn deref(&self) -> &Self::Target {
        self._buffer.as_ref().unwrap()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(buffer) = Arc::into_inner(self._buffer.take().unwrap()) {
            BufferRaw::destroy_impl(&self._device, &buffer);
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self._buffer.fmt(f)
    }
}

impl graphics::Context {
    pub fn create_buffer(&self, name: impl Into<Cow<'static, str>>, desc: &BufferDesc) -> Buffer {
        let buffer = BufferRaw::create_impl(&self.device, name.into(), desc, None);
        Buffer { _buffer: Some(Arc::new(buffer)), _device: self.device.clone() }
    }

    pub fn create_buffer_init(&self, name: impl Into<Cow<'static, str>>, desc: &BufferDesc, init: &[u8]) -> Buffer {
        let mut desc = *desc;
        
        if desc.memory_location != MemoryLocation::CpuToGpu {
            desc.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        let buffer = BufferRaw::create_impl(&self.device, name.into(), &desc, None);
        self.immediate_write_buffer(&buffer, init, 0);

        Buffer { _buffer: Some(Arc::new(buffer)), _device: self.device.clone() }
    }

    pub fn immediate_write_buffer(&self, buffer: &graphics::BufferRaw, data: &[u8], offset: usize) {
        puffin::profile_function!();
        
        if data.is_empty() {
            return;
        }

        let copy_size = usize::min(buffer.size as usize - offset, data.len());
        if let Some(mapped_ptr) = buffer.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr.as_ptr().add(offset), copy_size);
            }
        } else {
            let scratch_buffer = BufferRaw::create_impl(
                &self.device,
                "scratch_buffer".into(),
                &BufferDesc {
                    size: copy_size,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    memory_location: MemoryLocation::CpuToGpu,
                },
                None
            );

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

            BufferRaw::destroy_impl(&self.device, &scratch_buffer);
        }
    }
}
