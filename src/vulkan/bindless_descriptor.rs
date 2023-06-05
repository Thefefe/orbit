use std::sync::{
    atomic::{self, AtomicU32},
    Mutex,
};

use crate::vulkan;
use ash::vk;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuBindingContants {
    indices: [RawDescriptorIndex; 32],
}

impl GpuBindingContants {
    pub fn new() -> Self {
        Self {
            indices: [RawDescriptorIndex(0); 32],
        }
    }

    pub fn set(&mut self, bindings: impl IntoIterator<Item = RawDescriptorIndex>) -> &[RawDescriptorIndex] {
        let mut len = 0;
        for binding in bindings {
            self.indices[len] = binding;
            len += 1;
        }

        &self.indices[0..len]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DescriptorTableType {
    Buffer,
    Image,
}

impl DescriptorTableType {
    fn all_types() -> impl Iterator<Item = Self> {
        [Self::Buffer, Self::Image].into_iter()
    }

    fn set_index(self) -> u32 {
        self as u32
    }

    fn from_set_index(set_index: u32) -> Self {
        match set_index {
            0 => Self::Buffer,
            1 => Self::Image,
            _ => panic!("invalid set index"),
        }
    }

    fn to_vk(self) -> vk::DescriptorType {
        match self {
            DescriptorTableType::Buffer => vk::DescriptorType::STORAGE_BUFFER,
            DescriptorTableType::Image => vk::DescriptorType::SAMPLED_IMAGE,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DescriptorTableType::Buffer => "buffer",
            DescriptorTableType::Image => "sampled_image",
        }
    }

    fn max_count(self, device: &vulkan::Device) -> u32 {
        let props = &device.gpu.properties.properties12;
        match self {
            DescriptorTableType::Buffer => u32::min(
                props.max_descriptor_set_update_after_bind_storage_buffers,
                props.max_per_stage_descriptor_update_after_bind_storage_buffers,
            ),
            DescriptorTableType::Image => u32::min(
                props.max_descriptor_set_update_after_bind_sampled_images,
                props.max_per_stage_descriptor_update_after_bind_sampled_images,
            ),
        }
    }
}

struct IndexAllocator {
    index_counter: AtomicU32,
    freed_indices: Mutex<Vec<u32>>,
}

impl IndexAllocator {
    pub fn new(start: u32) -> Self {
        Self {
            index_counter: AtomicU32::new(start),
            freed_indices: Mutex::new(Vec::new()),
        }
    }

    pub fn alloc(&self) -> u32 {
        if let Some(free_index) = self.freed_indices.lock().unwrap().pop() {
            free_index
        } else {
            self.index_counter.fetch_add(1, atomic::Ordering::Relaxed)
        }
    }

    pub fn free(&self, index: u32) {
        self.freed_indices.lock().unwrap().push(index);
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RawDescriptorIndex(u32);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Zeroable, bytemuck::Pod)]
pub struct BufferDescriptorIndex(u32);

pub trait DescriptorHandle {
    fn to_raw(self) -> RawDescriptorIndex;
}

impl DescriptorHandle for BufferDescriptorIndex {
    fn to_raw(self) -> RawDescriptorIndex {
        RawDescriptorIndex(self.0)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ImageDescriptorIndex(u32);

impl DescriptorHandle for ImageDescriptorIndex {
    fn to_raw(self) -> RawDescriptorIndex {
        RawDescriptorIndex(self.0)
    }
}

pub struct BindlessDescriptors {
    descriptor_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,

    pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    global_index_allocator: IndexAllocator,

    immutable_sampler_count: usize,
}

impl BindlessDescriptors {
    pub fn new(device: &vulkan::Device, immutable_samplers: &[vk::Sampler]) -> Self {
        let descriptor_layouts: Vec<_> = DescriptorTableType::all_types()
            .map(|desc_type| {
                let mut descriptor_binding_flags = vec![
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                ];

                let mut set = vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: desc_type.to_vk(),
                    descriptor_count: desc_type.max_count(&device),
                    stage_flags: vk::ShaderStageFlags::ALL,
                    p_immutable_samplers: std::ptr::null(),
                }];

                if desc_type == DescriptorTableType::Image && !immutable_samplers.is_empty() {
                    descriptor_binding_flags.push(vk::DescriptorBindingFlags::empty());

                    // Set texture binding start at the end of the immutable samplers.
                    set[0].binding = immutable_samplers.len() as u32;
                    set.push(vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        descriptor_count: immutable_samplers.len() as u32,
                        stage_flags: vk::ShaderStageFlags::ALL,
                        p_immutable_samplers: immutable_samplers.as_ptr(),
                    });
                }

                let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                    .binding_flags(&descriptor_binding_flags);

                let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&set)
                    .flags(
                        vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                        // vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT,
                    )
                    .push_next(&mut ext_flags);

                let layout = unsafe { device.raw.create_descriptor_set_layout(&layout_create_info, None) }.unwrap();
                device.set_debug_name(layout, &format!("bindless_{}_layout", desc_type.name()));
                layout
            })
            .collect();

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: std::mem::size_of::<GpuBindingContants>() as u32,
        };

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap() };
        device.set_debug_name(pipeline_layout, "bindless_pipeline_layout");

        let pool_sizes: Vec<_> = DescriptorTableType::all_types()
            .map(|desc_ty| vk::DescriptorPoolSize {
                ty: desc_ty.to_vk(),
                descriptor_count: desc_ty.max_count(&device),
            })
            .collect();

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(4)
            .pool_sizes(&pool_sizes);

        let pool = unsafe { device.raw.create_descriptor_pool(&pool_create_info, None).unwrap() };

        device.set_debug_name(pool, "bindless_descriptor_pool");

        let descriptor_counts: Vec<_> = DescriptorTableType::all_types().map(|ty| ty.max_count(&device)).collect();

        let mut variable_count =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder().descriptor_counts(&descriptor_counts);

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&descriptor_layouts)
            .push_next(&mut variable_count);

        let descriptor_sets = unsafe { device.raw.allocate_descriptor_sets(&alloc_info).unwrap() };

        let names = ["buffer_descriptor_set", "image_descriptor_set"];

        for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
            device.set_debug_name(*descriptor_set, names[i]);
        }

        Self {
            descriptor_layouts,
            pipeline_layout,

            pool,
            descriptor_sets,

            global_index_allocator: IndexAllocator::new(0),

            immutable_sampler_count: immutable_samplers.len(),
        }
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn bind_descriptors(
        &self,
        recorder: &vulkan::RenderPassRecorder,
        bind_point: vk::PipelineBindPoint
    ) {
        unsafe {
            recorder.device.raw.cmd_bind_descriptor_sets(
                recorder.buffer(),
                bind_point,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            );
        }
    }

    pub fn push_bindings(
        &self,
        recorder: &vulkan::RenderPassRecorder,
        bindings: &[RawDescriptorIndex],
    ) {
        unsafe {
            recorder.device.raw.cmd_push_constants(
                recorder.buffer(),
                self.pipeline_layout,
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::cast_slice(bindings),
            )
        }
    }

    pub fn alloc_buffer_resource(&self, device: &vulkan::Device, handle: vk::Buffer) -> BufferDescriptorIndex {
        let index = self.global_index_allocator.alloc();

        unsafe {
            let buffer_info =
                vk::DescriptorBufferInfo::builder().buffer(handle).offset(0).range(vk::WHOLE_SIZE).build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::Buffer.set_index() as usize])
                .dst_binding(0)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::Buffer.to_vk())
                .buffer_info(std::slice::from_ref(&buffer_info));

            device.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }

        BufferDescriptorIndex(index)
    }

    pub fn free_descriptor_handle(&self, handle: impl DescriptorHandle) {
        self.global_index_allocator.free(handle.to_raw().0);
    }

    pub fn alloc_image_resource(&self, device: &vulkan::Device, image: &vulkan::ImageView) -> ImageDescriptorIndex {
        let index = self.global_index_allocator.alloc();

        unsafe {
            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image.view)
                .build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::Image.set_index() as usize])
                .dst_binding(self.immutable_sampler_count as u32)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::Image.to_vk())
                .image_info(std::slice::from_ref(&image_info));

            device.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }

        ImageDescriptorIndex(index)
    }

    pub fn destroy(&self, device: &vulkan::Device) {
        unsafe {
            device.raw.destroy_descriptor_pool(self.pool, None);

            device.raw.destroy_pipeline_layout(self.pipeline_layout, None);

            for descriptor_layout in &self.descriptor_layouts {
                device.raw.destroy_descriptor_set_layout(*descriptor_layout, None);
            }
        }
    }
}
