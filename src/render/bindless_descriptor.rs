use crate::render;

use std::sync::{
    atomic::{self, AtomicU32},
    Mutex,
};
use ash::vk;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DescriptorTableType {
    StorageBuffer = 0,
    SampledImage  = 1,
    StorageImage  = 2,
}

impl DescriptorTableType {
    fn all_types() -> impl Iterator<Item = Self> {
        [Self::StorageBuffer, Self::SampledImage].into_iter()
    }

    fn set_index(self) -> u32 {
        self as u32
    }

    fn from_set_index(set_index: u32) -> Self {
        match set_index {
            0 => Self::StorageBuffer,
            1 => Self::SampledImage,
            _ => panic!("invalid set index"),
        }
    }

    fn to_vk(self) -> vk::DescriptorType {
        match self {
            DescriptorTableType::StorageBuffer  => vk::DescriptorType::STORAGE_BUFFER,
            DescriptorTableType::SampledImage   => vk::DescriptorType::SAMPLED_IMAGE,
            DescriptorTableType::StorageImage   => vk::DescriptorType::STORAGE_IMAGE,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DescriptorTableType::StorageBuffer  => "storage_buffer",
            DescriptorTableType::SampledImage   => "sampled_image",
            DescriptorTableType::StorageImage   => "storage_image",
        }
    }

    fn max_count(self, device: &render::Device) -> u32 {
        let props = &device.gpu.properties.properties12;
        match self {
            DescriptorTableType::StorageBuffer => u32::min(
                props.max_descriptor_set_update_after_bind_storage_buffers,
                props.max_per_stage_descriptor_update_after_bind_storage_buffers,
            ),
            DescriptorTableType::SampledImage => u32::min(
                props.max_descriptor_set_update_after_bind_sampled_images,
                props.max_per_stage_descriptor_update_after_bind_sampled_images,
            ),
            DescriptorTableType::StorageImage => u32::min(
                props.max_descriptor_set_update_after_bind_storage_images,
                props.max_per_stage_descriptor_update_after_bind_storage_images,
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

pub type DescriptorIndex = u32;

pub struct BindlessDescriptors {
    descriptor_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,

    pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    global_index_allocator: IndexAllocator,

    immutable_samplers: [vk::Sampler; SAMPLER_COUNT],
}

impl BindlessDescriptors {
    pub fn new(device: &render::Device) -> Self {
        let immutable_samplers = SamplerKind::ALL.map(|sampler_kind| unsafe {
            device.raw.create_sampler(&sampler_kind.create_info(), None).unwrap()
        });

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

                if desc_type == DescriptorTableType::SampledImage && !immutable_samplers.is_empty() {
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
            size: 128,
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

            immutable_samplers,
        }
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn bind_descriptors(
        &self,
        recorder: &render::CommandRecorder,
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

    pub fn alloc_index(&self) -> DescriptorIndex {
        self.global_index_allocator.alloc()
    }

    pub fn write_storage_buffer_resource(
        &self,
        device: &render::Device,
        index: DescriptorIndex,
        handle: vk::Buffer
    ) {
        unsafe {
            let buffer_info =
                vk::DescriptorBufferInfo::builder().buffer(handle).offset(0).range(vk::WHOLE_SIZE).build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::StorageBuffer.set_index() as usize])
                .dst_binding(0)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::StorageBuffer.to_vk())
                .buffer_info(std::slice::from_ref(&buffer_info));

            device.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }
    }

    pub fn write_sampled_image(
        &self,
        device: &render::Device,
        index: DescriptorIndex,
        handle: vk::ImageView
    ) {
        unsafe {
            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(handle)
                .build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::SampledImage.set_index() as usize])
                .dst_binding(self.immutable_samplers.len() as u32)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::SampledImage.to_vk())
                .image_info(std::slice::from_ref(&image_info));

            device.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }
    }

    pub fn write_storage_image(
        &self,
        device: &render::Device,
        index: DescriptorIndex,
        handle: vk::ImageView,
    ) {
        unsafe {
            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(handle)
                .build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::StorageImage.set_index() as usize])
                .dst_binding(0)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::StorageImage.to_vk())
                .image_info(std::slice::from_ref(&image_info));

            device.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }
    }

    pub fn free_descriptor_index(&self, index: DescriptorIndex) {
        self.global_index_allocator.free(strip_sampler(index));
    }

    pub fn destroy(&self, device: &render::Device) {
        unsafe {
            for sampler in self.immutable_samplers.iter() {
                device.raw.destroy_sampler(*sampler, None);
            }

            device.raw.destroy_descriptor_pool(self.pool, None);
            device.raw.destroy_pipeline_layout(self.pipeline_layout, None);

            for descriptor_layout in &self.descriptor_layouts {
                device.raw.destroy_descriptor_set_layout(*descriptor_layout, None);
            }
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    LinearClamp      = 0,
    LinearRepeat     = 1,
    NearestClamp     = 2,
    NearestRepeat    = 3,
    ShadowComparison = 4,
    ShadowDepth      = 5,
}

const SAMPLER_COUNT: usize = 6;

impl SamplerKind {
    pub const ALL: [SamplerKind; SAMPLER_COUNT] = [
        Self::LinearClamp,
        Self::LinearRepeat,
        Self::NearestClamp,
        Self::NearestRepeat,
        Self::ShadowComparison,
        Self::ShadowDepth,
    ];

    fn create_info(self) -> vk::SamplerCreateInfo {
        match self {
            SamplerKind::LinearClamp => vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE)
                .build(),
            SamplerKind::LinearRepeat => vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE)
                .build(),
            SamplerKind::NearestClamp => vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .min_filter(vk::Filter::NEAREST)
                .mag_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE)
                .build(),
            SamplerKind::NearestRepeat => vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .min_filter(vk::Filter::NEAREST)
                .mag_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE)
                .build(),
            SamplerKind::ShadowComparison => vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE)
                .compare_enable(true)
                .compare_op(vk::CompareOp::GREATER_OR_EQUAL)
                .build(),
            SamplerKind::ShadowDepth => vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE)
                .build(),
                
        }
    }
}

pub fn strip_sampler(index: DescriptorIndex) -> DescriptorIndex {
    index << 8 >> 8
}

pub fn descriptor_index_with_sampler(index: DescriptorIndex, sampler: SamplerKind) -> DescriptorIndex {
    assert!(index >> 24 == 0);
    index | ((sampler as u32) << 24)
}