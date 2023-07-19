use std::{ops::{Range, RangeBounds}, borrow::Cow};

use crate::render;
use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::{AllocationCreateDesc, AllocationScheme}};
use render::AccessKind;

#[derive(Debug, Clone, Copy)]
pub struct ImageView {
    pub handle: vk::Image,
    pub descriptor_index: Option<render::DescriptorIndex>,
    pub format: vk::Format,
    pub view: vk::ImageView,
    pub extent: vk::Extent3D,
    pub subresource_range: vk::ImageSubresourceRange,
}

impl ImageView {
    #[inline(always)]
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    #[inline(always)]
    pub fn height(&self) -> u32 {
        self.extent.height
    }

    #[inline(always)]
    pub fn full_viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: 0.0,
            y: self.extent.height as f32,
            width: self.extent.width as f32,
            height: -(self.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    #[inline(always)]
    pub fn full_rect(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: self.extent.width, height: self.extent.height },
        }
    }

    #[inline(always)]
    pub fn subresource_range(
        &self,
        mip_levels: impl RangeBounds<u32>,
        layers: impl RangeBounds<u32>
    ) -> vk::ImageSubresourceRange {
        let (base_array_layer, layer_count) =
            crate::utils::range_bounds_to_base_count(layers, 0, self.subresource_range.layer_count);
        let (base_mip_level, level_count) =
            crate::utils::range_bounds_to_base_count(mip_levels, 0, self.subresource_range.level_count);
        vk::ImageSubresourceRange {
            aspect_mask: self.subresource_range.aspect_mask,
            base_array_layer,
            layer_count,
            base_mip_level,
            level_count,
        }
    }


    #[inline(always)]
    pub fn subresource_layers(&self, mip_level: u32, layers: impl RangeBounds<u32>) -> vk::ImageSubresourceLayers {
        let (base_array_layer, layer_count) =
            crate::utils::range_bounds_to_base_count(layers, 0, self.subresource_range.layer_count);
        vk::ImageSubresourceLayers {
            aspect_mask: self.subresource_range.aspect_mask,
            mip_level,
            base_array_layer,
            layer_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageType {
    Single1D,
    Single2D,
    Single3D,
    Array1D(u32),
    Array2D(u32),
    Array3D(u32),
    Cube,
    CubeArray(u32),
}

impl ImageType {
    fn vk_image_flags(self) -> vk::ImageCreateFlags {
        match self {
            ImageType::Single1D     => vk::ImageCreateFlags::empty(),
            ImageType::Single2D     => vk::ImageCreateFlags::empty(),
            ImageType::Single3D     => vk::ImageCreateFlags::empty(),
            ImageType::Array1D(_)   => vk::ImageCreateFlags::empty(),
            ImageType::Array2D(_)   => vk::ImageCreateFlags::empty(),
            ImageType::Array3D(_)   => vk::ImageCreateFlags::empty(),
            ImageType::Cube         => vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ImageType::CubeArray(_) => vk::ImageCreateFlags::empty(),
        }
    }

    fn vk_image_type(self) -> vk::ImageType {
        match self {
            ImageType::Single1D     => vk::ImageType::TYPE_1D,
            ImageType::Single2D     => vk::ImageType::TYPE_2D,
            ImageType::Single3D     => vk::ImageType::TYPE_3D,
            ImageType::Array1D(_)   => vk::ImageType::TYPE_1D,
            ImageType::Array2D(_)   => vk::ImageType::TYPE_2D,
            ImageType::Array3D(_)   => vk::ImageType::TYPE_3D,
            ImageType::Cube         => vk::ImageType::TYPE_2D,
            ImageType::CubeArray(_) => vk::ImageType::TYPE_2D,
        }
    }

    fn full_vk_view_image_type(self) -> vk::ImageViewType {
        match self {
            ImageType::Single1D     => vk::ImageViewType::TYPE_1D,
            ImageType::Single2D     => vk::ImageViewType::TYPE_2D,
            ImageType::Single3D     => vk::ImageViewType::TYPE_3D,
            ImageType::Array1D(_)   => vk::ImageViewType::TYPE_1D,
            ImageType::Array2D(_)   => vk::ImageViewType::TYPE_2D,
            ImageType::Array3D(_)   => vk::ImageViewType::TYPE_3D,
            ImageType::Cube         => vk::ImageViewType::CUBE,
            ImageType::CubeArray(_) => vk::ImageViewType::CUBE_ARRAY,
        }
    }

    fn layer_vk_view_image_type(self) -> vk::ImageViewType {
        match self {
            ImageType::Single1D     => vk::ImageViewType::TYPE_1D,
            ImageType::Single2D     => vk::ImageViewType::TYPE_2D,
            ImageType::Single3D     => vk::ImageViewType::TYPE_3D,
            ImageType::Array1D(_)   => vk::ImageViewType::TYPE_1D,
            ImageType::Array2D(_)   => vk::ImageViewType::TYPE_2D,
            ImageType::Array3D(_)   => vk::ImageViewType::TYPE_3D,
            ImageType::Cube         => vk::ImageViewType::TYPE_2D,
            ImageType::CubeArray(_) => vk::ImageViewType::TYPE_2D_ARRAY,
        }
    }

    fn layer_count(self) -> u32 {
        match self {
            ImageType::Single2D         => 1,
            ImageType::Single1D         => 1,
            ImageType::Single3D         => 1,
            ImageType::Array1D(count)   => count,
            ImageType::Array2D(count)   => count,
            ImageType::Array3D(count)   => count,
            ImageType::Cube             => 6,
            ImageType::CubeArray(count) => 6 * count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageDesc {
    pub ty: ImageType,
    pub format: vk::Format,
    pub dimensions: [u32; 3],
    pub mip_levels: u32,
    pub samples: render::MultisampleCount,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
}

#[derive(Debug, Clone)]
pub struct Image {
    pub name: Cow<'static, str>,
    pub full_view: render::ImageView,
    pub layer_views: Vec<vk::ImageView>,
    alloc_index: render::AllocIndex,
}

impl std::ops::Deref for Image {
    type Target = render::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.full_view
    }
}

impl Image {
    pub(super) fn create_impl(
        device: &render::Device,
        descriptors: &render::BindlessDescriptors,
        name: Cow<'static, str>,
        desc: &ImageDesc,
        preallocated_descriptor_index: Option<render::DescriptorIndex>,
    ) -> Image {
        puffin::profile_function!(&name);

        let extent = vk::Extent3D {
            width: desc.dimensions[0],
            height: desc.dimensions[1],
            depth: desc.dimensions[2],
        };

        let layer_count = desc.ty.layer_count();

        let create_info = vk::ImageCreateInfo::builder()
            .flags(desc.ty.vk_image_flags())
            .image_type(desc.ty.vk_image_type())
            .extent(extent)
            .array_layers(layer_count)
            .mip_levels(desc.mip_levels)
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
            layer_count: desc.ty.layer_count(),
        };

        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(handle)
            .view_type(desc.ty.full_vk_view_image_type())
            .format(desc.format)
            .subresource_range(subresource_range);

        let view = unsafe {
            device.raw.create_image_view(&image_view_create_info, None).unwrap()
        };
        device.set_debug_name(view, &format!("{}_full_view", name));

        let descriptor_index = if desc.usage.contains(vk::ImageUsageFlags::SAMPLED) {
            let index = preallocated_descriptor_index
                .unwrap_or_else(|| descriptors.alloc_index());
            descriptors.write_sampled_image(device, index, view);
            Some(index)
        } else {
            None
        };

        let layer_views = if layer_count == 1 {
            Vec::new()
        } else {
            (0..layer_count).map(|layer_index| unsafe {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: desc.aspect,
                    base_mip_level: 0,
                    level_count: desc.mip_levels,
                    base_array_layer: layer_index,
                    layer_count: 1,
                };

                let image_view_create_info = vk::ImageViewCreateInfo::builder()
                    .image(handle)
                    .view_type(desc.ty.layer_vk_view_image_type())
                    .format(desc.format)
                    .subresource_range(subresource_range);
        
                let view = device.raw.create_image_view(&image_view_create_info, None).unwrap();
                device.set_debug_name(view, &format!("{name}_layer_{layer_index}"));

                view
            }).collect()
        };

        Image {
            name,
            full_view: render::ImageView {
                handle,
                descriptor_index,
                format: desc.format,
                view,
                extent,
                subresource_range,
            },
            layer_views,
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
            for layer_view in image.layer_views.iter() {
                device.raw.destroy_image_view(*layer_view, None);
            }
        }
        device.deallocate(image.alloc_index);
    }

    pub fn set_sampler_flags(&mut self, sampler_flags: render::SamplerKind) {
        if let Some(index) = self.full_view.descriptor_index.as_mut() {
            *index = render::descriptor_index_with_sampler(*index, sampler_flags);
        }
    }
}

impl render::Context {
    pub fn create_image(&self, name: impl Into<Cow<'static, str>>, desc: &ImageDesc) -> Image {
        Image::create_impl(&self.device, &self.descriptors, name.into(), desc, None)
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
        puffin::profile_function!();

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