use std::{
    borrow::Cow,
    ops::{Range, RangeBounds},
    sync::Arc,
};

use crate::graphics;
use ash::vk;
use gpu_allocator::{
    vulkan::{AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};
use graphics::AccessKind;

#[derive(Debug, Clone, Copy)]
pub struct ImageView {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub _descriptor_index: graphics::DescriptorIndex,
    pub _descriptor_flags: graphics::ImageDescriptorFlags,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub sample_count: graphics::MultisampleCount,
    pub subresource_range: vk::ImageSubresourceRange,
}

impl ImageView {
    pub fn descriptor_index(&self) -> Option<graphics::DescriptorIndex> {
        (!self._descriptor_flags.is_empty()).then_some(self._descriptor_index)
    }

    #[inline(always)]
    pub fn sampled_index(&self) -> Option<graphics::DescriptorIndex> {
        self._descriptor_flags.contains(ImageDescriptorFlags::SAMPLED).then_some(self._descriptor_index)
    }

    #[inline(always)]
    pub fn storage_index(&self) -> Option<graphics::DescriptorIndex> {
        self._descriptor_flags
            .contains(ImageDescriptorFlags::STORAGE)
            .then_some(graphics::strip_sampler(self._descriptor_index))
    }

    #[inline(always)]
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    #[inline(always)]
    pub fn height(&self) -> u32 {
        self.extent.height
    }

    #[inline(always)]
    pub fn depth(&self) -> u32 {
        self.extent.depth
    }

    #[inline(always)]
    pub fn sample_count(&self) -> u32 {
        self.sample_count.sample_count()
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
            extent: vk::Extent2D {
                width: self.extent.width,
                height: self.extent.height,
            },
        }
    }

    #[inline(always)]
    pub fn mip_level(&self) -> u32 {
        self.subresource_range.level_count
    }

    #[inline(always)]
    pub fn layers(&self) -> u32 {
        self.subresource_range.layer_count
    }

    #[inline(always)]
    pub fn subresource_range(
        &self,
        mip_levels: impl RangeBounds<u32>,
        layers: impl RangeBounds<u32>,
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

#[derive(Debug, Clone, Copy)]
pub struct SubresourceView {
    pub view: vk::ImageView,
    pub descriptor_index: Option<graphics::DescriptorIndex>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ImageType {
    Single1D,
    #[default]
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
            ImageType::Single1D => vk::ImageCreateFlags::empty(),
            ImageType::Single2D => vk::ImageCreateFlags::empty(),
            ImageType::Single3D => vk::ImageCreateFlags::empty(),
            ImageType::Array1D(_) => vk::ImageCreateFlags::empty(),
            ImageType::Array2D(_) => vk::ImageCreateFlags::empty(),
            ImageType::Array3D(_) => vk::ImageCreateFlags::empty(),
            ImageType::Cube => vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ImageType::CubeArray(_) => vk::ImageCreateFlags::empty(),
        }
    }

    fn vk_image_type(self) -> vk::ImageType {
        match self {
            ImageType::Single1D => vk::ImageType::TYPE_1D,
            ImageType::Single2D => vk::ImageType::TYPE_2D,
            ImageType::Single3D => vk::ImageType::TYPE_3D,
            ImageType::Array1D(_) => vk::ImageType::TYPE_1D,
            ImageType::Array2D(_) => vk::ImageType::TYPE_2D,
            ImageType::Array3D(_) => vk::ImageType::TYPE_3D,
            ImageType::Cube => vk::ImageType::TYPE_2D,
            ImageType::CubeArray(_) => vk::ImageType::TYPE_2D,
        }
    }

    fn full_vk_view_image_type(self) -> vk::ImageViewType {
        match self {
            ImageType::Single1D => vk::ImageViewType::TYPE_1D,
            ImageType::Single2D => vk::ImageViewType::TYPE_2D,
            ImageType::Single3D => vk::ImageViewType::TYPE_3D,
            ImageType::Array1D(_) => vk::ImageViewType::TYPE_1D,
            ImageType::Array2D(_) => vk::ImageViewType::TYPE_2D,
            ImageType::Array3D(_) => vk::ImageViewType::TYPE_3D,
            ImageType::Cube => vk::ImageViewType::CUBE,
            ImageType::CubeArray(_) => vk::ImageViewType::CUBE_ARRAY,
        }
    }

    fn layer_vk_view_image_type(self) -> vk::ImageViewType {
        match self {
            ImageType::Single1D => vk::ImageViewType::TYPE_1D,
            ImageType::Single2D => vk::ImageViewType::TYPE_2D,
            ImageType::Single3D => vk::ImageViewType::TYPE_3D,
            ImageType::Array1D(_) => vk::ImageViewType::TYPE_1D,
            ImageType::Array2D(_) => vk::ImageViewType::TYPE_2D,
            ImageType::Array3D(_) => vk::ImageViewType::TYPE_3D,
            ImageType::Cube => vk::ImageViewType::TYPE_2D,
            ImageType::CubeArray(_) => vk::ImageViewType::TYPE_2D_ARRAY,
        }
    }

    pub fn layer_count(self) -> u32 {
        match self {
            ImageType::Single2D => 1,
            ImageType::Single1D => 1,
            ImageType::Single3D => 1,
            ImageType::Array1D(count) => count,
            ImageType::Array2D(count) => count,
            ImageType::Array3D(count) => count,
            ImageType::Cube => 6,
            ImageType::CubeArray(count) => 6 * count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageDescriptorFlags(u32);
ash::vk_bitflags_wrapped!(ImageDescriptorFlags, u32);

impl ImageDescriptorFlags {
    pub const NONE: Self = Self(0b0);
    pub const SAMPLED: Self = Self(0b1);
    pub const STORAGE: Self = Self(0b01);
}

/// Describes how to create additional image views for the subresources
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ImageSubresourceViewDesc {
    /// How many mip levels should have their own image view.
    /// A value of 0 means no additional image view will be created for the levels.
    pub mip_count: u32,
    /// What extra descriptor indices should the additional image views have.
    pub mip_descriptors: ImageDescriptorFlags,

    // How many mip levels should have additional image views for their layers.
    pub layer_mip_count: u32,
    /// How many layers should have their own image view in the first `layer_mip_granularity`
    /// number of levels.
    /// A value of 0 means no additional image view will be created for the layers.
    pub layer_count: u32,
    /// What extra descriptor indices should the additional image views have.
    pub layer_descrptors: ImageDescriptorFlags,
}

/// The queue ownership of the resource
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SharingMode {
    /// The (sub)resource can only be used on one queue at a time. Queue ownership and transfer
    /// will be managed automaticaly.
    #[default]Exclusive,
    /// The resource can be used concurently on multiple queues, slower than `Exclusive`
    Concurent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ImageDesc {
    pub ty: ImageType,
    pub format: vk::Format,
    pub dimensions: [u32; 3],
    pub mip_levels: u32,
    pub samples: graphics::MultisampleCount,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    pub subresource_desc: ImageSubresourceViewDesc,
    pub default_sampler: Option<graphics::SamplerKind>,
    pub sharing_mode: SharingMode,
}

#[derive(Debug, Clone)]
pub struct ImageRaw {
    pub name: Cow<'static, str>,
    pub full_view: graphics::ImageView,
    pub subresource_views: Vec<SubresourceView>,
    pub desc: ImageDesc,
    pub alloc_index: graphics::AllocIndex,
}

impl std::ops::Deref for ImageRaw {
    type Target = graphics::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.full_view
    }
}

impl ImageRaw {
    pub(super) fn create_impl(
        device: &graphics::Device,
        name: Cow<'static, str>,
        desc: &ImageDesc,
        preallocated_descriptor_index: Option<graphics::DescriptorIndex>,
    ) -> ImageRaw {
        puffin::profile_function!(&name);

        let extent = vk::Extent3D {
            width: desc.dimensions[0],
            height: desc.dimensions[1],
            depth: desc.dimensions[2],
        };

        let layer_count = desc.ty.layer_count();

        let mut create_info = vk::ImageCreateInfo::builder()
            .flags(desc.ty.vk_image_flags())
            .image_type(desc.ty.vk_image_type())
            .extent(extent)
            .array_layers(layer_count)
            .mip_levels(desc.mip_levels)
            .format(desc.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .samples(desc.samples.to_vk())
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(desc.usage);

        match desc.sharing_mode {
            graphics::SharingMode::Concurent if device.queue_family_count > 1 => {
                create_info = create_info
                    .sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(device.queue_family_indices())
            }
            _ => create_info = create_info.sharing_mode(vk::SharingMode::EXCLUSIVE),
        }

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

        let view = unsafe { device.raw.create_image_view(&image_view_create_info, None).unwrap() };
        device.set_debug_name(view, &format!("{}_full_view", name));

        let mut descriptor_flags = ImageDescriptorFlags::empty();
        let full_descriptor_index =
            if desc.usage.intersects(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE) {
                let index = preallocated_descriptor_index.unwrap_or_else(|| device.alloc_descriptor_index());

                if desc.usage.contains(vk::ImageUsageFlags::SAMPLED) {
                    device.write_sampled_image(index, view);
                    descriptor_flags |= ImageDescriptorFlags::SAMPLED;
                };

                if desc.usage.contains(vk::ImageUsageFlags::STORAGE) {
                    device.write_storage_image(index, view);
                    descriptor_flags |= ImageDescriptorFlags::STORAGE;
                }
                Some(index)
            } else {
                None
            };

        let mut subresource_count = 0;

        if desc.mip_levels > 1 {
            subresource_count += desc.subresource_desc.mip_count.min(desc.mip_levels);
        }

        if layer_count > 1 {
            subresource_count += desc.subresource_desc.layer_mip_count.min(desc.mip_levels)
                * desc.subresource_desc.layer_count.min(layer_count)
        }

        let mut subresource_views = Vec::with_capacity(subresource_count as usize);

        if desc.mip_levels > 1 {
            for mip_level in 0..desc.subresource_desc.mip_count.min(desc.mip_levels) {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: desc.aspect,
                    base_mip_level: mip_level,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: layer_count,
                };

                let image_view_create_info = vk::ImageViewCreateInfo::builder()
                    .image(handle)
                    .view_type(desc.ty.full_vk_view_image_type())
                    .format(desc.format)
                    .subresource_range(subresource_range);

                let view = unsafe { device.raw.create_image_view(&image_view_create_info, None).unwrap() };
                device.set_debug_name(view, &format!("{name}_level_{mip_level}"));

                let subresource_descriptor_index = if !desc.subresource_desc.mip_descriptors.is_empty() {
                    let subresource_descriptor_index = device.alloc_descriptor_index();

                    if desc.subresource_desc.mip_descriptors.contains(ImageDescriptorFlags::SAMPLED)
                        | desc.usage.contains(vk::ImageUsageFlags::SAMPLED)
                    {
                        device.write_sampled_image(subresource_descriptor_index, view);
                    }

                    if desc.subresource_desc.mip_descriptors.contains(ImageDescriptorFlags::STORAGE)
                        | desc.usage.contains(vk::ImageUsageFlags::STORAGE)
                    {
                        device.write_storage_image(subresource_descriptor_index, view);
                    }

                    Some(subresource_descriptor_index)
                } else {
                    None
                };

                subresource_views.push(SubresourceView {
                    view,
                    descriptor_index: subresource_descriptor_index,
                });
            }
        }

        if layer_count > 1 {
            for mip_level in 0..desc.subresource_desc.layer_mip_count.min(desc.mip_levels) {
                for layer in 0..desc.subresource_desc.layer_count.min(layer_count) {
                    let subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: desc.aspect,
                        base_mip_level: mip_level,
                        level_count: 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    };

                    let image_view_create_info = vk::ImageViewCreateInfo::builder()
                        .image(handle)
                        .view_type(desc.ty.layer_vk_view_image_type())
                        .format(desc.format)
                        .subresource_range(subresource_range);

                    let view = unsafe { device.raw.create_image_view(&image_view_create_info, None).unwrap() };
                    device.set_debug_name(view, &format!("{name}_level_{mip_level}_layer_{layer}"));

                    let subresource_descriptor_index = if !desc.subresource_desc.layer_descrptors.is_empty() {
                        let subresource_descriptor_index = device.alloc_descriptor_index();

                        if desc.subresource_desc.layer_descrptors.contains(ImageDescriptorFlags::SAMPLED)
                            | desc.usage.contains(vk::ImageUsageFlags::SAMPLED)
                        {
                            device.write_sampled_image(subresource_descriptor_index, view);
                        }

                        if desc.subresource_desc.layer_descrptors.contains(ImageDescriptorFlags::STORAGE)
                            | desc.usage.contains(vk::ImageUsageFlags::STORAGE)
                        {
                            device.write_storage_image(subresource_descriptor_index, view);
                        }

                        Some(subresource_descriptor_index)
                    } else {
                        None
                    };

                    subresource_views.push(SubresourceView {
                        view,
                        descriptor_index: subresource_descriptor_index,
                    });
                }
            }
        }

        let _descriptor_index = graphics::descriptor_index_with_sampler(
            full_descriptor_index.unwrap_or(0),
            desc.default_sampler.unwrap_or(graphics::SamplerKind::LinearClamp),
        );

        ImageRaw {
            name,
            full_view: graphics::ImageView {
                handle,
                _descriptor_index,
                _descriptor_flags: descriptor_flags,
                format: desc.format,
                view,
                extent,
                sample_count: desc.samples,
                subresource_range,
            },
            desc: desc.clone(),
            subresource_views,
            alloc_index: memory,
        }
    }

    pub(super) fn destroy_impl(device: &graphics::Device, image: &graphics::ImageRaw) {
        puffin::profile_function!(&image.name);

        if !image._descriptor_flags.is_empty() {
            device.free_descriptor_index(image._descriptor_index);
        }
        unsafe {
            puffin::profile_scope!("free_vulkan_resources");
            device.raw.destroy_image_view(image.view, None);
            device.raw.destroy_image(image.handle, None);

            for subresource_view in image.subresource_views.iter() {
                device.raw.destroy_image_view(subresource_view.view, None);
                if let Some(descriptor_index) = subresource_view.descriptor_index {
                    device.free_descriptor_index(descriptor_index);
                }
            }
        }
        device.deallocate(image.alloc_index);
    }

    pub fn mip_view_count(&self) -> usize {
        if self.desc.mip_levels > 1 {
            self.desc.subresource_desc.mip_count.min(self.desc.mip_levels) as usize
        } else {
            0
        }
    }

    pub fn layer_view_count(&self) -> usize {
        if self.desc.ty.layer_count() > 1 {
            (self.desc.subresource_desc.layer_mip_count.min(self.desc.mip_levels)
                * self.desc.subresource_desc.layer_count.min(self.desc.ty.layer_count())) as usize
        } else {
            0
        }
    }

    pub fn mip_view(&self, mip_level: usize) -> Option<ImageView> {
        if mip_level >= self.mip_view_count() {
            return None;
        }

        let SubresourceView { view, descriptor_index } = self.subresource_views[mip_level];

        let [width, height, depth] = self.desc.dimensions.map(|x| u32::max(x >> mip_level as u32, 1));
        let _descriptor_index = graphics::descriptor_index_with_sampler(
            descriptor_index.unwrap_or(0),
            self.desc.default_sampler.unwrap_or(graphics::SamplerKind::LinearClamp),
        );

        Some(ImageView {
            handle: self.handle,
            view,
            _descriptor_index,
            _descriptor_flags: self.desc.subresource_desc.mip_descriptors,
            format: self.desc.format,
            extent: vk::Extent3D { width, height, depth },
            sample_count: self.desc.samples,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: self.desc.aspect,
                base_mip_level: mip_level as u32,
                level_count: 1,
                base_array_layer: 0,
                layer_count: self.desc.ty.layer_count(),
            },
        })
    }

    pub fn layer_view(&self, mip_level: u32, layer: u32) -> Option<ImageView> {
        if mip_level >= self.desc.subresource_desc.layer_mip_count.min(self.desc.mip_levels)
            || layer >= self.desc.subresource_desc.layer_count.min(self.desc.ty.layer_count())
        {
            return None;
        }

        let layer_view_index =
            mip_level * self.desc.subresource_desc.layer_count.min(self.desc.ty.layer_count()) + layer;

        let SubresourceView { view, descriptor_index } =
            self.subresource_views[self.mip_view_count() + layer_view_index as usize];

        let [width, height, depth] = self.desc.dimensions.map(|x| u32::max(x >> mip_level as u32, 1));
        let _descriptor_index = graphics::descriptor_index_with_sampler(
            descriptor_index.unwrap_or(0),
            self.desc.default_sampler.unwrap_or(graphics::SamplerKind::LinearClamp),
        );

        Some(ImageView {
            handle: self.handle,
            view,
            _descriptor_index,
            _descriptor_flags: self.desc.subresource_desc.layer_descrptors,
            format: self.desc.format,
            extent: vk::Extent3D { width, height, depth },
            sample_count: self.sample_count,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: self.desc.aspect,
                base_mip_level: mip_level as u32,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            },
        })
    }
}

#[derive(Clone)]
pub struct Image {
    pub _image: Option<Arc<graphics::ImageRaw>>,
    pub _device: Arc<graphics::Device>,
}

impl Image {
    pub fn recreate(&mut self, desc: &ImageDesc) -> bool {
        if &self.desc == desc {
            return false;
        }

        let mut image = Arc::new(ImageRaw::create_impl(&self._device, self.name.clone(), desc, None));
        std::mem::swap(self._image.as_mut().unwrap(), &mut image);

        if let Some(image) = Arc::into_inner(image) {
            ImageRaw::destroy_impl(&self._device, &image);
        }

        true
    }
}

impl std::ops::Deref for graphics::Image {
    type Target = graphics::ImageRaw;

    fn deref(&self) -> &Self::Target {
        self._image.as_ref().unwrap()
    }
}

impl Drop for graphics::Image {
    fn drop(&mut self) {
        if let Some(image) = Arc::into_inner(self._image.take().unwrap()) {
            ImageRaw::destroy_impl(&self._device, &image);
        }
    }
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self._image.fmt(f)
    }
}

impl graphics::Context {
    pub fn create_image(&self, name: impl Into<Cow<'static, str>>, desc: &ImageDesc) -> Image {
        let image = ImageRaw::create_impl(&self.device, name.into(), desc, None);
        Image {
            _image: Some(Arc::new(image)),
            _device: self.device.clone(),
        }
    }

    pub fn immediate_write_image(
        &self,
        image: &graphics::ImageView,
        mip_level: u32,
        layers: Range<u32>,
        prev_access: AccessKind,
        target_access: Option<AccessKind>,
        bytes: &[u8],
        subregion: Option<(vk::Offset3D, vk::Extent3D)>,
    ) {
        puffin::profile_function!();

        let scratch_buffer = self.create_buffer_init(
            "scratch_buffer",
            &graphics::BufferDesc {
                size: bytes.len(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
            bytes,
        );

        let (image_offset, image_extent) = subregion.unwrap_or((vk::Offset3D::default(), image.extent));

        self.record_and_submit(|cmd| {
            cmd.barrier(
                &[],
                &[graphics::image_barrier(image, prev_access, AccessKind::TransferWrite)],
                &[],
            );

            cmd.copy_buffer_to_image(
                scratch_buffer.handle,
                image.handle,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: image.subresource_layers(mip_level, layers),
                    image_offset,
                    image_extent,
                }],
            );

            if let Some(target_access) = target_access {
                cmd.barrier(
                    &[],
                    &[graphics::image_barrier(image, AccessKind::TransferWrite, target_access)],
                    &[],
                );
            }
        });

        drop(scratch_buffer);
    }
}
