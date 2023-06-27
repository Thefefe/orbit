use crate::render;

use std::ops::RangeBounds;
use ash::vk;

#[derive(Debug, Clone, Copy)]
pub struct ImageView {
    pub handle: vk::Image,
    pub descriptor_index: Option<render::ImageDescriptorIndex>,
    pub format: vk::Format,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub subresource_range: vk::ImageSubresourceRange,
}

impl ImageView   {
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
            extent: self.extent,
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

#[derive(Debug, Clone, Copy)]
pub struct BufferView {
    pub handle: vk::Buffer,
    pub descriptor_index: Option<render::BufferDescriptorIndex>,
    pub size: u64
}