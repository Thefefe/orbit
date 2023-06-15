use crate::render;

use std::ops::Range;
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
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    pub fn height(&self) -> u32 {
        self.extent.height
    }

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

    pub fn full_rect(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        }
    }

    pub fn subresource_layers(&self, mip_level: u32, layers: Range<u32>) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: self.subresource_range.aspect_mask,
            mip_level,
            base_array_layer: layers.start,
            layer_count: layers.len() as u32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferView {
    pub handle: vk::Buffer,
    pub descriptor_index: Option<render::BufferDescriptorIndex>,
    pub size: u64
}