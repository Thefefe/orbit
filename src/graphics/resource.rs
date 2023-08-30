use std::{borrow::Cow, sync::Arc};

use ash::vk;

use crate::graphics;


pub trait RenderResource: Into<AnyResource> {
    type RawResource: OwnedRenderResource;
}

impl<'a> RenderResource for &'a graphics::Buffer {
    type RawResource = graphics::BufferRaw;
}

impl RenderResource for graphics::BufferRaw {
    type RawResource = graphics::BufferRaw;
}

impl<'a> RenderResource for &'a graphics::Image {
    type RawResource = graphics::ImageRaw;
}

impl RenderResource for graphics::ImageRaw {
    type RawResource = graphics::ImageRaw;
}

pub trait OwnedRenderResource: Into<graphics::AnyResource> {
    type Desc: Into<graphics::AnyResourceDesc>;
}

impl OwnedRenderResource for graphics::BufferRaw {
    type Desc = graphics::BufferDesc;
}

impl OwnedRenderResource for graphics::ImageRaw {
    type Desc = graphics::ImageDesc;
}

#[derive(Debug)]
pub enum AnyResource {
    BufferOwned(graphics::BufferRaw),
    BufferShared(Arc<graphics::BufferRaw>),
    ImageOwned(graphics::ImageRaw),
    ImageShared(Arc<graphics::ImageRaw>),
}

impl From<Arc<graphics::ImageRaw>> for AnyResource {
    fn from(v: Arc<graphics::ImageRaw>) -> Self {
        Self::ImageShared(v)
    }
}

impl AnyResource {
    #[inline(always)]
    pub fn as_ref(&self) -> graphics::AnyResourceRef {
        match self {
            AnyResource::BufferOwned(buffer)  => graphics::AnyResourceRef::Buffer(&buffer),
            AnyResource::BufferShared(buffer) => graphics::AnyResourceRef::Buffer(&buffer),
            AnyResource::ImageOwned(image)    => graphics::AnyResourceRef::Image(&image),
            AnyResource::ImageShared(image)   => graphics::AnyResourceRef::Image(&image),
        }
    }

    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        match self {
            AnyResource::BufferOwned(_)  => true,
            AnyResource::ImageOwned(_)   => true,
            AnyResource::BufferShared(_) => false,
            AnyResource::ImageShared(_)  => false,
        }
    }

    #[inline(always)]
    pub fn name(&self) -> &str {
        match self.as_ref() {
            graphics::AnyResourceRef::Buffer(buffer) => &buffer.name,
            graphics::AnyResourceRef::Image(image)   => &image.name,
        }
    }

    #[inline(always)]
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResource::BufferOwned(_)  => ResourceKind::Buffer,
            AnyResource::BufferShared(_) => ResourceKind::Buffer,
            AnyResource::ImageOwned(_)   => ResourceKind::Image,
            AnyResource::ImageShared(_)  => ResourceKind::Image,
        }    
    }

    #[inline(always)]
    pub fn desc(&self) -> AnyResourceDesc {
        match self {
            AnyResource::BufferOwned(buffer)  => AnyResourceDesc::Buffer(buffer.desc),
            AnyResource::BufferShared(buffer) => AnyResourceDesc::Buffer(buffer.desc),
            AnyResource::ImageOwned(image)    => AnyResourceDesc::Image(image.desc),
            AnyResource::ImageShared(image)   => AnyResourceDesc::Image(image.desc),
        }
    }

    pub fn create_owned(
        device: &graphics::Device,
        name: Cow<'static, str>,
        desc: &AnyResourceDesc,
        preallocated_descriptor_index: Option<graphics::DescriptorIndex>,
    ) -> Self {
        match desc {
            AnyResourceDesc::Buffer(desc) => {
                AnyResource::BufferOwned(graphics::BufferRaw::create_impl(
                    device,
                    name,
                    desc,
                    preallocated_descriptor_index,
                ))
            },
            AnyResourceDesc::Image(desc) => {
                AnyResource::ImageOwned(graphics::ImageRaw::create_impl(
                    device,
                    name,
                    desc,
                    preallocated_descriptor_index,
                ))
            },
        }
    }

    pub fn destroy(device: &graphics::Device, resource: AnyResource) {
        match resource {
            AnyResource::BufferOwned(buffer) => {
                graphics::BufferRaw::destroy_impl(device, &buffer);
            },
            AnyResource::ImageOwned(image) => {
                graphics::ImageRaw::destroy_impl(device, &image);
            },
            AnyResource::BufferShared(buffer) => if let Some(buffer) = Arc::into_inner(buffer) {
                graphics::BufferRaw::destroy_impl(device, &buffer);
            },
            AnyResource::ImageShared(image) => if let Some(image) = Arc::into_inner(image) {
                graphics::ImageRaw::destroy_impl(device, &image);
            },
        }
    }
}

impl From<graphics::ImageRaw> for AnyResource {
    fn from(v: graphics::ImageRaw) -> Self {
        Self::ImageOwned(v)
    }
}

impl From<Arc<graphics::BufferRaw>> for AnyResource {
    fn from(v: Arc<graphics::BufferRaw>) -> Self {
        Self::BufferShared(v)
    }
}

impl From<graphics::BufferRaw> for AnyResource {
    fn from(v: graphics::BufferRaw) -> Self {
        Self::BufferOwned(v)
    }
}

impl<'a> From<&'a graphics::Buffer> for AnyResource {
    fn from(value: &'a graphics::Buffer) -> Self {
        Self::BufferShared(value._buffer.as_ref().unwrap().clone())
    }
}

impl<'a> From<&'a graphics::Image> for AnyResource {
    fn from(value: &'a graphics::Image) -> Self {
        Self::ImageShared(value._image.as_ref().unwrap().clone())
    }
}

pub enum AnyResourceRef<'a> {
    Buffer(&'a graphics::BufferRaw),
    Image(&'a graphics::ImageRaw),
}

impl AnyResourceRef<'_> {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceRef::Buffer(_) => ResourceKind::Buffer,
            AnyResourceRef::Image(_)  => ResourceKind::Image,
        }
    }

    pub fn handle(&self) -> AnyResourceHandle {
        match self {
            AnyResourceRef::Buffer(buffer) => AnyResourceHandle::Buffer(buffer.handle),
            AnyResourceRef::Image(image)   => AnyResourceHandle::Image(image.handle),
        }
    }

    pub fn descriptor_index(&self) -> Option<u32> {
        match self {
            AnyResourceRef::Buffer(buffer) => buffer.descriptor_index,
            AnyResourceRef::Image(image)   => (!image._descriptor_flags.is_empty()).then_some(image._descriptor_index),
        }
    }

    pub fn name(&self) -> &Cow<'static, str> {
        match self {
            AnyResourceRef::Buffer(buffer) => &buffer.name,
            AnyResourceRef::Image(image)   => &image.name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyResourceHandle {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyResourceDesc {
    Buffer(graphics::BufferDesc),
    Image(graphics::ImageDesc),
}

impl AnyResourceDesc {
    pub fn needs_descriptor_index(&self) -> bool {
        match self {
            AnyResourceDesc::Buffer(buffer) => buffer.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER),
            AnyResourceDesc::Image(image)   => image.usage.intersects(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
        }
    }
}

impl From<graphics::ImageDesc> for AnyResourceDesc {
    fn from(v: graphics::ImageDesc) -> Self {
        Self::Image(v)
    }
}

impl From<graphics::BufferDesc> for AnyResourceDesc {
    fn from(v: graphics::BufferDesc) -> Self {
        Self::Buffer(v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Buffer,
    Image,
}

impl AnyResourceDesc {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceDesc::Buffer(_)  => ResourceKind::Buffer,
            AnyResourceDesc::Image(_)   => ResourceKind::Image,
        }
    }
}