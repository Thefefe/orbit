use std::borrow::Cow;

use ash::vk;

use crate::graphics;


pub trait RenderResource: Into<AnyResource> {
    type RawResource: RawRenderResource;
}

impl RenderResource for graphics::Buffer {
    type RawResource = graphics::BufferRaw;
}

impl RenderResource for graphics::BufferRaw {
    type RawResource = graphics::BufferRaw;
}

impl RenderResource for graphics::Image {
    type RawResource = graphics::ImageRaw;
}

impl RenderResource for graphics::ImageRaw {
    type RawResource = graphics::ImageRaw;
}

pub trait RawRenderResource: Into<graphics::AnyResource> {
    type Desc: Into<graphics::AnyResourceDesc>;

    fn create_raw(
        device: &graphics::Device,
        name: Cow<'static, str>,
        desc: &Self::Desc,
        descriptor_index: Option<graphics::DescriptorIndex>
    ) -> Self;
    
    fn destroy_raw(&self, device: &graphics::Device);
    
    fn desc(&self) -> &Self::Desc;
    fn descriptor_index(&self) -> Option<graphics::DescriptorIndex>;
    fn extract(any: AnyResourceRef) -> Option<&Self>;
}

impl RawRenderResource for graphics::BufferRaw {
    type Desc = graphics::BufferDesc;
    
    fn create_raw(
        device: &graphics::Device,
        name: Cow<'static, str>,
        desc: &Self::Desc,
        descriptor_index: Option<graphics::DescriptorIndex>
    ) -> Self {
        graphics::BufferRaw::create_impl(device, name, desc, descriptor_index)
    }
    
    fn destroy_raw(&self, device: &graphics::Device) {
        graphics::BufferRaw::destroy_impl(device, self);
    }

    fn desc(&self) -> &Self::Desc {
        &self.desc
    }

    fn descriptor_index(&self) -> Option<graphics::DescriptorIndex> {
        self.descriptor_index
    }

    fn extract(any: AnyResourceRef) -> Option<&Self> {
        match any {
            AnyResourceRef::Buffer(buffer) => Some(buffer),
            AnyResourceRef::Image(_)       => None,
        }
    }
}

impl RawRenderResource for graphics::ImageRaw {
    type Desc = graphics::ImageDesc;
    
    fn create_raw(
        device: &graphics::Device,
        name: Cow<'static, str>,
        desc: &Self::Desc,
        descriptor_index: Option<graphics::DescriptorIndex>
    ) -> Self {
        graphics::ImageRaw::create_impl(device, name, desc, descriptor_index)
    }
    
    fn destroy_raw(&self, device: &graphics::Device) {
        graphics::ImageRaw::destroy_impl(device, self);
    }

    fn desc(&self) -> &Self::Desc {
        &self.desc
    }

    fn descriptor_index(&self) -> Option<graphics::DescriptorIndex> {
        (!self._descriptor_flags.is_empty()).then_some(self._descriptor_index)
    }

    fn extract(any: AnyResourceRef) -> Option<&Self> {
        match any {
            AnyResourceRef::Buffer(_)    => None,
            AnyResourceRef::Image(image) => Some(image),
        }
    }
}

#[derive(Debug)]
pub enum AnyResource {
    Shared(graphics::AnyResourceShared),
    Owned(graphics::AnyResourceOwned),
}

impl From<graphics::Buffer> for AnyResource {
    fn from(value: graphics::Buffer) -> Self {
        AnyResource::Shared(AnyResourceShared::Buffer(value))
    }
}

impl From<graphics::BufferRaw> for AnyResource {
    fn from(value: graphics::BufferRaw) -> Self {
        AnyResource::Owned(AnyResourceOwned::Buffer(value))
    }
}

impl From<graphics::Image> for AnyResource {
    fn from(value: graphics::Image) -> Self {
        AnyResource::Shared(AnyResourceShared::Image(value))
    }
}

impl From<graphics::ImageRaw> for AnyResource {
    fn from(value: graphics::ImageRaw) -> Self {
        AnyResource::Owned(AnyResourceOwned::Image(value))
    }
}

impl AnyResource {
    #[inline(always)]
    pub fn as_ref(&self) -> graphics::AnyResourceRef {
        match self {
            AnyResource::Shared(imported) => imported.as_any_ref(),
            AnyResource::Owned(transient) => transient.as_any_ref(),
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
            AnyResource::Shared(res) => res.kind(),
            AnyResource::Owned(res) => res.kind(),
        }    
    }
}

#[derive(Debug, Clone)]
pub enum AnyResourceShared {
    Buffer(graphics::Buffer),
    Image(graphics::Image),
}

impl From<graphics::Buffer> for graphics::AnyResourceShared {
    fn from(value: graphics::Buffer) -> Self {
        graphics::AnyResourceShared::Buffer(value)
    }
}

impl From<graphics::Image> for graphics::AnyResourceShared {
    fn from(value: graphics::Image) -> Self {
        graphics::AnyResourceShared::Image(value)
    }
}

impl graphics::AnyResourceShared {
    #[inline(always)]
    pub fn as_any_ref(&self) -> graphics::AnyResourceRef {
        match self {
            AnyResourceShared::Buffer(buffer) => graphics::AnyResourceRef::Buffer(&buffer),
            AnyResourceShared::Image(image)   => graphics::AnyResourceRef::Image(&image),
        }
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

#[derive(Debug, Clone)]
pub enum AnyResourceOwned {
    Buffer(graphics::BufferRaw),
    Image(graphics::ImageRaw),
}

impl graphics::AnyResourceOwned {
    #[inline(always)]
    pub fn as_any_ref(&self) -> graphics::AnyResourceRef {
        match self {
            AnyResourceOwned::Buffer(buffer) => graphics::AnyResourceRef::Buffer(&buffer),
            AnyResourceOwned::Image(image)   => graphics::AnyResourceRef::Image(&image),
        }
    }
}

impl From<graphics::BufferRaw> for graphics::AnyResourceOwned {
    fn from(value: graphics::BufferRaw) -> Self {
        graphics::AnyResourceOwned::Buffer(value)
    }
}

impl From<graphics::ImageRaw> for graphics::AnyResourceOwned {
    fn from(value: graphics::ImageRaw) -> Self {
        graphics::AnyResourceOwned::Image(value)
    }
}

impl AnyResourceOwned {
    pub fn get_buffer(&self) -> Option<&graphics::BufferRaw> {
        match self {
            AnyResourceOwned::Buffer(buffer) => Some(buffer),
            AnyResourceOwned::Image(_) => None,
        }
    }

    pub fn get_image(&self) -> Option<&graphics::ImageRaw> {
        match self {
            AnyResourceOwned::Buffer(_) => None,
            AnyResourceOwned::Image(image) => Some(image),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AnyResourceView {
    Buffer(graphics::BufferView),
    Image(graphics::ImageView),
}

impl AnyResourceView {
    pub fn handle(&self) -> AnyResourceHandle {
        match self {
            AnyResourceView::Buffer(buffer) => AnyResourceHandle::Buffer(buffer.handle),
            AnyResourceView::Image(image) => AnyResourceHandle::Image(image.handle),
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

// impl RawRenderResource for AnyResourceOwned {
//     type Desc = AnyResourceDesc;

//     fn create_raw(
//         device: &graphics::Device,
//         name: Cow<'static, str>,
//         desc: &Self::Desc,
//         descriptor_index: Option<graphics::DescriptorIndex>
//     ) -> Self {
//         match desc {
//             AnyResourceDesc::Buffer(desc) => AnyResourceOwned::Buffer(
//                 graphics::BufferRaw::create_impl(device, name, desc, descriptor_index)
//             ),
//             AnyResourceDesc::Image(desc) => AnyResourceOwned::Image(
//                 graphics::ImageRaw::create_impl(device, name, desc, descriptor_index)
//             ),
//         }
//     }

//     fn destroy_raw(&self, device: &graphics::Device) {
//         match self {
//             AnyResourceOwned::Buffer(buffer) => buffer.destroy_raw(device),
//             AnyResourceOwned::Image(image) => image.destroy_raw(device),
//         }
//     }

//     fn desc(&self) -> &Self::Desc {
//         todo!()
//     }

//     fn descriptor_index(&self) -> Option<graphics::DescriptorIndex> {
//         match self {
//             AnyResourceOwned::Buffer(buffer) => buffer.descriptor_index,
//             AnyResourceOwned::Image(image) =>
//                 (!image._descriptor_flags.is_empty()).then_some(image._descriptor_index),
//         }
//     }
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Buffer,
    Image,
}

impl AnyResourceOwned {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceOwned::Buffer(_)  => ResourceKind::Buffer,
            AnyResourceOwned::Image(_)   => ResourceKind::Image,
        }
    }

    pub fn desc(&self) -> AnyResourceDesc {
        match self {
            AnyResourceOwned::Buffer(buffer) => AnyResourceDesc::Buffer(buffer.desc),
            AnyResourceOwned::Image(image)   => AnyResourceDesc::Image(image.desc),
        }
    }

    pub fn create(
        device: &graphics::Device,
        name: Cow<'static, str>,
        desc: &AnyResourceDesc,
        preallocated_descriptor_index: Option<graphics::DescriptorIndex>,
    ) -> Self {
        match desc {
            AnyResourceDesc::Buffer(desc) => {
                AnyResourceOwned::Buffer(graphics::BufferRaw::create_impl(
                    device,
                    name,
                    desc,
                    preallocated_descriptor_index,
                ))
            },
            AnyResourceDesc::Image(desc) => {
                AnyResourceOwned::Image(graphics::ImageRaw::create_impl(
                    device,
                    name,
                    desc,
                    preallocated_descriptor_index,
                ))
            },
        }
    }

    pub fn destroy(&self, device: &graphics::Device) {
        match self {
            AnyResourceOwned::Buffer(buffer) => {
                graphics::BufferRaw::destroy_impl(device, buffer);
            },
            AnyResourceOwned::Image(image) => {
                graphics::ImageRaw::destroy_impl(device, image);
            },
        }
    }
}

impl AnyResourceShared {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceShared::Buffer(_) => ResourceKind::Buffer,
            AnyResourceShared::Image(_)  => ResourceKind::Image,
        }
    }
}

impl AnyResourceView {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceView::Buffer(_)  => ResourceKind::Buffer,
            AnyResourceView::Image(_)   => ResourceKind::Image,
        }
    }
}

impl AnyResourceDesc {
    pub fn kind(&self) -> ResourceKind {
        match self {
            AnyResourceDesc::Buffer(_)  => ResourceKind::Buffer,
            AnyResourceDesc::Image(_)   => ResourceKind::Image,
        }
    }
}