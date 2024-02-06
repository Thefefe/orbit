use std::{borrow::Cow, ops::RangeBounds};

use crate::graphics;
use ash::vk;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AccessKind {
    #[default]
    None,
    IndirectBuffer,
    IndexBuffer,
    VertexBuffer,
    AllGraphicsRead,
    AllGraphicsReadGeneral,
    AllGraphicsWrite,
    PreRasterizationRead,
    PreRasterizationReadGeneral,
    PreRasterizationWrite,
    TaskShaderRead,
    TaskShaderReadGeneral,
    TaskShaderWrite,
    MeshShaderRead,
    MeshShaderWrite,
    VertexShaderRead,
    VertexShaderWrite,
    FragmentShaderRead,
    FragmentShaderReadGeneral,
    FragmentShaderWrite,
    ColorAttachmentRead,
    ColorAttachmentWrite,
    DepthAttachmentRead,
    DepthAttachmentWrite,
    ComputeShaderRead,
    ComputeShaderReadGeneral,
    ComputeShaderWrite,
    TransferRead,
    TransferWrite,
    Present,
}

impl AccessKind {
    #[rustfmt::skip]
    #[inline(always)]
    pub fn writes(self) -> bool {
        match self {
            AccessKind::None                        => false,
            AccessKind::IndirectBuffer              => false,
            AccessKind::IndexBuffer                 => false,
            AccessKind::VertexBuffer                => false,
            AccessKind::AllGraphicsRead             => false,
            AccessKind::AllGraphicsReadGeneral      => false,
            AccessKind::AllGraphicsWrite            => true,
            AccessKind::PreRasterizationRead        => false,
            AccessKind::PreRasterizationReadGeneral => false,
            AccessKind::PreRasterizationWrite       => true,
            AccessKind::TaskShaderRead              => false,
            AccessKind::TaskShaderReadGeneral       => false,
            AccessKind::TaskShaderWrite             => true,
            AccessKind::MeshShaderRead              => false,
            AccessKind::MeshShaderWrite             => true,
            AccessKind::VertexShaderRead            => false,
            AccessKind::VertexShaderWrite           => true,
            AccessKind::FragmentShaderRead          => false,
            AccessKind::FragmentShaderReadGeneral   => false,
            AccessKind::FragmentShaderWrite         => true,
            AccessKind::ColorAttachmentRead         => false,
            AccessKind::ColorAttachmentWrite        => true,
            AccessKind::DepthAttachmentRead         => false,
            AccessKind::DepthAttachmentWrite        => true,
            AccessKind::ComputeShaderRead           => false,
            AccessKind::ComputeShaderReadGeneral    => false,
            AccessKind::ComputeShaderWrite          => true,
            AccessKind::Present                     => false,
            AccessKind::TransferRead                => true,
            AccessKind::TransferWrite               => false,
            
        }
    }

    pub fn read_only(self) -> bool {
        !self.writes()
    }

    #[rustfmt::skip]
    #[inline(always)]
    pub fn stage_mask(self) -> vk::PipelineStageFlags2 {
        match self {
            AccessKind::None                        => vk::PipelineStageFlags2::NONE,
            AccessKind::IndirectBuffer              => vk::PipelineStageFlags2::DRAW_INDIRECT,
            AccessKind::IndexBuffer                 => vk::PipelineStageFlags2::INDEX_INPUT,
            AccessKind::VertexBuffer                => vk::PipelineStageFlags2::VERTEX_INPUT,
            AccessKind::AllGraphicsRead             => vk::PipelineStageFlags2::ALL_GRAPHICS,
            AccessKind::AllGraphicsReadGeneral      => vk::PipelineStageFlags2::ALL_GRAPHICS,
            AccessKind::AllGraphicsWrite            => vk::PipelineStageFlags2::ALL_GRAPHICS,
            AccessKind::PreRasterizationRead        => vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS,
            AccessKind::PreRasterizationReadGeneral => vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS,
            AccessKind::PreRasterizationWrite       => vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS,
            AccessKind::TaskShaderRead              => vk::PipelineStageFlags2::TASK_SHADER_EXT,
            AccessKind::TaskShaderReadGeneral       => vk::PipelineStageFlags2::TASK_SHADER_EXT,
            AccessKind::TaskShaderWrite             => vk::PipelineStageFlags2::TASK_SHADER_EXT,
            AccessKind::MeshShaderRead              => vk::PipelineStageFlags2::MESH_SHADER_EXT,
            AccessKind::MeshShaderWrite             => vk::PipelineStageFlags2::MESH_SHADER_EXT,
            AccessKind::VertexShaderRead            => vk::PipelineStageFlags2::VERTEX_SHADER,
            AccessKind::VertexShaderWrite           => vk::PipelineStageFlags2::VERTEX_SHADER,
            AccessKind::FragmentShaderRead          => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            AccessKind::FragmentShaderReadGeneral   => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            AccessKind::FragmentShaderWrite         => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            AccessKind::ColorAttachmentRead         => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            AccessKind::ColorAttachmentWrite        => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            AccessKind::DepthAttachmentRead         => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            AccessKind::DepthAttachmentWrite        => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            AccessKind::ComputeShaderRead           => vk::PipelineStageFlags2::COMPUTE_SHADER,
            AccessKind::ComputeShaderReadGeneral    => vk::PipelineStageFlags2::COMPUTE_SHADER,
            AccessKind::ComputeShaderWrite          => vk::PipelineStageFlags2::COMPUTE_SHADER,
            AccessKind::Present                     => vk::PipelineStageFlags2::NONE,
            AccessKind::TransferRead                => vk::PipelineStageFlags2::TRANSFER,
            AccessKind::TransferWrite               => vk::PipelineStageFlags2::TRANSFER,
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    pub fn access_mask(self) -> vk::AccessFlags2 {
        match self {
            AccessKind::None                        => vk::AccessFlags2::NONE,
            AccessKind::IndirectBuffer              => vk::AccessFlags2::INDIRECT_COMMAND_READ,
            AccessKind::IndexBuffer                 => vk::AccessFlags2::INDEX_READ,
            AccessKind::VertexBuffer                => vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            AccessKind::AllGraphicsRead             => vk::AccessFlags2::SHADER_READ,
            AccessKind::AllGraphicsReadGeneral      => vk::AccessFlags2::SHADER_READ,
            AccessKind::AllGraphicsWrite            => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::PreRasterizationRead        => vk::AccessFlags2::SHADER_READ,
            AccessKind::PreRasterizationReadGeneral => vk::AccessFlags2::SHADER_READ,
            AccessKind::PreRasterizationWrite       => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::TaskShaderRead              => vk::AccessFlags2::SHADER_READ,
            AccessKind::TaskShaderReadGeneral       => vk::AccessFlags2::SHADER_READ,
            AccessKind::TaskShaderWrite             => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::MeshShaderRead              => vk::AccessFlags2::SHADER_READ,
            AccessKind::MeshShaderWrite             => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::VertexShaderRead            => vk::AccessFlags2::SHADER_READ,
            AccessKind::VertexShaderWrite           => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::FragmentShaderRead          => vk::AccessFlags2::SHADER_READ,
            AccessKind::FragmentShaderReadGeneral   => vk::AccessFlags2::SHADER_READ,
            AccessKind::FragmentShaderWrite         => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::ColorAttachmentRead         => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            AccessKind::ColorAttachmentWrite        => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            AccessKind::DepthAttachmentRead         => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            AccessKind::DepthAttachmentWrite        => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            AccessKind::ComputeShaderRead           => vk::AccessFlags2::SHADER_READ,
            AccessKind::ComputeShaderReadGeneral    => vk::AccessFlags2::SHADER_READ,
            AccessKind::ComputeShaderWrite          => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::Present                     => vk::AccessFlags2::NONE,
            AccessKind::TransferRead                => vk::AccessFlags2::TRANSFER_READ,
            AccessKind::TransferWrite               => vk::AccessFlags2::TRANSFER_WRITE,
            
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    pub fn image_layout(self) -> vk::ImageLayout {
        match self {
            AccessKind::None                        => vk::ImageLayout::UNDEFINED,
            AccessKind::IndirectBuffer              => vk::ImageLayout::UNDEFINED,
            AccessKind::IndexBuffer                 => vk::ImageLayout::UNDEFINED,
            AccessKind::VertexBuffer                => vk::ImageLayout::UNDEFINED,
            AccessKind::AllGraphicsRead             => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::AllGraphicsReadGeneral      => vk::ImageLayout::GENERAL,
            AccessKind::AllGraphicsWrite            => vk::ImageLayout::GENERAL,
            AccessKind::PreRasterizationRead        => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::PreRasterizationReadGeneral => vk::ImageLayout::GENERAL,
            AccessKind::PreRasterizationWrite       => vk::ImageLayout::GENERAL,
            AccessKind::TaskShaderRead              => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::TaskShaderReadGeneral       => vk::ImageLayout::GENERAL,
            AccessKind::TaskShaderWrite             => vk::ImageLayout::GENERAL,
            AccessKind::MeshShaderRead              => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::MeshShaderWrite             => vk::ImageLayout::GENERAL, 
            AccessKind::VertexShaderRead            => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::VertexShaderWrite           => vk::ImageLayout::GENERAL,
            AccessKind::FragmentShaderRead          => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::FragmentShaderReadGeneral   => vk::ImageLayout::GENERAL,
            AccessKind::FragmentShaderWrite         => vk::ImageLayout::GENERAL,
            AccessKind::ColorAttachmentRead         => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            AccessKind::ColorAttachmentWrite        => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            AccessKind::DepthAttachmentRead         => vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            AccessKind::DepthAttachmentWrite        => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            AccessKind::ComputeShaderRead           => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::ComputeShaderReadGeneral    => vk::ImageLayout::GENERAL,
            AccessKind::ComputeShaderWrite          => vk::ImageLayout::GENERAL,
            AccessKind::Present                     => vk::ImageLayout::PRESENT_SRC_KHR,
            AccessKind::TransferRead                => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            AccessKind::TransferWrite               => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            
        }
    }

    pub fn access_flags(self) -> AccessFlags {
        AccessFlags {
            stage_flags: self.stage_mask(),
            access_flags: self.access_mask(),
            layout: self.image_layout(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AccessFlags {
    pub stage_flags: vk::PipelineStageFlags2,
    pub access_flags: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
}

impl From<AccessKind> for AccessFlags {
    fn from(value: AccessKind) -> Self {
        Self {
            stage_flags: value.stage_mask(),
            access_flags: value.access_mask(),
            layout: value.image_layout(),
        }
    }
}

impl AccessFlags {
    pub fn extend_buffer_access(&mut self, access: AccessKind) {
        self.stage_flags = access.stage_mask();
        self.access_flags = access.access_mask();
    }

    #[track_caller]
    pub fn extend_image_access(&mut self, access: AccessKind) {
        if self.layout == vk::ImageLayout::UNDEFINED {
            self.layout = access.image_layout()
        }
        assert_eq!(self.layout, access.image_layout());
        self.stage_flags = access.stage_mask();
        self.access_flags = access.access_mask();
        self.layout = access.image_layout();
    }
}

#[inline]
pub fn buffer_barrier(
    buffer: &graphics::BufferView,
    src_access: impl Into<AccessFlags>,
    dst_access: impl Into<AccessFlags>,
) -> vk::BufferMemoryBarrier2 {
    let src_access = src_access.into();
    let dst_access = dst_access.into();
    vk::BufferMemoryBarrier2 {
        src_stage_mask: src_access.stage_flags,
        src_access_mask: src_access.access_flags,
        dst_stage_mask: dst_access.stage_flags,
        dst_access_mask: dst_access.access_flags,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        buffer: buffer.handle,
        offset: 0,
        size: vk::WHOLE_SIZE,
        ..Default::default()
    }
}

#[inline]
pub fn image_barrier(
    image: &graphics::ImageView,
    src_access: impl Into<AccessFlags>,
    dst_access: impl Into<AccessFlags>,
) -> vk::ImageMemoryBarrier2 {
    let src_access = src_access.into();
    let dst_access = dst_access.into();
    vk::ImageMemoryBarrier2 {
        src_stage_mask: src_access.stage_flags,
        src_access_mask: src_access.access_flags,
        dst_stage_mask: dst_access.stage_flags,
        dst_access_mask: dst_access.access_flags,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        old_layout: src_access.layout,
        new_layout: dst_access.layout,
        image: image.handle,
        subresource_range: image.subresource_range,
        ..Default::default()
    }
}

#[inline]
pub fn image_subresource_barrier(
    image: &graphics::ImageView,
    mip_level: impl RangeBounds<u32>,
    layers: impl RangeBounds<u32>,
    src_access: impl Into<AccessFlags>,
    dst_access: impl Into<AccessFlags>,
) -> vk::ImageMemoryBarrier2 {
    let src_access = src_access.into();
    let dst_access = dst_access.into();
    vk::ImageMemoryBarrier2 {
        src_stage_mask: src_access.stage_flags,
        src_access_mask: src_access.access_flags,
        dst_stage_mask: dst_access.stage_flags,
        dst_access_mask: dst_access.access_flags,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        old_layout: src_access.layout,
        new_layout: dst_access.layout,
        image: image.handle,
        subresource_range: image.subresource_range(mip_level, layers),
        ..Default::default()
    }
}

pub fn extend_memory_barrier(barrier: &mut vk::MemoryBarrier2, src_access: AccessFlags, dst_access: AccessFlags) {
    barrier.src_stage_mask |= src_access.stage_flags;
    barrier.src_access_mask |= src_access.access_flags;
    barrier.dst_stage_mask |= dst_access.stage_flags;
    barrier.dst_access_mask |= dst_access.access_flags;
}

pub fn is_memory_barrier_not_useless(barrier: &vk::MemoryBarrier2) -> bool {
    barrier.src_stage_mask != vk::PipelineStageFlags2::TOP_OF_PIPE
        || barrier.dst_stage_mask != vk::PipelineStageFlags2::BOTTOM_OF_PIPE
        || barrier.src_access_mask | barrier.dst_access_mask != vk::AccessFlags2::NONE
}

#[derive(Clone)]
pub struct Semaphore {
    pub name: Cow<'static, str>,
    pub handle: vk::Semaphore,
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.name.as_ref())
    }
}

impl graphics::Device {
    pub fn create_semaphore(&self, name: impl Into<Cow<'static, str>>) -> graphics::Semaphore {
        let name = name.into();
        let handle = unsafe { self.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() };
        self.set_debug_name(handle, &name);
        graphics::Semaphore { name, handle }
    }
}
