use crate::render;
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadWriteKind {
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AccessKind {
    #[default]
    None,
    IndirectBuffer,
    IndexBuffer,
    VertexBuffer,
    VertexShaderRead,
    VertexShaderWrite,
    FragmentShaderRead,
    FragmentShaderWrite,
    ColorAttachmentRead,
    ColorAttachmentWrite,
    DepthAttachmentRead,
    DepthAttachmentWrite,
    ComputeShaderRead,
    ComputeShaderWrite,
    TransferRead,
    TransferWrite,
    Present,
}

impl AccessKind {
    #[rustfmt::skip]
    #[inline(always)]
    pub fn read_write_kind(self) -> ReadWriteKind {
        match self {
            AccessKind::None                    => ReadWriteKind::Read,
            AccessKind::IndirectBuffer          => ReadWriteKind::Read,
            AccessKind::IndexBuffer             => ReadWriteKind::Read,
            AccessKind::VertexBuffer            => ReadWriteKind::Read,
            AccessKind::VertexShaderRead        => ReadWriteKind::Read,
            AccessKind::VertexShaderWrite       => ReadWriteKind::Write,
            AccessKind::FragmentShaderRead      => ReadWriteKind::Read,
            AccessKind::FragmentShaderWrite     => ReadWriteKind::Write,
            AccessKind::ColorAttachmentRead     => ReadWriteKind::Read,
            AccessKind::ColorAttachmentWrite    => ReadWriteKind::Write,
            AccessKind::DepthAttachmentRead     => ReadWriteKind::Read,
            AccessKind::DepthAttachmentWrite    => ReadWriteKind::Write,
            AccessKind::ComputeShaderRead       => ReadWriteKind::Read,
            AccessKind::ComputeShaderWrite      => ReadWriteKind::Write,
            AccessKind::Present                 => ReadWriteKind::Read,
            AccessKind::TransferRead            => ReadWriteKind::Write,
            AccessKind::TransferWrite           => ReadWriteKind::Read,
            
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    pub fn stage_mask(self) -> vk::PipelineStageFlags2 {
        match self {
            AccessKind::None                    => vk::PipelineStageFlags2::NONE,
            AccessKind::IndirectBuffer          => vk::PipelineStageFlags2::DRAW_INDIRECT,
            AccessKind::IndexBuffer             => vk::PipelineStageFlags2::INDEX_INPUT,
            AccessKind::VertexBuffer            => vk::PipelineStageFlags2::VERTEX_INPUT,
            AccessKind::VertexShaderRead        => vk::PipelineStageFlags2::VERTEX_SHADER,
            AccessKind::VertexShaderWrite       => vk::PipelineStageFlags2::VERTEX_SHADER,
            AccessKind::FragmentShaderRead      => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            AccessKind::FragmentShaderWrite     => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            AccessKind::ColorAttachmentRead     => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            AccessKind::ColorAttachmentWrite    => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            AccessKind::DepthAttachmentRead     => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            AccessKind::DepthAttachmentWrite    => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            AccessKind::ComputeShaderRead       => vk::PipelineStageFlags2::COMPUTE_SHADER,
            AccessKind::ComputeShaderWrite      => vk::PipelineStageFlags2::COMPUTE_SHADER,
            AccessKind::Present                 => vk::PipelineStageFlags2::NONE,
            AccessKind::TransferRead            => vk::PipelineStageFlags2::TRANSFER,
            AccessKind::TransferWrite           => vk::PipelineStageFlags2::TRANSFER,
            
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    pub fn access_mask(self) -> vk::AccessFlags2 {
        match self {
            AccessKind::None                    => vk::AccessFlags2::NONE,
            AccessKind::IndirectBuffer          => vk::AccessFlags2::INDIRECT_COMMAND_READ,
            AccessKind::IndexBuffer             => vk::AccessFlags2::INDEX_READ,
            AccessKind::VertexBuffer            => vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            AccessKind::VertexShaderRead        => vk::AccessFlags2::SHADER_READ,
            AccessKind::VertexShaderWrite       => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::FragmentShaderRead      => vk::AccessFlags2::SHADER_READ,
            AccessKind::FragmentShaderWrite     => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::ColorAttachmentRead     => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            AccessKind::ColorAttachmentWrite    => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            AccessKind::DepthAttachmentRead     => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            AccessKind::DepthAttachmentWrite    => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            AccessKind::ComputeShaderRead       => vk::AccessFlags2::SHADER_READ,
            AccessKind::ComputeShaderWrite      => vk::AccessFlags2::SHADER_WRITE,
            AccessKind::Present                 => vk::AccessFlags2::NONE,
            AccessKind::TransferRead            => vk::AccessFlags2::TRANSFER_READ,
            AccessKind::TransferWrite           => vk::AccessFlags2::TRANSFER_WRITE,
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    pub fn image_layout(self) -> vk::ImageLayout {
        match self {
            AccessKind::None                    => vk::ImageLayout::UNDEFINED,
            AccessKind::IndirectBuffer          => vk::ImageLayout::UNDEFINED,
            AccessKind::IndexBuffer             => vk::ImageLayout::UNDEFINED,
            AccessKind::VertexBuffer            => vk::ImageLayout::UNDEFINED,
            AccessKind::VertexShaderRead        => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::VertexShaderWrite       => vk::ImageLayout::GENERAL,
            AccessKind::FragmentShaderRead      => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::FragmentShaderWrite     => vk::ImageLayout::GENERAL,
            AccessKind::ColorAttachmentRead     => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            AccessKind::ColorAttachmentWrite    => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            AccessKind::DepthAttachmentRead     => vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            AccessKind::DepthAttachmentWrite    => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            AccessKind::ComputeShaderRead       => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            AccessKind::ComputeShaderWrite      => vk::ImageLayout::GENERAL,
            AccessKind::Present                 => vk::ImageLayout::PRESENT_SRC_KHR,
            AccessKind::TransferRead            => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            AccessKind::TransferWrite           => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            
        }
    }

}

// #[inline]
// pub fn buffer_barrier(
//     buffer: &vulkan::Buffer,
//     src_access: vulkan::AccessKind,
//     dst_access: vulkan::AccessKind,
// ) -> vk::BufferMemoryBarrier2 {
//     vk::BufferMemoryBarrier2 {
//         src_stage_mask: src_access.stage_mask(),
//         src_access_mask: src_access.access_mask(),
//         dst_stage_mask: dst_access.stage_mask(),
//         dst_access_mask: dst_access.access_mask(),
//         src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
//         dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
//         buffer: buffer.handle,
//         offset: 0,
//         size: vk::WHOLE_SIZE,
//         ..Default::default()
//     }
// }

#[inline]
pub fn image_barrier(
    image: &render::ImageView,
    src_access: AccessKind,
    dst_access: AccessKind,
) -> vk::ImageMemoryBarrier2 {
    vk::ImageMemoryBarrier2 {
        src_stage_mask: src_access.stage_mask(),
        src_access_mask: src_access.access_mask(),
        dst_stage_mask: dst_access.stage_mask(),
        dst_access_mask: dst_access.access_mask(),
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        old_layout: src_access.image_layout(),
        new_layout: dst_access.image_layout(),
        image: image.handle,
        subresource_range: image.subresource_range,
        ..Default::default()
    }
}
