mod device;
pub use device::*;

mod swapchain;
pub use swapchain::*;

mod commands;
pub use commands::*;

mod sync;
pub use sync::*;

mod context;
pub use context::*;

mod buffer;
pub use buffer::*;

mod image;
pub use self::image::*;

mod resource;
pub use resource::*;

mod pipeline;
pub use pipeline::*;

mod graph;
pub use graph::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuDrawIndiexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
