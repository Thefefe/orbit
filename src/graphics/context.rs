use std::{borrow::Cow, collections::HashMap, marker::PhantomData, ops::Range, sync::Arc, time::Instant};

use ash::vk;
use parking_lot::Mutex;
use rayon::prelude::*;
use winit::window::Window;

use crate::{
    assets::GpuMeshletDrawCommand,
    graphics::{self, QueueType},
    passes::draw_gen::MAX_DRAW_COUNT,
    utils,
};

use super::{
    graph::{self, CompiledRenderGraph, RenderGraph},
    AccessKind, Dispatch, TransientResourceCache,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuGlobalData {
    screen_size: [u32; 2],
    elapsed_frames: u32,
    elapsed_time: f32,
}

pub const FRAME_COUNT: usize = 2;
const MAX_TIMESTAMP_COUNT: u32 = 128;

pub struct Frame {
    first_time_use: bool,

    in_flight_fence_value: u64,
    image_available_semaphore: graphics::Semaphore,
    render_finished_semaphore: graphics::Semaphore,
    uses_async_transfer: bool,

    command_pools: Vec<Mutex<graphics::CommandPool>>,

    compiled_graph: CompiledRenderGraph,
    graph_debug_info: GraphDebugInfo,
    timestamp_query_pool: vk::QueryPool,
}

struct RecordSubmitStuff {
    command_pool: graphics::CommandPool,
    fence: vk::Fence,
}

#[derive(Debug, Clone)]
enum CopyDst {
    Buffer {
        dst_buffer: vk::Buffer,
        dst_offset: usize,
    },
    Image {
        dst_image: vk::Image,
        dst_mip_level: u32,
        dst_layers: Range<u32>,
        dst_offset: Option<vk::Offset3D>,
        dst_extent: vk::Extent3D,
        prev_access: AccessKind,
        is_concurent: bool,
    },
}

#[derive(Debug, Clone)]
struct CopyCommand {
    dst: CopyDst,
    src_offset: usize,
    size: usize,
}

struct StagingPage {
    command_pool: graphics::CommandPool,
    fence_value: u64,
}

const STAGING_PAGE_SIZE: usize = 12 * 1024 * 1024;
const STAGING_PAGE_COUNT: usize = 3;

struct StagedTransferQueue {
    staging_buffer: graphics::BufferRaw,
    pages: Vec<StagingPage>,
    page_index: usize,
    page_size: usize,

    pending_data: Vec<u8>,
    pending_copy_commands: Vec<CopyCommand>,
    pending_image_transfers: Vec<ImagePostTransferOp>,

    pre_transfer_command_pool: graphics::CommandPool,
    pre_transfer_fence: u64,

    last_fence_value: u64,
}

impl StagedTransferQueue {
    fn new(device: &graphics::Device, page_count: usize, page_size: usize) -> Self {
        assert!(page_count > 0, "at least one page is required");
        assert!(page_size >= 1024, "page_size must be at least 1024 bytes");
        // TODO: handle alignment based on gpu properties

        let staging_buffer = graphics::BufferRaw::create_impl(
            device,
            "transfer_queue_staging_buffer".into(),
            &graphics::BufferDesc {
                size: page_count * page_size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: gpu_allocator::MemoryLocation::CpuToGpu,
                ..Default::default()
            },
            None,
        );

        let pages = Vec::from_iter((0..page_count).map(|_| {
            let command_pool =
                graphics::CommandPool::new(device, "staged_transfer_command_pool", QueueType::AsyncTransfer);

            StagingPage {
                command_pool,
                fence_value: 0,
            }
        }));

        let pre_transfer_command_pool =
            graphics::CommandPool::new(device, "staged_pre_transfer_command_pool", QueueType::Graphics);

        Self {
            staging_buffer,
            pages,
            page_index: 0,
            page_size,

            pending_data: Vec::new(),
            pending_copy_commands: Vec::new(),

            pending_image_transfers: Vec::new(),

            pre_transfer_command_pool,
            pre_transfer_fence: 0,

            last_fence_value: 0,
        }
    }

    fn destroy(&self, device: &graphics::Device) {
        for page in self.pages.iter() {
            page.command_pool.destroy(device);
        }
        self.pre_transfer_command_pool.destroy(device);
        graphics::BufferRaw::destroy_impl(device, &self.staging_buffer);
    }

    fn pending_bytes(&self) -> usize {
        self.pending_data.len()
    }

    #[track_caller]
    fn queue_write_buffer(&mut self, buffer: &graphics::BufferView, offset: usize, data: &[u8]) {
        puffin::profile_function!();
        assert!(
            data.len() <= self.page_size,
            "copies bigger then the page size isn't yet supported"
        );

        if data.len() == 0 {
            return;
        }

        let src_offset = self.pending_data.len();
        self.pending_data.extend_from_slice(data);
        self.pending_copy_commands.push(CopyCommand {
            dst: CopyDst::Buffer {
                dst_buffer: buffer.handle,
                dst_offset: offset,
            },
            src_offset,
            size: data.len(),
        });
    }

    #[track_caller]
    fn queue_write_image(
        &mut self,
        image: &graphics::ImageRaw,
        mip_level: u32,
        layers: Range<u32>,
        region: Option<(vk::Offset3D, vk::Extent3D)>,
        prev_access: AccessKind,
        data: &[u8],
    ) {
        puffin::profile_function!();
        assert!(
            data.len() <= self.page_size,
            "copies bigger then the page size isn't yet supported"
        );
        assert_eq!(image.desc.aspect, vk::ImageAspectFlags::COLOR);

        if data.len() == 0 {
            return;
        }

        let (dst_offset, dst_extent) = region.unwrap_or((vk::Offset3D::default(), image.extent));
        let dst_offset = region.is_some().then_some(dst_offset);
        let src_offset = self.pending_data.len();
        self.pending_data.extend_from_slice(data);
        self.pending_copy_commands.push(CopyCommand {
            dst: CopyDst::Image {
                dst_image: image.handle,
                dst_mip_level: mip_level,
                dst_layers: layers.clone(),
                dst_offset,
                dst_extent,
                prev_access,
                is_concurent: image.desc.sharing_mode == graphics::SharingMode::Concurent
            },
            src_offset,
            size: data.len(),
        });
    }

    fn fence_value_after_submit(&self) -> u64 {
        self.last_fence_value + self.pending_bytes().div_ceil(self.page_size) as u64
    }

    fn queue_pre_transfer(&mut self, device: &graphics::Device, copy_range: Range<usize>) -> Option<u64> {
        let src_queue_family_index = device.get_queue(QueueType::Graphics).family.index;
        let dst_queue_family_index = device.get_queue(QueueType::AsyncTransfer).family.index;

        if src_queue_family_index == dst_queue_family_index {
            return None;
        }

        let mut image_barriers = Vec::new();
        for copy_command in &self.pending_copy_commands[copy_range] {
            if let CopyDst::Image {
                dst_image,
                dst_mip_level,
                dst_layers,
                dst_offset,
                prev_access,
                is_concurent,
                ..
            } = copy_command.dst.clone()
            {
                if dst_offset.is_some() && !is_concurent {
                    let barrier = vk::ImageMemoryBarrier2 {
                        src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                        src_access_mask: vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
                        old_layout: prev_access.image_layout(),
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        src_queue_family_index,
                        dst_queue_family_index,
                        image: dst_image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: dst_mip_level,
                            level_count: 1,
                            base_array_layer: dst_layers.start,
                            layer_count: dst_layers.len().try_into().unwrap(),
                        },
                        ..Default::default()
                    };
                    image_barriers.push(barrier);
                }
            }
        }

        if image_barriers.is_empty() {
            return None;
        }

        device.wait_queue_semaphore(QueueType::Graphics, self.pre_transfer_fence, None).unwrap();
        self.pre_transfer_command_pool.reset(&device);

        let recorder = self.pre_transfer_command_pool.begin_new(&device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        recorder.barrier(&[], &image_barriers, &[]);

        drop(recorder);

        self.pre_transfer_fence =
            device.queue_submit(QueueType::Graphics, self.pre_transfer_command_pool.submit_info());

        Some(self.pre_transfer_fence)
    }

    fn submit(&mut self, device: &graphics::Device) -> Option<u64> {
        puffin::profile_function!();
        if self.pending_copy_commands.is_empty() {
            return None;
        }

        let mut copy_cursor = 0;
        // let mut byte_cursor = 0;
        loop {
            puffin::profile_scope!("transfer_submit_page");
            let buffer_offset = self.page_index * self.page_size;

            let mut local_byte_cursor = 0;
            let mut local_copy_count = 0;
            loop {
                let copy_index = copy_cursor + local_copy_count;

                if copy_index >= self.pending_copy_commands.len() {
                    break;
                }

                let required_alignment = match &self.pending_copy_commands[copy_index].dst {
                    CopyDst::Buffer { .. } => 4,
                    // TODO: get the alignment of the associated format, meanwhile 64 is working for most formats
                    CopyDst::Image { .. } => 64,
                };
                let padding = local_byte_cursor % required_alignment;

                if local_byte_cursor + self.pending_copy_commands[copy_index].size + padding > self.page_size {
                    break;
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.pending_data.as_ptr().add(self.pending_copy_commands[copy_index].src_offset),
                        self.staging_buffer.mapped_ptr.unwrap().as_ptr().add(buffer_offset + local_byte_cursor + padding),
                        self.pending_copy_commands[copy_index].size,
                    );
                }
                self.pending_copy_commands[copy_index].src_offset = buffer_offset + local_byte_cursor + padding;

                local_byte_cursor += self.pending_copy_commands[copy_index].size + padding;
                local_copy_count += 1;
            }

            device.wait_queue_semaphore(QueueType::AsyncTransfer, self.pages[self.page_index].fence_value, None)
                .unwrap();

            let copy_range = copy_cursor..copy_cursor + local_copy_count;
            let pre_transfer_fence = self.queue_pre_transfer(device, copy_range.clone());

            self.pages[self.page_index].command_pool.reset(device);
            let mut recorder = self.pages[self.page_index]
                .command_pool
                .begin_new(device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            if let Some(fence) = pre_transfer_fence {
                recorder.wait_queue(QueueType::Graphics, vk::PipelineStageFlags2::TRANSFER, fence);
            }

            {
                puffin::profile_scope!("record_copy_commands");
                recorder.begin_debug_label("async_copy", Some([1.0, 1.0, 0.0, 1.0]));

                let graphics_queue_index = device.get_queue(QueueType::Graphics).family.index;
                let transfer_queue_index = device.get_queue(QueueType::AsyncTransfer).family.index;

                let mut image_barriers = Vec::new();
                for command in &self.pending_copy_commands[copy_range.clone()] {
                    if let CopyDst::Image {
                        dst_image,
                        dst_mip_level,
                        dst_layers,
                        dst_offset,
                        prev_access,
                        is_concurent,
                        ..
                    } = command.dst.clone()
                    {
                        let discard = dst_offset.is_none();
                        let mut barrier = vk::ImageMemoryBarrier2 {
                            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                            dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                            old_layout: if discard {
                                vk::ImageLayout::UNDEFINED
                            } else {
                                prev_access.image_layout()
                                // vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            },
                            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            image: dst_image,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: dst_mip_level,
                                level_count: 1,
                                base_array_layer: dst_layers.start,
                                layer_count: dst_layers.len().try_into().unwrap(),
                            },
                            ..Default::default()
                        };

                        if !discard && graphics_queue_index != transfer_queue_index && !is_concurent {
                            barrier.src_queue_family_index = graphics_queue_index;
                            barrier.dst_queue_family_index = transfer_queue_index;
                        }
                        image_barriers.push(barrier);
                    }
                }

                recorder.barrier(&[], &image_barriers, &[]);

                for command in &self.pending_copy_commands[copy_range.clone()] {
                    match command.dst.clone() {
                        CopyDst::Buffer { dst_buffer, dst_offset } => {
                            let region = vk::BufferCopy {
                                src_offset: command.src_offset.try_into().unwrap(),
                                dst_offset: dst_offset.try_into().unwrap(),
                                size: command.size.try_into().unwrap(),
                            };

                            recorder.copy_buffer(self.staging_buffer.handle, dst_buffer, &[region]);
                        }
                        CopyDst::Image {
                            dst_image,
                            dst_mip_level,
                            dst_layers,
                            dst_offset,
                            dst_extent,
                            ..
                        } => {
                            recorder.copy_buffer_to_image(
                                self.staging_buffer.handle,
                                dst_image,
                                &[vk::BufferImageCopy {
                                    buffer_offset: command.src_offset.try_into().unwrap(),
                                    buffer_row_length: 0,
                                    buffer_image_height: 0,
                                    image_subresource: vk::ImageSubresourceLayers {
                                        aspect_mask: vk::ImageAspectFlags::COLOR,
                                        mip_level: dst_mip_level,
                                        base_array_layer: dst_layers.start,
                                        layer_count: dst_layers.len().try_into().unwrap(),
                                    },
                                    // image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                                    image_offset: dst_offset.unwrap_or_default(),
                                    image_extent: dst_extent,
                                }],
                            );
                        }
                    }
                }

                let transfer_queue = device.get_queue(QueueType::AsyncTransfer).family.index;
                let graphics_queue = device.get_queue(QueueType::Graphics).family.index;

                if transfer_queue != graphics_queue {
                    for image_barrier in image_barriers.iter_mut() {
                        let queue_transfer = image_barrier.src_queue_family_index != image_barrier.dst_queue_family_index;
                        *image_barrier = vk::ImageMemoryBarrier2 {
                            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                            dst_access_mask: vk::AccessFlags2::NONE,

                            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,

                            ..*image_barrier
                        };

                        if queue_transfer {
                            image_barrier.src_queue_family_index = transfer_queue;
                            image_barrier.dst_queue_family_index = graphics_queue;
                        }

                        let base_layer = image_barrier.subresource_range.base_array_layer;
                        let layer_count = image_barrier.subresource_range.layer_count;
                        let level = image_barrier.subresource_range.base_mip_level;
                        let layers = base_layer..base_layer + layer_count;

                        self.pending_image_transfers.push(ImagePostTransferOp {
                            image: image_barrier.image,
                            subresource: ImageSubresource::Layers { level, layers },
                            queue_transfer: queue_transfer.then_some((transfer_queue, graphics_queue)),
                            access: AccessKind::TransferWrite,
                        });
                    }
                }

                recorder.barrier(&[], &image_barriers, &[]);

                recorder.end_debug_label();
            }

            drop(recorder);

            let fence_value = device.queue_submit(
                QueueType::AsyncTransfer,
                self.pages[self.page_index].command_pool.submit_info(),
            );
            self.pages[self.page_index].fence_value = fence_value;
            self.last_fence_value = fence_value;

            copy_cursor += local_copy_count;
            // byte_cursor += local_byte_cursor;

            self.page_index = (self.page_index + 1) % self.pages.len();

            if copy_cursor == self.pending_copy_commands.len() {
                break;
            }
        }

        self.pending_copy_commands.clear();
        self.pending_data.clear();

        Some(self.last_fence_value)
    }
}

struct FrameContext {
    acquired_image: graphics::AcquiredImage,
    acquired_image_handle: GraphImageHandle,
}

enum QueueOwnership {
    Owned { queue: QueueType },
    Released { src_queue: QueueType, dst_queue: QueueType },
}

enum ImageSubresource {
    Full,
    MipLevel { level: u32 },
    Layers { level: u32, layers: Range<u32> },
}

impl ImageSubresource {
    fn to_subresource_range(&self) -> vk::ImageSubresourceRange {
        match self {
            ImageSubresource::Full => vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            ImageSubresource::MipLevel { level } => vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: *level,
                level_count: 1,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            ImageSubresource::Layers { level, layers } => vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: *level,
                level_count: 1,
                base_array_layer: layers.start,
                layer_count: layers.len().try_into().unwrap(),
            },
        }
    }
}

struct ImagePostTransferOp {
    image: vk::Image,
    subresource: ImageSubresource,
    queue_transfer: Option<(u32, u32)>,
    access: AccessKind,
}

pub struct Context {
    pub window: Window,

    pub device: Arc<graphics::Device>,
    pub swapchain: graphics::Swapchain,

    pub shader_modules: HashMap<graphics::ShaderSource, vk::ShaderModule>,
    pub raster_pipelines: HashMap<graphics::RasterPipelineDesc, graphics::RasterPipeline>,
    pub compute_pipelines: HashMap<graphics::ShaderStage, graphics::ComputePipeline>,

    pub graph: RenderGraph,
    transient_resource_cache: TransientResourceCache,

    pub frames: [Frame; FRAME_COUNT],
    pub frame_index: usize,
    pub elapsed_frames: usize,
    start: Instant,

    record_submit_stuff: Mutex<RecordSubmitStuff>,
    transfer_queue: Mutex<StagedTransferQueue>,

    frame_context: Option<FrameContext>,
}

impl Context {
    pub fn new(window: Window) -> Self {
        let device = graphics::Device::new(&window).expect("failed to create device");
        let device = Arc::new(device);

        let swapchain = {
            let surface_info = &device.gpu.surface_info;

            let window_size = window.inner_size();
            let extent = vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            };

            let surface_format = surface_info.choose_surface_format();

            let image_count = surface_info.choose_image_count(FRAME_COUNT as u32);

            let config = graphics::SwapchainConfig {
                extent,
                present_mode: vk::PresentModeKHR::FIFO,
                surface_format,
                frame_count: FRAME_COUNT,
                image_count,
                fullscreen_mode: graphics::SwapchainFullScreenMode::None,
            };

            graphics::Swapchain::new(&device, config)
        };

        let frames = std::array::from_fn(|_frame_index| {
            let image_available_semaphore = device.create_semaphore("image_available_semaphore");
            let render_finished_semaphore = device.create_semaphore("render_finished_semaphore");

            let timestamp_query_pool = unsafe {
                let create_info = vk::QueryPoolCreateInfo::builder()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(MAX_TIMESTAMP_COUNT);

                device.raw.create_query_pool(&create_info, None).unwrap()
            };

            Frame {
                first_time_use: true,

                in_flight_fence_value: 0,
                image_available_semaphore,
                render_finished_semaphore,
                uses_async_transfer: false,

                command_pools: (0..rayon::current_num_threads())
                    .map(|i| {
                        Mutex::new(graphics::CommandPool::new(
                            &device,
                            &format!("command_pool_thread_{i}"),
                            QueueType::Graphics,
                        ))
                    })
                    .collect(),

                compiled_graph: CompiledRenderGraph::new(),
                graph_debug_info: GraphDebugInfo::new(),
                timestamp_query_pool,
            }
        });

        let record_submit_stuff = {
            let command_pool = graphics::CommandPool::new(&device, "global", QueueType::Graphics);
            let fence = device.create_fence("record_submit_fence", false);

            Mutex::new(RecordSubmitStuff { command_pool, fence })
        };

        let transfer_queue = Mutex::new(StagedTransferQueue::new(&device, STAGING_PAGE_COUNT, STAGING_PAGE_SIZE));

        Self {
            window,

            device,
            swapchain,

            shader_modules: HashMap::new(),
            raster_pipelines: HashMap::new(),
            compute_pipelines: HashMap::new(),

            graph: RenderGraph::new(),
            transient_resource_cache: TransientResourceCache::new(),

            frames,
            frame_index: 0,
            elapsed_frames: 0,
            start: Instant::now(),

            record_submit_stuff,
            transfer_queue,

            frame_context: None,
        }
    }

    pub fn gpu(&self) -> &graphics::GpuInfo {
        &self.device.gpu
    }

    pub fn record_and_submit(&self, f: impl FnOnce(&graphics::CommandRecorder)) {
        puffin::profile_function!();
        let mut record_submit_stuff = self.record_submit_stuff.lock();

        record_submit_stuff.command_pool.reset(&self.device);
        let recorder = record_submit_stuff
            .command_pool
            .begin_new(&self.device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        f(&recorder);

        drop(recorder);

        let value = self.device.queue_submit(QueueType::Graphics, record_submit_stuff.command_pool.submit_info());
        self.device.wait_queue_semaphore(QueueType::Graphics, value, None).unwrap();
    }

    #[track_caller]
    pub fn queue_write_buffer(&self, buffer: &graphics::BufferView, offset: usize, data: &[u8]) {
        let data_size = data.len();
        let buffer_size = buffer.size as usize;

        assert!(offset <= buffer_size, "offset = {offset}, buffer_size = {buffer_size}");
        assert!(
            data_size <= buffer_size - offset,
            "data_size = {data_size}, buffer_size = {buffer_size}, offset = {offset}"
        );

        let mut transfer_queue = self.transfer_queue.lock();

        if transfer_queue.pending_bytes() >= STAGING_PAGE_SIZE {
            transfer_queue.submit(&self.device);
        }

        transfer_queue.queue_write_buffer(buffer, offset, data);
    }

    pub fn queue_write_image(
        &self,
        image: &graphics::ImageRaw,
        mip_level: u32,
        layers: Range<u32>,
        region: Option<(vk::Offset3D, vk::Extent3D)>,
        prev_access: AccessKind,
        data: &[u8],
    ) {
        let mut transfer_queue = self.transfer_queue.lock();

        if transfer_queue.pending_bytes() >= STAGING_PAGE_SIZE {
            transfer_queue.submit(&self.device);
        }

        transfer_queue.queue_write_image(image, mip_level, layers, region, prev_access, data);
    }

    pub fn submit_pending(&mut self) {
        self.transfer_queue.get_mut().submit(&self.device);
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }

        unsafe {
            self.device.raw.device_wait_idle().unwrap();
        }

        let record_submit_stuff = self.record_submit_stuff.get_mut();
        unsafe {
            self.device.raw.destroy_fence(record_submit_stuff.fence, None);
        }
        record_submit_stuff.command_pool.destroy(&self.device);

        self.transfer_queue.get_mut().destroy(&self.device);

        for frame in self.frames.iter_mut() {
            unsafe {
                self.device.raw.destroy_semaphore(frame.image_available_semaphore.handle, None);
                self.device.raw.destroy_semaphore(frame.render_finished_semaphore.handle, None);
                self.device.raw.destroy_query_pool(frame.timestamp_query_pool, None);
            }

            for command_pool in frame.command_pools.iter() {
                command_pool.lock().destroy(&self.device);
            }

            for graph::CompiledGraphResource {
                resource,
                owned_by_graph,
            } in frame.compiled_graph.resources.drain(..)
            {
                if owned_by_graph || !resource.is_owned() {
                    graphics::AnyResource::destroy(&self.device, resource);
                }
            }
        }

        for resource in self.transient_resource_cache.drain_resources() {
            graphics::AnyResource::destroy(&self.device, resource);
        }

        for shader_module in self.shader_modules.values().copied() {
            unsafe {
                self.device.raw.destroy_shader_module(shader_module, None);
            }
        }

        for raster_pipeline in self.raster_pipelines.values().copied() {
            unsafe {
                self.device.raw.destroy_pipeline(raster_pipeline.handle, None);
            }
        }

        for compute_pipeline in self.compute_pipelines.values().copied() {
            unsafe {
                self.device.raw.destroy_pipeline(compute_pipeline.handle, None);
            }
        }

        self.swapchain.destroy(&self.device);
    }
}

impl Context {
    pub fn begin_frame(&mut self) {
        assert!(self.frame_context.is_none(), "frame already began");
        puffin::profile_function!();
        let frame = &mut self.frames[self.frame_index];

        unsafe {
            puffin::profile_scope!("fence_wait");
            self.device.wait_queue_semaphore(QueueType::Graphics, frame.in_flight_fence_value, None).unwrap();

            if !frame.first_time_use {
                let timestamp_count = frame.graph_debug_info.timestamp_count;
                self.device
                    .raw
                    .get_query_pool_results::<u64>(
                        frame.timestamp_query_pool,
                        0,
                        timestamp_count,
                        &mut frame.graph_debug_info.timestamp_data,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .unwrap();
            } else {
                frame.first_time_use = false;
            }
        }

        self.swapchain.resize(self.window.inner_size().into());

        let acquired_image = self
            .swapchain
            .acquire_image(
                &mut self.device,
                self.frame_index,
                frame.image_available_semaphore.handle,
            )
            .unwrap();

        self.graph.clear();

        assert!(self.transient_resource_cache.is_empty());
        for graph::CompiledGraphResource {
            resource,
            owned_by_graph,
        } in frame.compiled_graph.resources.drain(..)
        {
            if !owned_by_graph && resource.is_owned() {
                // swapchain image
                continue;
            }
            self.transient_resource_cache.insert(resource);
        }

        let acquired_image_handle = self.graph.add_resource(graph::GraphResourceData {
            name: "swapchain_image".into(),

            source: graph::ResourceSource::Import {
                resource: acquired_image.image.clone().into(),
            },
            descriptor_index: None,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::Present,
            initial_queue: Some(QueueType::Graphics),
            target_queue: None,

            wait_semaphore: Some(frame.image_available_semaphore.clone()),
            finish_semaphore: Some(frame.render_finished_semaphore.clone()),
            versions: vec![],
        });

        self.frame_context = Some(FrameContext {
            acquired_image,
            acquired_image_handle: GraphHandle {
                resource_index: acquired_image_handle,
                _phantom: PhantomData,
            },
        });
    }
}

#[derive(Debug)]
pub struct GraphHandle<T: ?Sized> {
    pub resource_index: graphics::GraphResourceIndex,
    pub _phantom: PhantomData<T>,
}

impl<T> GraphHandle<T> {
    pub fn uninit() -> Self {
        Self {
            resource_index: 0,
            _phantom: PhantomData,
        }
    }
}

impl<T> Clone for GraphHandle<T> {
    fn clone(&self) -> Self {
        Self {
            resource_index: self.resource_index.clone(),
            _phantom: self._phantom.clone(),
        }
    }
}

impl<T> Copy for GraphHandle<T> {}

impl<T> PartialEq for GraphHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.resource_index == other.resource_index
    }
}

impl<T> Eq for GraphHandle<T> {}

pub type GraphBufferHandle = GraphHandle<graphics::BufferRaw>;
pub type GraphImageHandle = GraphHandle<graphics::ImageRaw>;

impl From<GraphHandle<graphics::ImageRaw>> for egui::TextureId {
    fn from(value: GraphHandle<graphics::ImageRaw>) -> Self {
        Self::User(value.resource_index as u64)
    }
}

pub struct PassBuilder<'a> {
    context: &'a mut Context,
    name: Cow<'static, str>,
    dependencies: Vec<(graphics::GraphResourceIndex, graphics::AccessKind)>,
}

pub trait IntoGraphDependency {
    fn into_dependency(self) -> (graphics::GraphResourceIndex, graphics::AccessKind);
}

impl<T> IntoGraphDependency for (graphics::GraphHandle<T>, graphics::AccessKind) {
    fn into_dependency(self) -> (graphics::GraphResourceIndex, graphics::AccessKind) {
        (self.0.resource_index, self.1)
    }
}

impl<'a> PassBuilder<'a> {
    pub fn with_dependency<T>(mut self, handle: graphics::GraphHandle<T>, access: graphics::AccessKind) -> Self {
        self.dependencies.push((handle.resource_index, access));
        self
    }

    pub fn with_dependencies<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoGraphDependency,
    {
        self.dependencies.extend(iter.into_iter().map(|item| item.into_dependency()));
        self
    }

    pub fn record_custom(
        self,
        f: impl Fn(&graphics::CommandRecorder, &graphics::CompiledRenderGraph) + Send + Sync + 'static,
    ) -> graphics::GraphPassIndex {
        let pass = self.context.graph.add_pass(self.name, graphics::Pass::CustomPass(Box::new(f)), &self.dependencies);
        pass
    }
}

pub struct CustomPass<'a> {
    context: &'a mut Context,
    name: Cow<'static, str>,
    dependencies: Vec<(graphics::GraphResourceIndex, graphics::AccessKind)>,
}

impl CustomPass<'_> {
    pub fn with_dependency<T>(mut self, handle: graphics::GraphHandle<T>, access: graphics::AccessKind) -> Self {
        self.dependencies.push((handle.resource_index, access));
        self
    }

    pub fn with_dependencies<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoGraphDependency,
    {
        self.dependencies.extend(iter.into_iter().map(|item| item.into_dependency()));
        self
    }

    pub fn record(
        self,
        f: impl Fn(&graphics::CommandRecorder, &graphics::CompiledRenderGraph) + Send + Sync + 'static,
    ) -> graphics::GraphPassIndex {
        let pass = self.context.graph.add_pass(self.name, graphics::Pass::CustomPass(Box::new(f)), &self.dependencies);
        pass
    }
}

pub struct ComputePass<'a> {
    context: &'a mut Context,
    name: Cow<'static, str>,
    pipeline: graphics::ComputePipeline,
    constant_data: utils::StructuredDataBuilder<128>,
    dependencies: Vec<(graphics::GraphResourceIndex, graphics::AccessKind)>,
}

impl<'a> ComputePass<'a> {
    pub fn new(
        context: &'a mut graphics::Context,
        name: impl Into<Cow<'static, str>>,
        pipeline: graphics::ComputePipeline,
    ) -> Self {
        Self {
            context,
            name: name.into(),
            pipeline,
            constant_data: utils::StructuredDataBuilder::new(),
            dependencies: Vec::new(),
        }
    }

    #[must_use = "missing dispatch call"]
    pub fn with_dependency<T>(mut self, handle: graphics::GraphHandle<T>, access: graphics::AccessKind) -> Self {
        self.dependencies.push((handle.resource_index, access));
        self
    }

    #[must_use = "missing dispatch call"]
    pub fn with_dependencies<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoGraphDependency,
    {
        self.dependencies.extend(iter.into_iter().map(|item| item.into_dependency()));
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn push_bytes(mut self, bytes: &[u8]) -> Self {
        self.constant_data.push_bytes_with_align(bytes, 4);
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn push_bytes_with_align(mut self, bytes: &[u8], align: usize) -> Self {
        self.constant_data.push_bytes_with_align(bytes, align);
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn push_data<T: bytemuck::NoUninit>(mut self, val: T) -> Self {
        self.constant_data
            .push_bytes_with_align(bytemuck::bytes_of(&val), std::mem::align_of_val(&val).min(4));
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn read_buffer(mut self, buffer: GraphBufferHandle) -> Self {
        self.dependencies.push((buffer.resource_index, graphics::AccessKind::ComputeShaderRead));
        let desc_index = self.context.get_resource_descriptor_index(buffer).unwrap();
        self.push_data(desc_index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn write_buffer(mut self, buffer: GraphBufferHandle) -> Self {
        self.dependencies.push((buffer.resource_index, graphics::AccessKind::ComputeShaderWrite));
        let desc_index = self.context.get_resource_descriptor_index(buffer).unwrap();
        self.push_data(desc_index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn read_image(mut self, image: GraphImageHandle) -> Self {
        self.dependencies.push((image.resource_index, graphics::AccessKind::ComputeShaderRead));
        let desc_index = self.context.get_resource_descriptor_index(image).unwrap();
        self.push_data(desc_index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn read_image_general(mut self, image: GraphImageHandle) -> Self {
        self.dependencies.push((image.resource_index, graphics::AccessKind::ComputeShaderReadGeneral));
        let desc_index = self.context.get_resource_descriptor_index(image).unwrap();
        self.push_data(desc_index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing dispatch call"]
    pub fn write_image(mut self, image: GraphImageHandle) -> Self {
        self.dependencies.push((image.resource_index, graphics::AccessKind::ComputeShaderWrite));
        let desc_index = self.context.get_resource_descriptor_index(image).unwrap();
        self.push_data(desc_index)
    }

    pub fn dispatch(self, workgroup_count: [u32; 3]) -> graphics::GraphPassIndex {
        self.context.graph.add_pass(
            self.name,
            graphics::Pass::ComputeDispatch {
                pipeline: self.pipeline,
                push_constant_data: self.constant_data.constants,
                push_constant_size: self.constant_data.byte_cursor,
                dispatch: Dispatch::DispatchWorkgroups(workgroup_count),
            },
            &self.dependencies,
        )
    }

    pub fn dispatch_indirect(
        mut self,
        indirect_buffer: graphics::GraphBufferHandle,
        offset: u64,
    ) -> graphics::GraphPassIndex {
        self.dependencies.push((indirect_buffer.resource_index, graphics::AccessKind::IndirectBuffer));

        self.context.graph.add_pass(
            self.name,
            graphics::Pass::ComputeDispatch {
                pipeline: self.pipeline,
                push_constant_data: self.constant_data.constants,
                push_constant_size: self.constant_data.byte_cursor,
                dispatch: Dispatch::DispatchIndirect(indirect_buffer, offset),
            },
            &self.dependencies,
        )
    }
}

pub struct RenderPass<'a> {
    context: &'a mut graphics::Context,
    name: Cow<'static, str>,
    color_attachments: [graphics::ColorAttachmentDesc; graphics::MAX_COLOR_ATTACHMENT_COUNT],
    color_attachment_count: usize,
    depth_attachment: Option<graphics::DepthAttachmentDesc>,
    draw_range: Range<usize>,
    dependencies: Vec<(graphics::GraphResourceIndex, graphics::AccessKind)>,
}

impl<'a> RenderPass<'a> {
    pub fn new(context: &'a mut graphics::Context, name: impl Into<Cow<'static, str>>) -> Self {
        let dummy_color_attachment = graphics::ColorAttachmentDesc {
            target: graphics::GraphImageHandle::uninit(),
            resolve: None,
            load_op: graphics::LoadOp::Load,
            store: false,
        };

        let draw_range = context.graph.draws.len()..context.graph.draws.len();

        Self {
            context,
            name: name.into(),
            color_attachments: [dummy_color_attachment; graphics::MAX_COLOR_ATTACHMENT_COUNT],
            color_attachment_count: 0,
            depth_attachment: None,
            draw_range,
            dependencies: Vec::new(),
        }
    }

    #[track_caller]
    pub fn color_attachments(mut self, color_attachments: &[graphics::ColorAttachmentDesc]) -> Self {
        assert!(color_attachments.len() <= graphics::MAX_COLOR_ATTACHMENT_COUNT);
        self.color_attachments[..color_attachments.len()].copy_from_slice(color_attachments);
        self.color_attachment_count = color_attachments.len();
        self
    }

    pub fn depth_attachment(mut self, depth_attachment: graphics::DepthAttachmentDesc) -> Self {
        self.depth_attachment = Some(depth_attachment);
        self
    }

    pub fn finish(mut self) -> graphics::GraphPassIndex {
        assert!(
            self.depth_attachment.is_some() || self.color_attachment_count > 0,
            "all render passes should have at least one attachment"
        );

        let get_render_area = |handle| {
            let dimensions = self.context.get_image_desc(handle).dimensions;
            [dimensions[0], dimensions[1]]
        };

        let render_area = if let Some(depth_attachment) = self.depth_attachment.as_ref() {
            get_render_area(depth_attachment.target)
        } else {
            get_render_area(self.color_attachments[0].target)
        };

        for attachment in &self.color_attachments[..self.color_attachment_count] {
            assert_eq!(
                render_area,
                get_render_area(attachment.target),
                "all attachments must have the same dimensions"
            );

            let access = if attachment.store || matches!(attachment.load_op, graphics::LoadOp::Clear(_)) {
                graphics::AccessKind::ColorAttachmentWrite
            } else {
                graphics::AccessKind::ColorAttachmentRead
            };
            self.dependencies.push((attachment.target.resource_index, access));
            if let Some((resolve, _)) = attachment.resolve {
                self.dependencies.push((resolve.resource_index, graphics::AccessKind::ColorAttachmentWrite));
                assert_eq!(
                    render_area,
                    get_render_area(resolve),
                    "all attachments must have the same dimensions"
                );
            }
        }

        if let Some(attachment) = self.depth_attachment {
            assert_eq!(
                render_area,
                get_render_area(attachment.target),
                "all attachments must have the same dimensions"
            );

            let access = if attachment.store || matches!(attachment.load_op, graphics::LoadOp::Clear(_)) {
                graphics::AccessKind::DepthAttachmentWrite
            } else {
                graphics::AccessKind::DepthAttachmentRead
            };
            self.dependencies.push((attachment.target.resource_index, access));
            if let Some((resolve, _)) = attachment.resolve {
                self.dependencies.push((resolve.resource_index, graphics::AccessKind::DepthAttachmentWrite));
                assert_eq!(
                    render_area,
                    get_render_area(resolve),
                    "all attachments must have the same dimensions"
                );
            }
        }

        self.draw_range.end = self.context.graph.draws.len();

        let pass = graphics::Pass::RenderPass {
            color_attachments: self.color_attachments,
            color_attachment_count: self.color_attachment_count,
            depth_attachment: self.depth_attachment,
            render_area,
            draw_range: self.draw_range,
        };

        self.context.graph.add_pass(self.name, pass, &self.dependencies)
    }
}

pub struct DrawPass<'a, 'b> {
    render_pass: &'b mut graphics::RenderPass<'a>,
    pipeline: graphics::RasterPipeline,
    depth_test_enable: Option<bool>,
    depth_bias: Option<[f32; 3]>,
    index_buffer: Option<(graphics::GraphBufferHandle, u64, vk::IndexType)>,
    constant_data: utils::StructuredDataBuilder<128>,
}

impl<'a: 'b, 'b> DrawPass<'a, 'b> {
    pub fn new(render_pass: &'b mut graphics::RenderPass<'a>, pipeline: graphics::RasterPipeline) -> Self {
        Self {
            render_pass,
            pipeline,
            depth_test_enable: None,
            depth_bias: None,
            index_buffer: None,
            constant_data: utils::StructuredDataBuilder::new(),
        }
    }

    #[must_use = "missing draw call"]
    pub fn with_depth_test(mut self, depth_test: bool) -> Self {
        self.depth_test_enable = Some(depth_test);
        self
    }

    #[must_use = "missing draw call"]
    pub fn with_bias(mut self, constant_factor: f32, clamp: f32, slope_factor: f32) -> Self {
        self.depth_bias = Some([constant_factor, clamp, slope_factor]);
        self
    }

    #[must_use = "missing draw call"]
    pub fn with_index_buffer(mut self, buffer: GraphBufferHandle, offset: u64, index_type: vk::IndexType) -> Self {
        self.index_buffer = Some((buffer, offset, index_type));
        self
    }

    #[must_use = "missing draw call"]
    pub fn with_dependency<T>(self, handle: graphics::GraphHandle<T>, access: graphics::AccessKind) -> Self {
        self.render_pass.dependencies.push((handle.resource_index, access));
        self
    }

    #[must_use = "missing draw call"]
    pub fn with_dependencies<I>(self, iter: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoGraphDependency,
    {
        self.render_pass.dependencies.extend(iter.into_iter().map(|item| item.into_dependency()));
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn push_bytes_with_align(mut self, bytes: &[u8], align: usize) -> Self {
        self.constant_data.push_bytes_with_align(bytes, align);
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn push_data<T: bytemuck::NoUninit>(mut self, val: T) -> Self {
        self.constant_data
            .push_bytes_with_align(bytemuck::bytes_of(&val), std::mem::align_of_val(&val).min(4));
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn push_data_ref<T: bytemuck::NoUninit>(mut self, val: &T) -> Self {
        self.constant_data
            .push_bytes_with_align(bytemuck::bytes_of(val), std::mem::align_of_val(val).min(4));
        self
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn read_buffer(self, buffer: GraphBufferHandle) -> Self {
        self.render_pass.dependencies.push((buffer.resource_index, graphics::AccessKind::AllGraphicsRead));
        let desc_index = self.render_pass.context.get_resource_descriptor_index(buffer).unwrap();
        self.push_data(desc_index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn write_buffer(self, buffer: GraphBufferHandle) -> Self {
        self.render_pass.dependencies.push((buffer.resource_index, graphics::AccessKind::AllGraphicsWrite));
        let desc_index = self.render_pass.context.get_resource_descriptor_index(buffer).unwrap();
        self.push_data(desc_index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn read_image(self, image: impl Into<Option<GraphImageHandle>>) -> Self {
        let index = if let Some(image) = image.into() {
            self.render_pass.dependencies.push((image.resource_index, graphics::AccessKind::AllGraphicsRead));
            self.render_pass.context.get_resource_descriptor_index(image).unwrap()
        } else {
            u32::MAX
        };

        self.push_data(index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn read_image_general(self, image: impl Into<Option<GraphImageHandle>>) -> Self {
        let index = if let Some(image) = image.into() {
            self.render_pass
                .dependencies
                .push((image.resource_index, graphics::AccessKind::AllGraphicsReadGeneral));
            self.render_pass.context.get_resource_descriptor_index(image).unwrap()
        } else {
            u32::MAX
        };

        self.push_data(index)
    }

    #[track_caller]
    #[inline(always)]
    #[must_use = "missing draw call"]
    pub fn write_image(self, image: impl Into<Option<GraphImageHandle>>) -> Self {
        let index = if let Some(image) = image.into() {
            self.render_pass.dependencies.push((image.resource_index, graphics::AccessKind::AllGraphicsWrite));
            self.render_pass.context.get_resource_descriptor_index(image).unwrap()
        } else {
            u32::MAX
        };

        self.push_data(index)
    }

    pub fn draw_command(self, draw_command: graphics::DrawCommand) {
        if let Some((index_buffer, _, _)) = self.index_buffer {
            self.render_pass.dependencies.push((index_buffer.resource_index, graphics::AccessKind::IndexBuffer));
        }

        if let Some(draw_buffer) = draw_command.draw_buffer() {
            self.render_pass
                .dependencies
                .push((draw_buffer.resource_index, graphics::AccessKind::IndirectBuffer));
        }

        if let Some(count_buffer) = draw_command.count_buffer() {
            self.render_pass
                .dependencies
                .push((count_buffer.resource_index, graphics::AccessKind::IndirectBuffer));
        }

        self.render_pass.context.graph.draws.push(graphics::RenderPassDraw {
            pipeline: self.pipeline,
            depth_test_enable: self.depth_test_enable,
            depth_bias: self.depth_bias,
            index_buffer: self.index_buffer,
            constant_data: self.constant_data.constants,
            constant_size: self.constant_data.byte_cursor,
            draw_command,
        });
    }

    // see draw_gen.rs
    pub fn draw_meshlets(self, draw_buffer: graphics::GraphBufferHandle, mesh_shading: bool) {
        let command = if mesh_shading {
            graphics::DrawCommand::DrawMeshTasksIndirect {
                task_buffer: draw_buffer,
                task_buffer_offset: 0,
                draw_count: 1,
                stride: 0,
            }
        } else {
            graphics::DrawCommand::DrawIndexedIndirectCount {
                draw_buffer: draw_buffer,
                draw_buffer_offset: 4,
                count_buffer: draw_buffer,
                count_buffer_offset: 0,
                max_draw_count: MAX_DRAW_COUNT as u32,
                stride: std::mem::size_of::<GpuMeshletDrawCommand>() as u32,
            }
        };
        self.draw_command(command);
    }

    pub fn draw(self, vertex_range: Range<u32>, instance_range: Range<u32>) {
        self.draw_command(graphics::DrawCommand::Draw {
            vertex_range,
            instance_range,
        });
    }
}

impl Context {
    pub fn frame_index(&self) -> usize {
        self.frame_index
    }

    pub fn swapchain_extent(&self) -> vk::Extent3D {
        self.frame_context.as_ref().unwrap().acquired_image.extent
    }

    pub fn swapchain_format(&self) -> vk::Format {
        self.frame_context.as_ref().unwrap().acquired_image.format
    }

    pub fn get_swapchain_image(&self) -> graphics::GraphImageHandle {
        self.frame_context.as_ref().unwrap().acquired_image_handle
    }

    pub fn get_resource_descriptor_index<T>(&self, handle: GraphHandle<T>) -> Option<graphics::DescriptorIndex> {
        self.graph.resources[handle.resource_index].descriptor_index
    }

    #[track_caller]
    pub fn get_buffer_desc(&self, handle: GraphBufferHandle) -> graphics::BufferDesc {
        match self.graph.resources[handle.resource_index].source.desc() {
            graphics::AnyResourceDesc::Buffer(desc) => desc,
            _ => panic!("resource isn't a buffer"),
        }
    }

    #[track_caller]
    pub fn get_image_desc(&self, handle: GraphImageHandle) -> graphics::ImageDesc {
        match self.graph.resources[handle.resource_index].source.desc() {
            graphics::AnyResourceDesc::Image(desc) => desc,
            _ => panic!("resource isn't an image"),
        }
    }

    pub fn import<R: graphics::RenderResource>(&mut self, resource: R) -> GraphHandle<R::RawResource> {
        let resource = resource.into();
        let descriptor_index = resource.as_ref().descriptor_index();
        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name: resource.as_ref().name().clone(),
            source: graph::ResourceSource::Import { resource },
            descriptor_index,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            initial_queue: Some(QueueType::Graphics),
            target_queue: Some(QueueType::Graphics),
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
    }

    pub fn import_with<R: graphics::RenderResource>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        resource: R,
        desc: graphics::GraphResourceImportDesc,
    ) -> GraphHandle<R::RawResource> {
        let name = name.into();
        let resource = resource.into();
        let descriptor_index = resource.as_ref().descriptor_index();
        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,
            source: graph::ResourceSource::Import { resource },
            descriptor_index,

            initial_access: desc.initial_access,
            target_access: desc.target_access,
            initial_queue: desc.initial_queue,
            target_queue: desc.target_queue,
            wait_semaphore: desc.wait_semaphore,
            finish_semaphore: desc.finish_semaphore,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
    }

    pub fn transient_storage_data(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        data: &[u8],
    ) -> GraphHandle<graphics::BufferRaw> {
        let name: Cow<'static, str> = name.into();
        let buffer_desc = graphics::BufferDesc {
            size: data.len(),
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: gpu_allocator::MemoryLocation::GpuOnly,
            ..Default::default()
        };
        let desc = graphics::AnyResourceDesc::Buffer(buffer_desc);
        let (mut cache, cache_needs_rename) = self.transient_resource_cache.get(&name, &desc).unwrap_or_else(|| {
            (
                graphics::AnyResource::create_owned(&self.device, name.clone(), &desc, None),
                false,
            )
        });

        if cache_needs_rename {
            cache.rename(&self.device, name.clone());
        }

        let graphics::AnyResourceRef::Buffer(buffer) = cache.as_ref() else {
            unreachable!()
        };

        self.transfer_queue.get_mut().queue_write_buffer(&buffer, 0, data);
        self.frames[self.frame_index].uses_async_transfer = true;

        let descriptor_index = buffer.descriptor_index;

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,

            source: graph::ResourceSource::Create {
                desc,
                cache: Some(cache),
            },
            descriptor_index,

            initial_queue: None,
            target_queue: None,
            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            // wait_semaphore: Some(self.frames[self.frame_index].transfer_finished_semaphore.clone()),
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
    }

    pub fn create_transient<R: graphics::OwnedRenderResource>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: R::Desc,
    ) -> GraphHandle<R> {
        let name: Cow<'static, str> = name.into();
        let desc = desc.into();
        let cache = if let Some((mut cache, needs_rename)) = self.transient_resource_cache.get(&name, &desc) {
            if needs_rename {
                cache.rename(&self.device, name.clone());
            }

            Some(cache)
        } else {
            None
        };
        let descriptor_index = match &cache {
            Some(cache) => cache.as_ref().descriptor_index(),
            None if desc.needs_descriptor_index() => Some(self.device.alloc_descriptor_index()),
            _ => None,
        };

        let resource_index = self.graph.add_resource(graph::GraphResourceData {
            name,

            source: graph::ResourceSource::Create { desc, cache },
            descriptor_index,

            initial_access: graphics::AccessKind::None,
            target_access: graphics::AccessKind::None,
            initial_queue: None,
            target_queue: None,
            wait_semaphore: None,
            finish_semaphore: None,
            versions: vec![],
        });

        GraphHandle {
            resource_index,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn create_transient_buffer(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: graphics::BufferDesc,
    ) -> GraphBufferHandle {
        self.create_transient(name, desc)
    }

    #[inline(always)]
    pub fn create_transient_image(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        desc: graphics::ImageDesc,
    ) -> GraphImageHandle {
        self.create_transient(name, desc)
    }

    pub fn add_pass(&mut self, name: impl Into<Cow<'static, str>>) -> PassBuilder {
        PassBuilder {
            context: self,
            name: name.into(),
            dependencies: Vec::new(),
        }
    }

    pub fn end_frame(&mut self) {
        puffin::profile_function!();
        let frame_context = self.frame_context.take().unwrap();

        let frame = &mut self.frames[self.frame_index];

        frame.uses_async_transfer = false;

        self.transfer_queue.get_mut().submit(&self.device);
        let last_transfer_fence = self.transfer_queue.get_mut().last_fence_value;

        let image_transfers: Vec<ImagePostTransferOp> =
            self.transfer_queue.get_mut().pending_image_transfers.drain(..).collect();

        frame.graph_debug_info.clear();

        self.graph.compile_and_flush(&self.device, &mut frame.compiled_graph);

        frame
            .graph_debug_info
            .batch_infos
            .resize(frame.compiled_graph.batches.len(), GraphBatchDebugInfo::default());
        frame.graph_debug_info.timestamp_count = frame.compiled_graph.batches.len() as u32 * 2;

        unsafe {
            self.device.raw.reset_query_pool(frame.timestamp_query_pool, 0, MAX_TIMESTAMP_COUNT);
        }

        for command_pool in frame.command_pools.iter_mut() {
            command_pool.get_mut().mark_for_reset();
        }

        let batch_cmd_indices: Vec<(usize, usize)> = {
            puffin::profile_scope!("command_recording");
            frame
                .compiled_graph
                .batches
                .par_iter()
                .zip(frame.graph_debug_info.batch_infos.par_iter_mut())
                .enumerate()
                .map_init(
                    || frame.command_pools[rayon::current_thread_index().unwrap()].lock(),
                    |command_pool, (batch_index, (batch_data, batch_debug_info))| {
                        let batch_ref = frame.compiled_graph.get_batch_ref(&batch_data);

                        let transfer_semaphore = (batch_index == 0).then_some(last_transfer_fence);

                        let command_index = record_batch(
                            &self.device,
                            &frame.compiled_graph,
                            command_pool,
                            frame.timestamp_query_pool,
                            batch_index,
                            batch_ref,
                            batch_debug_info,
                            transfer_semaphore,
                            if batch_index == 0 {
                                image_transfers.as_slice()
                            } else {
                                &[]
                            },
                        );

                        (rayon::current_thread_index().unwrap(), command_index)
                    },
                )
                .collect()
        };

        {
            puffin::profile_scope!("command_submit");
            let mut command_batches = Vec::new();
            let mut semaphore_infos = Vec::new();
            for (pool_index, batch_index) in batch_cmd_indices.iter().copied() {
                let (command_buffer, wait_semaphores, signal_semaphores) =
                    frame.command_pools[pool_index].get_mut().get_batch(batch_index);

                let wait_start = semaphore_infos.len();
                semaphore_infos.extend(wait_semaphores);
                let wait_end = semaphore_infos.len();

                let signal_start = semaphore_infos.len();
                semaphore_infos.extend(signal_semaphores);
                let signal_end = semaphore_infos.len();

                command_batches.push(graphics::CommandBatch {
                    command_buffer,
                    wait_semaphore_range: wait_start..wait_end,
                    signal_semaphore_range: signal_start..signal_end,
                });
            }

            frame.in_flight_fence_value = self.device.queue_submit(
                QueueType::Graphics,
                graphics::SubmitInfo {
                    batches: &command_batches,
                    semaphores: &semaphore_infos,
                },
            );
        }
        {
            puffin::profile_scope!("queue_present");
            self.swapchain.queue_present(
                &self.device,
                frame_context.acquired_image,
                frame.render_finished_semaphore.handle,
            );
        }

        {
            puffin::profile_scope!("leftover_resource_releases");
            for resource in self.transient_resource_cache.drain_resources() {
                graphics::AnyResource::destroy(&self.device, resource);
            }
            self.transient_resource_cache.clear();
        }

        self.frame_index = (self.frame_index + 1) % FRAME_COUNT;
        self.elapsed_frames += 1;
    }
}

fn record_batch(
    device: &graphics::Device,
    compiled_graph: &graph::CompiledRenderGraph,
    command_pool: &mut graphics::CommandPool,
    timestamp_query_pool: vk::QueryPool,
    batch_index: usize,
    batch_ref: graph::BatchRef,
    batch_debug_info: &mut GraphBatchDebugInfo,
    transfer_fence: Option<u64>,
    image_trasnfers: &[ImagePostTransferOp],
) -> usize {
    puffin::profile_scope!("batch_record", format!("{batch_index}"));

    let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = batch_ref
        .begin_image_barriers
        .iter()
        .map(|b| {
            graphics::image_barrier(
                compiled_graph.resources[b.image_index].resource.as_ref().as_image().unwrap(),
                b.src_access,
                b.dst_access,
            )
        })
        .collect();

    let mut recorder = command_pool.begin_new(device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    if let Some(fence_value) = transfer_fence {
        let semaphore = device.get_queue(QueueType::AsyncTransfer).semaphore;
        recorder.wait_semaphore(semaphore, vk::PipelineStageFlags2::ALL_COMMANDS, Some(fence_value));
    }

    for (semaphore, stage) in batch_ref.wait_semaphores {
        recorder.wait_semaphore(semaphore.handle, *stage, None);
    }

    for (semaphore, stage) in batch_ref.signal_semaphores {
        recorder.signal_semaphore(semaphore.handle, *stage, None);
    }

    image_barriers.reserve(image_trasnfers.len());
    for image_transfer in image_trasnfers {
        let Some((src_queue_family_index, dst_queue_family_index)) = image_transfer.queue_transfer else {
            continue;
        };
        let barrier = vk::ImageMemoryBarrier2 {
            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            dst_access_mask: vk::AccessFlags2::MEMORY_READ,
            old_layout: image_transfer.access.image_layout(),
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index,
            dst_queue_family_index,
            image: image_transfer.image,
            subresource_range: image_transfer.subresource.to_subresource_range(),
            ..Default::default()
        };
        image_barriers.push(barrier);
    }

    let memory_barrier = if graphics::is_memory_barrier_not_useless(&batch_ref.memory_barrier) {
        std::slice::from_ref(&batch_ref.memory_barrier)
    } else {
        &[]
    };
    recorder.barrier(&[], &image_barriers, memory_barrier);

    batch_debug_info.timestamp_start_index = batch_index as u32 * 2;
    batch_debug_info.timestamp_end_index = batch_index as u32 * 2 + 1;

    recorder.write_query(
        vk::PipelineStageFlags2::NONE,
        timestamp_query_pool,
        batch_debug_info.timestamp_start_index,
    );

    for pass in batch_ref.passes {
        recorder.begin_debug_label(&pass.name, None);
        pass.func.record(&recorder, compiled_graph);
        recorder.end_debug_label();
    }

    recorder.write_query(
        vk::PipelineStageFlags2::ALL_COMMANDS,
        timestamp_query_pool,
        batch_debug_info.timestamp_end_index,
    );

    image_barriers.clear();
    image_barriers.extend(batch_ref.finish_image_barriers.iter().map(|b| {
        graphics::image_barrier(
            compiled_graph.resources[b.image_index].resource.as_ref().as_image().unwrap(),
            b.src_access,
            b.dst_access,
        )
    }));

    if !image_barriers.is_empty() {
        recorder.barrier(&[], &image_barriers, &[]);
    }

    recorder.batch_index()
}

impl Context {
    pub fn graph_debugger(&self, egui_ctx: &egui::Context) -> bool {
        let mut open = true;
        egui::Window::new("rendergraph debugger")
            .open(&mut open)
            .show(egui_ctx, |ui| self.draw_graph_info(ui));
        open
    }

    fn draw_graph_info(&self, ui: &mut egui::Ui) {
        let timestamp_period = self.device.gpu.properties.properties10.limits.timestamp_period;
        let graph = &self.frames[self.frame_index].compiled_graph;
        let debug_info = &self.frames[self.frame_index].graph_debug_info;
        for (i, batch) in graph.iter_batches().enumerate() {
            let delta_ns = debug_info.timestamp_delta(i, timestamp_period);
            let delta_ms = delta_ns / 1_000_000.0;
            egui::CollapsingHeader::new(format!("batch {i} ({delta_ms:.2} ms)")).id_source(i).show(ui, |ui| {
                egui::CollapsingHeader::new(format!("wait_semaphores ({})", batch.wait_semaphores.len()))
                    .id_source([i, 0])
                    .show(ui, |ui| {
                        for (semaphore, stage) in batch.wait_semaphores {
                            ui.label(format!("{semaphore:?}, {stage:?}"));
                        }
                    });

                egui::CollapsingHeader::new("memory_barrier").id_source([i, 1]).show(ui, |ui| {
                    ui.label(format!("{:#?}", batch.memory_barrier));
                });

                egui::CollapsingHeader::new(format!("begin_dependencies ({})", batch.begin_dependencies.len()))
                    .id_source([i, 2])
                    .show(ui, |ui| {
                        for (j, dependency) in batch.begin_dependencies.iter().enumerate() {
                            let resource = &graph.resources[dependency.resource_index].resource.as_ref();
                            egui::CollapsingHeader::new(resource.name().as_ref()).id_source([i, 2, j]).show(ui, |ui| {
                                ui.label(format!("dst_access: {:?}", dependency.dst_access));
                            });
                        }
                    });

                egui::CollapsingHeader::new(format!("begin_image_barriers ({})", batch.begin_image_barriers.len()))
                    .id_source([i, 3])
                    .show(ui, |ui| {
                        for (j, barrier) in batch.begin_image_barriers.iter().enumerate() {
                            let resource = &graph.resources[barrier.image_index].resource.as_ref();
                            egui::CollapsingHeader::new(resource.name().as_ref()).id_source([i, 3, j]).show(ui, |ui| {
                                ui.label(format!("src: {:#?}", barrier.src_access));
                                ui.label(format!("dst: {:#?}", barrier.dst_access));
                            });
                        }
                    });

                egui::CollapsingHeader::new(format!("passes ({})", batch.passes.len())).id_source([i, 4]).show(
                    ui,
                    |ui| {
                        for pass in batch.passes {
                            ui.label(pass.name.as_ref());
                        }
                    },
                );

                egui::CollapsingHeader::new(format!("finish_dependencies ({})", batch.finish_dependencies.len()))
                    .id_source([i, 5])
                    .show(ui, |ui| {
                        for (j, dependency) in batch.finish_dependencies.iter().enumerate() {
                            let resource = &graph.resources[dependency.resource_index].resource.as_ref();
                            egui::CollapsingHeader::new(resource.name().as_ref()).id_source([i, 5, j]).show(ui, |ui| {
                                ui.label(format!("dst_access: {:?}", dependency.dst_access));
                            });
                        }
                    });

                egui::CollapsingHeader::new(format!("finish_image_barriers ({})", batch.finish_image_barriers.len()))
                    .id_source([i, 6])
                    .show(ui, |ui| {
                        for (j, barrier) in batch.finish_image_barriers.iter().enumerate() {
                            let resource = &graph.resources[barrier.image_index].resource.as_ref();
                            egui::CollapsingHeader::new(resource.name().as_ref()).id_source([i, 6, j]).show(ui, |ui| {
                                ui.label(format!("src: {:#?}", barrier.src_access));
                                ui.label(format!("dst: {:#?}", barrier.dst_access));
                            });
                        }
                    });

                egui::CollapsingHeader::new(format!("signal_semaphores ({})", batch.signal_semaphores.len()))
                    .id_source([i, 7])
                    .show(ui, |ui| {
                        for (semaphore, stage) in batch.signal_semaphores {
                            ui.label(format!("{semaphore:?}, {stage:?}"));
                        }
                    });
            });
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphBatchDebugInfo {
    pub timestamp_start_index: u32,
    pub timestamp_end_index: u32,
}

pub struct GraphDebugInfo {
    pub batch_infos: Vec<GraphBatchDebugInfo>,
    pub timestamp_data: [u64; MAX_TIMESTAMP_COUNT as usize],
    pub timestamp_count: u32,
}

impl GraphDebugInfo {
    pub fn new() -> Self {
        Self {
            batch_infos: Vec::new(),
            timestamp_data: [0; MAX_TIMESTAMP_COUNT as usize],
            timestamp_count: 0,
        }
    }

    pub fn clear(&mut self) {
        self.batch_infos.clear();
    }

    pub fn timestamp_delta(&self, batch_index: usize, timestamp_period: f32) -> f32 {
        let GraphBatchDebugInfo {
            timestamp_start_index,
            timestamp_end_index,
            ..
        } = self.batch_infos[batch_index];
        let delta =
            self.timestamp_data[timestamp_end_index as usize] - self.timestamp_data[timestamp_start_index as usize];
        delta as f32 * timestamp_period
    }
}
