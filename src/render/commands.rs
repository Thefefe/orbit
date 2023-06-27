use ash::vk;
use std::ffi::CString;
use std::marker::PhantomData;
use std::ops::Range;

use crate::render;
use crate::utils::Unsync;

pub struct CommandPool {
    handle: vk::CommandPool,
    name: String,
    command_buffers: Vec<CommandBuffer>,
    used_buffers: usize,

    _phantom_unsync: PhantomData<Unsync>, // only one thread at a time should use a pool
}

impl CommandPool {
    pub fn new(device: &render::Device, name: &str) -> Self {
        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(device.queue_family_index);

        let handle = unsafe { device.raw.create_command_pool(&command_pool_create_info, None).unwrap() };

        device.set_debug_name(handle, &format!("{name}_command_pool"));

        Self {
            handle,
            name: name.to_owned(),
            command_buffers: Vec::new(),
            used_buffers: 0,

            _phantom_unsync: PhantomData,
        }
    }

    #[inline(always)]
    pub fn buffers(&self) -> &[CommandBuffer] {
        &self.command_buffers[0..self.used_buffers]
    }

    pub fn reset(&mut self, device: &render::Device) {
        unsafe {
            device.raw.reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty()).unwrap();
            for command_buffer in self.command_buffers.iter_mut() {
                command_buffer.wait_infos.clear();
                command_buffer.signal_infos.clear();
            }
            self.used_buffers = 0;
        }
    }

    pub fn begin_new<'a>(&mut self, device: &render::Device, flags: vk::CommandBufferUsageFlags) -> &mut CommandBuffer {
        let index = self.get_next_command_buffer_index(device);
        let command_buffer = &mut self.command_buffers[index];

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(flags);

        unsafe { device.raw.begin_command_buffer(command_buffer.handle(), &begin_info).unwrap() }

        command_buffer
    }

    fn get_next_command_buffer_index(&mut self, device: &render::Device) -> usize {
        if self.used_buffers >= self.command_buffers.len() {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.handle)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = unsafe { device.raw.allocate_command_buffers(&alloc_info).unwrap()[0] };

            let index = self.command_buffers.len();
            device.set_debug_name(command_buffer, &format!("{}_command_buffer_#{index}", self.name));

            self.command_buffers.push(CommandBuffer {
                command_buffer_info: vk::CommandBufferSubmitInfo {
                    command_buffer,
                    ..Default::default()
                },
                wait_infos: Vec::new(),
                signal_infos: Vec::new(),
            })
        }

        let index = self.used_buffers;
        self.used_buffers += 1;
        index
    }

    pub fn destroy(&self, device: &render::Device) {
        unsafe {
            device.raw.destroy_command_pool(self.handle, None);
        }
    }
}

#[derive(Debug)]
pub struct CommandBuffer {
    command_buffer_info: vk::CommandBufferSubmitInfo,
    pub wait_infos: Vec<vk::SemaphoreSubmitInfo>,
    pub signal_infos: Vec<vk::SemaphoreSubmitInfo>,
}

impl CommandBuffer {
    #[inline(always)]
    pub fn handle(&self) -> vk::CommandBuffer {
        self.command_buffer_info.command_buffer
    }

    fn submit_info(&self) -> vk::SubmitInfo2 {
        vk::SubmitInfo2::builder()
            .command_buffer_infos(std::slice::from_ref(&self.command_buffer_info))
            .wait_semaphore_infos(&self.wait_infos)
            .signal_semaphore_infos(&self.signal_infos)
            .build()
    }

    pub fn wait_semaphore(&mut self, semaphore: vk::Semaphore, stage: vk::PipelineStageFlags2) {
        self.wait_infos
            .push(vk::SemaphoreSubmitInfo::builder().semaphore(semaphore).stage_mask(stage).build())
    }

    pub fn signal_semaphore(&mut self, semaphore: vk::Semaphore, stage: vk::PipelineStageFlags2) {
        self.signal_infos
            .push(vk::SemaphoreSubmitInfo::builder().semaphore(semaphore).stage_mask(stage).build())
    }

    pub fn record<'a>(
        &'a mut self,
        device: &'a render::Device,
        descriptor: &'a render::BindlessDescriptors,
    ) -> CommandRecorder<'a> {
        CommandRecorder {
            device,
            descriptor,
            command_buffer: self,
        }
    }
}

impl render::Device {
    pub fn submit(&self, command_buffers: &[CommandBuffer], fence: vk::Fence) {
        let submit_infos: Vec<_> = command_buffers.iter().map(|buf| buf.submit_info()).collect();
        unsafe {
            self.raw.queue_submit2(self.queue, &submit_infos, fence).unwrap();
        }
    }
}

pub struct CommandRecorder<'a> {
    pub device: &'a render::Device,
    pub descriptor: &'a render::BindlessDescriptors,
    pub command_buffer: &'a mut CommandBuffer,
}

impl<'a> CommandRecorder<'a> {
    #[inline(always)]
    pub fn buffer(&self) -> vk::CommandBuffer {
        self.command_buffer.handle()
    }

    pub fn begin_debug_label(&self, name: &str, color: Option<[f32; 4]>) {
        if let Some(ref debug_utils) = self.device.debug_utils_fns {
            unsafe {
                let cname = CString::new(name).unwrap();
                let label = vk::DebugUtilsLabelEXT::builder()
                    .label_name(cname.as_c_str())
                    .color(color.unwrap_or([0.0; 4]));
                debug_utils.cmd_begin_debug_utils_label(self.buffer(), &label);
            }
        }
    }

    #[inline(always)]
    pub fn end_debug_label(&self) {
        if let Some(ref debug_utils) = self.device.debug_utils_fns {
            unsafe {
                debug_utils.cmd_end_debug_utils_label(self.buffer());
            }
        }
    }

    #[inline(always)]
    pub fn copy_buffer(&self, src: &render::BufferView, dst: &render::BufferView, regions: &[vk::BufferCopy]) {
        unsafe {
            self.device.raw.cmd_copy_buffer(
                self.buffer(),
                src.handle,
                dst.handle,
                regions
            )
        }
    }

    #[inline(always)]
    pub fn copy_buffer_to_image(
        &self,
        src: &render::BufferView,
        dst: &render::ImageView,
        regions: &[vk::BufferImageCopy]
    ) {
        unsafe {
            self.device.raw.cmd_copy_buffer_to_image(
                self.buffer(),
                src.handle,
                dst.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions
            )
        }
    }

    #[inline(always)]
    pub fn barrier(
        &self,
        buffer_barriers: &[vk::BufferMemoryBarrier2],
        image_barriers: &[vk::ImageMemoryBarrier2],
        memory_barriers: &[vk::MemoryBarrier2],
    ) {
        let dependency_info = vk::DependencyInfo::builder()
            .buffer_memory_barriers(buffer_barriers)
            .image_memory_barriers(image_barriers)
            .memory_barriers(memory_barriers);

        unsafe {
            self.device.raw.cmd_pipeline_barrier2(self.buffer(), &dependency_info);
        }
    }

    #[inline(always)]
    pub fn blit_image(&self, blit_info: &vk::BlitImageInfo2) {
        unsafe {
            self.device.raw.cmd_blit_image2(self.buffer(), blit_info);
        }
    }

    #[inline(always)]
    pub fn begin_rendering(&self, rendering_info: &vk::RenderingInfo) {
        unsafe {
            self.device.raw.cmd_begin_rendering(self.buffer(), rendering_info);

            let offset = rendering_info.render_area.offset;
            let extent = rendering_info.render_area.extent;

            self.device.raw.cmd_set_viewport(
                self.buffer(),
                0,
                &[vk::Viewport {
                    x: offset.x as f32,
                    y: offset.y as f32 + extent.height as f32,
                    width: extent.width as f32,
                    height: -(extent.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );

            self.device.raw.cmd_set_scissor(self.buffer(), 0, std::slice::from_ref(&rendering_info.render_area));
        }
    }

    #[inline(always)]
    pub fn end_rendering(&self) {
        unsafe {
            self.device.raw.cmd_end_rendering(self.buffer());
        }
    }
    #[inline(always)]
    pub fn set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.device.raw.cmd_set_viewport(self.buffer(), first_viewport, viewports);
        }
    }

    #[inline(always)]
    pub fn set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device.raw.cmd_set_scissor(self.buffer(), first_scissor, scissors);
        }
    }

    #[inline(always)]
    pub fn bind_raster_pipeline(&self, pipeline: render::RasterPipeline) {
        unsafe {
            self.device.raw.cmd_bind_pipeline(self.buffer(), vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
        }
    }

    #[inline(always)]
    pub fn bind_index_buffer(&self, buffer: &render::BufferView) {
        unsafe {
            self.device.raw.cmd_bind_index_buffer(self.buffer(), buffer.handle, 0, vk::IndexType::UINT32);
        }
    }

    #[inline(always)]
    pub fn bind_vertex_buffer(&self, binding: u32, buffer: &render::BufferView, offset: u64) {
        unsafe {
            self.device.raw.cmd_bind_vertex_buffers(self.buffer(), binding, &[buffer.handle], &[offset])
        }
    }

    #[inline(always)]
    pub fn push_bindings(&self, bindings: &[render::RawDescriptorIndex]) {
        unsafe {
            self.device.raw.cmd_push_constants(
                self.buffer(),
                self.descriptor.layout(),
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::cast_slice(bindings),
            )
        }
    }

    #[inline(always)]
    pub fn draw(&self, vertices: Range<u32>, instances: Range<u32>) {
        unsafe {
            self.device.raw.cmd_draw(
                self.buffer(),
                vertices.len() as u32,
                instances.len() as u32,
                vertices.start,
                instances.start,
            );
        }
    }

    #[inline(always)]
    pub fn draw_indexed(&self, indices: Range<u32>, instances: Range<u32>, vertex_offset: i32) {
        unsafe {
            self.device.raw.cmd_draw_indexed(
                self.buffer(),
                indices.len() as u32,
                instances.len() as u32,
                indices.start,
                vertex_offset,
                instances.start,
            );
        }
    }

    #[inline(always)]
    pub fn draw_indexed_indirect(
        &self,
        indirect_buffer: &render::BufferView,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32
    ) {
        unsafe {
            self.device.raw.cmd_draw_indexed_indirect(
                self.buffer(),
                indirect_buffer.handle,
                offset,
                draw_count,
                stride
            );
        }
    }

    #[inline(always)]
    pub fn draw_indexed_indirect_count(
        &self,
        indirect_buffer: &render::BufferView,
        indirect_buffer_offset: vk::DeviceSize,
        count_buffer: &render::BufferView,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.device.raw.cmd_draw_indexed_indirect_count(
                self.buffer(),
                indirect_buffer.handle,
                indirect_buffer_offset,
                count_buffer.handle,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        }
    }

    #[inline(always)]
    pub fn bind_compute_pipeline(&self, pipeline: render::ComputePipeline) {
        unsafe {
            self.device.raw.cmd_bind_pipeline(self.buffer(), vk::PipelineBindPoint::COMPUTE, pipeline.handle);
        }
    }

    #[inline(always)]
    pub fn dispatch(&self, group_counts: [u32; 3]) {
        unsafe {
            self.device.raw.cmd_dispatch(self.buffer(), group_counts[0], group_counts[1], group_counts[2])
        }
    }
}

impl Drop for CommandRecorder<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.end_command_buffer(self.buffer()).unwrap();
        }
    }
}
