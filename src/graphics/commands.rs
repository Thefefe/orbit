use ash::vk;
use std::ffi::CString;
use std::ops::Range;

use crate::graphics;

pub struct CommandPool {
    name: String,
    handle: vk::CommandPool,
    queue_type: graphics::QueueType,
    command_buffers: Vec<CommandBuffer>,
    used_buffers: usize,
}

impl CommandPool {
    pub fn new(device: &graphics::Device, name: &str, queue_type: graphics::QueueType) -> Self {
        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(device.get_queue(queue_type).family.index);

        let handle = unsafe { device.raw.create_command_pool(&command_pool_create_info, None).unwrap() };

        device.set_debug_name(handle, &format!("{name}_command_pool"));

        Self {
            name: name.to_owned(),
            handle,
            queue_type,
            command_buffers: Vec::new(),
            used_buffers: 0,
        }
    }

    #[inline(always)]
    pub fn buffers(&self) -> &[CommandBuffer] {
        &self.command_buffers[0..self.used_buffers]
    }

    pub fn reset(&mut self, device: &graphics::Device) {
        unsafe {
            device.raw.reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty()).unwrap();
            for command_buffer in self.command_buffers.iter_mut() {
                command_buffer.wait_infos.clear();
                command_buffer.signal_infos.clear();
            }
            self.used_buffers = 0;
        }
    }

    pub fn begin_new<'a>(&mut self, device: &graphics::Device) -> &mut CommandBuffer {
        let index = self.get_next_command_buffer_index(device);
        let command_buffer = &mut self.command_buffers[index];

        command_buffer
    }

    fn get_next_command_buffer_index(&mut self, device: &graphics::Device) -> usize {
        if self.used_buffers >= self.command_buffers.len() {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.handle)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = unsafe { device.raw.allocate_command_buffers(&alloc_info).unwrap()[0] };

            let index = self.command_buffers.len();
            device.set_debug_name(command_buffer, &format!("{}_command_buffer_#{index}", self.name));

            self.command_buffers.push(CommandBuffer {
                queue_type: self.queue_type,
                index_within_pool: index,
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

    pub fn destroy(&self, device: &graphics::Device) {
        unsafe {
            device.raw.destroy_command_pool(self.handle, None);
        }
    }
}

#[derive(Debug)]
pub struct CommandBuffer {
    queue_type: graphics::QueueType,
    index_within_pool: usize,
    command_buffer_info: vk::CommandBufferSubmitInfo,
    pub wait_infos: Vec<vk::SemaphoreSubmitInfo>,
    pub signal_infos: Vec<vk::SemaphoreSubmitInfo>,
}

unsafe impl Sync for CommandBuffer {}
unsafe impl Send for CommandBuffer {}

impl CommandBuffer {
    #[inline(always)]
    pub fn handle(&self) -> vk::CommandBuffer {
        self.command_buffer_info.command_buffer
    }

    pub fn submit_info(&self) -> vk::SubmitInfo2 {
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
        device: &'a graphics::Device,
        flags: vk::CommandBufferUsageFlags,
    ) -> CommandRecorder<'a> {
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(flags);

        unsafe { device.raw.begin_command_buffer(self.handle(), &begin_info).unwrap() }

        if self.queue_type.supports_graphics() {
            device.bind_descriptors(self.handle(), vk::PipelineBindPoint::GRAPHICS);
        }
        
        if self.queue_type.supports_compute() {
            device.bind_descriptors(self.handle(), vk::PipelineBindPoint::COMPUTE);
        }

        CommandRecorder {
            device,
            command_buffer: self,
        }
    }
}

pub struct CommandRecorder<'a> {
    pub device: &'a graphics::Device,
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
                let label =
                    vk::DebugUtilsLabelEXT::builder().label_name(cname.as_c_str()).color(color.unwrap_or([0.0; 4]));
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
    pub fn reset_query_pool(&self, pool: vk::QueryPool, range: Range<u32>) {
        unsafe {
            self.device.raw.cmd_reset_query_pool(self.buffer(), pool, range.start, range.len() as u32);
        }
    }

    #[inline(always)]
    pub fn begin_query(&self, pool: vk::QueryPool, query: u32, flags: vk::QueryControlFlags) {
        unsafe {
            self.device.raw.cmd_begin_query(self.buffer(), pool, query, flags);
        }
    }

    #[inline(always)]
    pub fn end_query(&self, pool: vk::QueryPool, query: u32) {
        unsafe {
            self.device.raw.cmd_end_query(self.buffer(), pool, query);
        }
    }

    #[inline(always)]
    pub fn write_query(&self, stage: vk::PipelineStageFlags2, pool: vk::QueryPool, query: u32) {
        unsafe {
            self.device.raw.cmd_write_timestamp2(self.buffer(), stage, pool, query);
        }
    }

    #[inline(always)]
    pub fn fill_buffer(&self, buffer: &graphics::BufferView, offset: u64, size: u64, data: u32) {
        unsafe { self.device.raw.cmd_fill_buffer(self.buffer(), buffer.handle, offset, size, data) }
    }

    #[inline(always)]
    pub fn copy_buffer(&self, src_handle: vk::Buffer, dst_handle: vk::Buffer, regions: &[vk::BufferCopy]) {
        unsafe { self.device.raw.cmd_copy_buffer(self.buffer(), src_handle, dst_handle, regions) }
    }

    #[inline(always)]
    pub fn copy_buffer_to_image(
        &self,
        src_handle: vk::Buffer,
        dst_handle: vk::Image,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.raw.cmd_copy_buffer_to_image(
                self.buffer(),
                src_handle,
                dst_handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            )
        }
    }

    #[inline(always)]
    pub fn copy_image(
        &self,
        src_image: &graphics::ImageView,
        src_layout: vk::ImageLayout,
        dst_image: &graphics::ImageView,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        unsafe {
            self.device.raw.cmd_copy_image(
                self.buffer(),
                src_image.handle,
                src_layout,
                dst_image.handle,
                dst_layout,
                regions,
            );
        }
    }

    #[inline(always)]
    pub fn blit_image(
        &self,
        src_image: &graphics::ImageView,
        src_layout: vk::ImageLayout,
        dst_image: &graphics::ImageView,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        unsafe {
            self.device.raw.cmd_blit_image(
                self.buffer(),
                src_image.handle,
                src_layout,
                dst_image.handle,
                dst_layout,
                regions,
                filter,
            );
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
    pub fn set_depth_bias(&self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        unsafe { self.device.raw.cmd_set_depth_bias(self.buffer(), constant_factor, clamp, slope_factor) }
    }

    #[inline(always)]
    pub fn set_depth_test_enable(&self, enable: bool) {
        unsafe { self.device.raw.cmd_set_depth_test_enable(self.buffer(), enable) }
    }

    #[inline(always)]
    pub fn bind_raster_pipeline(&self, pipeline: graphics::RasterPipeline) {
        unsafe {
            self.device.raw.cmd_bind_pipeline(self.buffer(), vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
        }
    }

    #[inline(always)]
    pub fn bind_index_buffer(&self, buffer: &graphics::BufferView, offset: u64) {
        unsafe {
            self.device.raw.cmd_bind_index_buffer(self.buffer(), buffer.handle, offset, vk::IndexType::UINT32);
        }
    }

    #[inline(always)]
    pub fn bind_vertex_buffer(&self, binding: u32, buffer: &graphics::BufferView, offset: u64) {
        unsafe { self.device.raw.cmd_bind_vertex_buffers(self.buffer(), binding, &[buffer.handle], &[offset]) }
    }

    #[inline(always)]
    pub fn push_constants(&self, constansts: &[u8], offset: u32) {
        unsafe {
            self.device.raw.cmd_push_constants(
                self.buffer(),
                self.device.pipeline_layout,
                vk::ShaderStageFlags::ALL,
                offset,
                constansts,
            )
        }
    }

    pub fn build_constants(&self) -> PushConstantBuilder {
        PushConstantBuilder::new(self)
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
        indirect_buffer: &graphics::BufferView,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.device.raw.cmd_draw_indexed_indirect(
                self.buffer(),
                indirect_buffer.handle,
                offset,
                draw_count,
                stride,
            );
        }
    }

    #[inline(always)]
    pub fn draw_indexed_indirect_count(
        &self,
        indirect_buffer: &graphics::BufferView,
        indirect_buffer_offset: vk::DeviceSize,
        count_buffer: &graphics::BufferView,
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
    pub fn bind_compute_pipeline(&self, pipeline: graphics::ComputePipeline) {
        unsafe {
            self.device.raw.cmd_bind_pipeline(self.buffer(), vk::PipelineBindPoint::COMPUTE, pipeline.handle);
        }
    }

    #[inline(always)]
    pub fn dispatch(&self, group_counts: [u32; 3]) {
        unsafe { self.device.raw.cmd_dispatch(self.buffer(), group_counts[0], group_counts[1], group_counts[2]) }
    }

    pub fn generate_mipmaps(
        &self,
        image: &graphics::ImageView,
        target_layers: Range<u32>,
        initial_access: graphics::AccessKind,
        target_access: graphics::AccessKind,
    ) {
        let mip_levels = image.mip_level();
        let mut mip_width = image.width();
        let mut mip_height = image.height();

        if mip_height == 1 {
            log::warn!("attempted to generate mipmaps for image with only 1 mip levels");
            return;
        }

        self.barrier(
            &[],
            &[
                graphics::image_subresource_barrier(
                    &image,
                    0..1,
                    target_layers.clone(),
                    initial_access,
                    graphics::AccessKind::TransferRead,
                ),
                graphics::image_subresource_barrier(
                    &image,
                    1..mip_levels,
                    target_layers.clone(),
                    initial_access,
                    graphics::AccessKind::TransferWrite,
                ),
            ],
            &[],
        );

        for (last_level, current_level) in (1..mip_levels).enumerate() {
            let last_level = last_level as u32;
            let current_level = current_level as u32;

            let new_mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
            let new_mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };

            if last_level != 0 {
                self.barrier(
                    &[],
                    &[graphics::image_subresource_barrier(
                        &image,
                        last_level..last_level + 1,
                        target_layers.clone(),
                        graphics::AccessKind::TransferWrite,
                        graphics::AccessKind::TransferRead,
                    )],
                    &[],
                );
            }

            let blit_region = vk::ImageBlit {
                src_subresource: image.subresource_layers(last_level, target_layers.clone()),
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width as i32,
                        y: mip_height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: image.subresource_layers(current_level, target_layers.clone()),
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: new_mip_width as i32,
                        y: new_mip_height as i32,
                        z: 1,
                    },
                ],
            };

            self.blit_image(
                &image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                &image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit_region),
                vk::Filter::LINEAR,
            );

            if mip_width > 1 {
                mip_width /= 2
            };
            if mip_height > 1 {
                mip_height /= 2
            };
        }

        self.barrier(
            &[],
            &[
                graphics::image_subresource_barrier(
                    &image,
                    ..mip_levels - 1,
                    target_layers.clone(),
                    graphics::AccessKind::TransferRead,
                    target_access,
                ),
                graphics::image_subresource_barrier(
                    &image,
                    mip_levels - 1..mip_levels,
                    target_layers.clone(),
                    graphics::AccessKind::TransferWrite,
                    target_access,
                ),
            ],
            &[],
        );
    }
}

impl CommandRecorder<'_> {
    pub fn command_buffer_index(&self) -> usize {
        self.command_buffer.index_within_pool
    }
}

impl Drop for CommandRecorder<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.end_command_buffer(self.buffer()).unwrap();
        }
    }
}

pub struct PushConstantBuilder<'a> {
    constants: [u8; 128],
    byte_cursor: usize,
    command_recorder: &'a CommandRecorder<'a>,
}

impl<'a> PushConstantBuilder<'a> {
    pub fn new(command_recorder: &'a CommandRecorder<'a>) -> Self {
        Self {
            constants: [0; 128],
            byte_cursor: 0,
            command_recorder,
        }
    }

    #[inline(always)]
    pub fn reamaining_byte(&self) -> usize {
        128 - self.byte_cursor
    }

    #[track_caller]
    #[inline(always)]
    pub fn push_bytes_with_align(mut self, bytes: &[u8], align: usize) -> Self {
        let padding = self.byte_cursor % align;
        debug_assert!(padding + bytes.len() < self.reamaining_byte());

        let offset = self.byte_cursor + padding;
        self.constants[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.byte_cursor += padding + bytes.len();

        self
    }

    #[track_caller]
    #[inline(always)]
    pub fn array<T: bytemuck::NoUninit>(self, slice: &[T]) -> Self {
        self.push_bytes_with_align(bytemuck::cast_slice(slice), std::mem::align_of_val(slice))
    }

    #[track_caller]
    #[inline(always)]
    pub fn uint(self, val: u32) -> Self {
        self.push_bytes_with_align(bytemuck::bytes_of(&val), std::mem::align_of_val(&val))
    }

    #[track_caller]
    #[inline(always)]
    pub fn float(self, val: impl Into<f32>) -> Self {
        let val = val.into();
        self.push_bytes_with_align(bytemuck::bytes_of(&val), std::mem::align_of_val(&val))
    }

    #[track_caller]
    #[inline(always)]
    pub fn vec2(self, val: impl Into<glam::Vec2>) -> Self {
        let val = val.into();
        self.push_bytes_with_align(bytemuck::bytes_of(&val), std::mem::align_of_val(&val))
    }

    #[track_caller]
    #[inline(always)]
    pub fn vec3(self, val: impl Into<glam::Vec3>) -> Self {
        let val = val.into();
        self.push_bytes_with_align(bytemuck::bytes_of(&val), std::mem::align_of_val(&val))
    }

    #[track_caller]
    #[inline(always)]
    pub fn vec4(self, val: impl Into<glam::Vec4>) -> Self {
        let val = val.into();
        self.push_bytes_with_align(bytemuck::bytes_of(&val), std::mem::align_of_val(&val))
    }

    #[track_caller]
    #[inline(always)]
    pub fn mat4(self, val: &glam::Mat4) -> Self {
        self.push_bytes_with_align(bytemuck::bytes_of(val), std::mem::align_of_val(val))
    }

    #[track_caller]
    #[inline(always)]
    pub fn sampled_image(self, image: &graphics::ImageView) -> Self {
        let descriptor_index = image.sampled_index().expect("image doesn't have sampled descriptor index");
        self.push_bytes_with_align(
            bytemuck::bytes_of(&descriptor_index),
            std::mem::align_of_val(&descriptor_index),
        )
    }

    pub fn storage_image(self, image: &graphics::ImageView) -> Self {
        let descriptor_index = image.storage_index().expect("image doesn't have storage descriptor index");
        self.push_bytes_with_align(
            bytemuck::bytes_of(&descriptor_index),
            std::mem::align_of_val(&descriptor_index),
        )
    }

    #[track_caller]
    #[inline(always)]
    pub fn buffer(self, buffer: &graphics::BufferView) -> Self {
        let address = buffer.descriptor_index.unwrap();
        self.push_bytes_with_align(bytemuck::bytes_of(&address), std::mem::align_of_val(&address))
    }

    pub fn push(self) {}
}

impl Drop for PushConstantBuilder<'_> {
    fn drop(&mut self) {
        self.command_recorder.push_constants(&self.constants[0..self.byte_cursor], 0);
    }
}
