use crate::graphics;

use ash::{
    extensions::{ext, khr},
    vk::{self, Handle},
};

use std::{
    collections::HashSet,
    ffi::{c_void, CStr, CString},
    mem::ManuallyDrop,
    ptr::NonNull,
    sync::Mutex,
};

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    AllocationSizes, AllocatorDebugSettings,
};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

use crate::collections::{arena, index_alloc::IndexAllocator};

const VALIDATION_LAYER: &CStr = cstr::cstr!("VK_LAYER_KHRONOS_validation");

pub struct InstanceMetadata {
    pub api_version: u32,
    pub layers: HashSet<CString>,
    pub extensions: HashSet<CString>,
}

impl InstanceMetadata {
    pub fn available(entry: &ash::Entry) -> Option<Self> {
        let api_version = entry.try_enumerate_instance_version().unwrap()?;
        let layers = entry
            .enumerate_instance_layer_properties()
            .unwrap()
            .into_iter()
            .map(|layer| unsafe { CStr::from_ptr(layer.layer_name.as_ptr()).to_owned() })
            .collect();
        let extensions = entry
            .enumerate_instance_extension_properties(None)
            .unwrap()
            .into_iter()
            .map(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()).to_owned() })
            .collect();

        Some(Self {
            api_version,
            layers,
            extensions,
        })
    }

    #[inline]
    pub fn has_layer(&self, query: &CStr) -> bool {
        self.layers.contains(query)
    }

    #[inline]
    pub fn has_extension(&self, query: &CStr) -> bool {
        self.extensions.contains(query)
    }

    pub fn has_all_extensions(&self, query: &[&CStr]) -> bool {
        query.iter().all(|query| self.extensions.contains(*query))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QueueType {
    #[default]
    Graphics,
    AsyncCompute,
    AsyncTransfer,
}

impl QueueType {
    pub fn from_flags(flags: vk::QueueFlags) -> Option<Self> {
        if flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER) {
            return Some(Self::Graphics);
        }

        if flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER) {
            return Some(Self::AsyncCompute);
        }

        if flags.contains(vk::QueueFlags::TRANSFER) {
            return Some(Self::AsyncTransfer);
        }

        None
    }

    #[inline]
    pub fn supports_graphics(self) -> bool {
        match self {
            QueueType::Graphics => true,
            _ => false,
        }
    }

    #[inline]
    pub fn supports_compute(self) -> bool {
        match self {
            QueueType::AsyncTransfer => false,
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QueueFamily {
    pub index: u32,
    pub ty: QueueType,
    pub count: u32,
    pub supports_present: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SurfaceInfo {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub formats: Vec<vk::SurfaceFormatKHR>,
}

impl SurfaceInfo {
    fn query(surface_fns: &khr::Surface, physical_device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> Self {
        unsafe {
            let capabilities = surface_fns.get_physical_device_surface_capabilities(physical_device, surface).unwrap();
            let present_modes =
                surface_fns.get_physical_device_surface_present_modes(physical_device, surface).unwrap();
            let formats = surface_fns.get_physical_device_surface_formats(physical_device, surface).unwrap();

            Self {
                capabilities,
                present_modes,
                formats,
            }
        }
    }

    pub fn new(device: &Device) -> Self {
        Self::query(&device.surface_fns, device.gpu.handle, device.surface)
    }

    pub fn refresh_capabilities(&mut self, device: &Device) {
        unsafe {
            (device.surface_fns.fp().get_physical_device_surface_capabilities_khr)(
                device.gpu.handle,
                device.surface,
                &mut self.capabilities,
            )
            .result()
            .unwrap();
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GpuProperties {
    pub properties10: vk::PhysicalDeviceProperties,
    pub properties12: vk::PhysicalDeviceVulkan12Properties,
}

impl GpuProperties {
    fn new(instance: &ash::Instance, handle: vk::PhysicalDevice) -> Self {
        let mut properties12 = vk::PhysicalDeviceVulkan12Properties::default();

        let mut properties = vk::PhysicalDeviceProperties2::builder().push_next(&mut properties12);

        unsafe {
            instance.get_physical_device_properties2(handle, &mut properties);
        }

        Self {
            properties10: properties.properties,
            properties12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub handle: vk::PhysicalDevice,
    pub properties: GpuProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub queue_families: Vec<QueueFamily>,
    pub extensions: HashSet<CString>,

    pub surface_info: SurfaceInfo,
}

impl GpuInfo {
    #[inline]
    pub fn name(&self) -> &str {
        unsafe {
            // SAFETY: this should always be valid utf-8
            core::str::from_utf8_unchecked(CStr::from_ptr(self.properties.properties10.device_name.as_ptr()).to_bytes())
        }
    }

    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.properties.properties10.device_type
    }

    #[inline]
    pub fn queues(&self) -> impl Iterator<Item = &QueueFamily> {
        self.queue_families.iter()
    }

    #[inline]
    pub fn has_extensions(&self, query: &CStr) -> bool {
        self.extensions.contains(query)
    }

    #[inline]
    pub fn has_all_extensions(&self, query: &[&CStr]) -> bool {
        query.iter().all(|&query| self.extensions.contains(query))
    }

    pub fn supported_multisample_counts(&self) -> impl Iterator<Item = graphics::MultisampleCount> + '_ {
        graphics::MultisampleCount::ALL.into_iter().filter(|sample_count| {
            self.properties.properties10.limits.framebuffer_color_sample_counts.contains(sample_count.to_vk())
        })
    }
}

fn enumerate_gpus<'a>(
    instance: &'a ash::Instance,
    surface_fns: &'a khr::Surface,
    surface: vk::SurfaceKHR,
) -> impl Iterator<Item = GpuInfo> + DoubleEndedIterator + 'a {
    unsafe {
        instance.enumerate_physical_devices().unwrap().into_iter().map(move |handle| {
            let properties = GpuProperties::new(instance, handle);
            let features = instance.get_physical_device_features(handle);
            let queue_families: Vec<_> = instance
                .get_physical_device_queue_family_properties(handle)
                .into_iter()
                .enumerate()
                .map(|(index, queue_family)| {
                    let index = index as u32;
                    let count = queue_family.queue_count;
                    let supports_present =
                        surface_fns.get_physical_device_surface_support(handle, index, surface).unwrap();

                    QueueFamily {
                        index,
                        ty: QueueType::from_flags(queue_family.queue_flags).unwrap(),
                        count,
                        supports_present,
                    }
                })
                .collect();

            let extensions = instance
                .enumerate_device_extension_properties(handle)
                .unwrap()
                .into_iter()
                .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()).to_owned())
                .collect();

            let surface_info = if queue_families.iter().any(|queue| queue.supports_present) {
                SurfaceInfo::query(surface_fns, handle, surface)
            } else {
                SurfaceInfo::default() // SAFETY: we don't use it if not supported
            };

            GpuInfo {
                handle,
                properties,
                features,
                queue_families,
                extensions,
                surface_info,
            }
        })
    }
}

pub struct DeviceMetadata {
    enabled_extensions: HashSet<CString>,
}

#[derive(Debug)]
pub enum DeviceCreateError {
    LoadError(ash::LoadingError),
    NoSuitableVersion,
    MissingInstanceExtensions(Vec<CString>),
    NoSuitableGpu,
}

impl std::error::Error for DeviceCreateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DeviceCreateError::LoadError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ash::LoadingError> for DeviceCreateError {
    fn from(value: ash::LoadingError) -> Self {
        Self::LoadError(value)
    }
}

impl std::fmt::Display for DeviceCreateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceCreateError::LoadError(err) => write!(f, "vulkan loading error: {err}"),
            DeviceCreateError::NoSuitableVersion => write!(f, "vulkan 1.3 required"),
            DeviceCreateError::MissingInstanceExtensions(ext) => write!(f, "missing instance extensions: {ext:?}"),
            DeviceCreateError::NoSuitableGpu => write!(f, "no suitable graphics device found"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AllocIndex(arena::Index);

impl AllocIndex {
    pub fn null() -> Self {
        Self(arena::Index::null())
    }
}

struct AllocatorStuff {
    allocator: ManuallyDrop<Allocator>,
    allocations: arena::Arena<Allocation>,
}

pub struct Queue {
    pub handle: vk::Queue,
    pub family: QueueFamily,
}

pub struct Device {
    pub entry: ash::Entry,

    pub instance: ash::Instance,
    pub instance_metadata: InstanceMetadata,

    pub debug_utils_fns: Option<ext::DebugUtils>,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    pub surface_fns: khr::Surface,
    pub surface: vk::SurfaceKHR,

    pub gpu: GpuInfo,

    pub raw: ash::Device,
    pub device_metadata: DeviceMetadata,

    pub graphics_queue: Queue,
    pub async_compute_queue: Option<Queue>,
    pub async_transfer_queue: Option<Queue>,
    pub queue_family_count: u32,
    queue_family_indices: [u32; 3],

    allocator_stuff: Mutex<AllocatorStuff>,

    pub swapchain_fns: khr::Swapchain,
    pub mesh_shader_fns: Option<ext::MeshShader>,

    // descriptor stuff
    descriptor_layouts: Vec<vk::DescriptorSetLayout>,
    pub pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    global_descriptor_index_allocator: IndexAllocator,
    immutable_samplers: [vk::Sampler; SAMPLER_COUNT],
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}

impl Device {
    pub fn new(window: &Window) -> Result<Self, DeviceCreateError> {
        let entry = unsafe { ash::Entry::load()? };
        let name = cstr::cstr!("orbit");
        let version = vk::make_api_version(0, 0, 0, 1);

        let application_info = vk::ApplicationInfo::builder()
            .application_name(name)
            .application_version(version)
            .engine_name(name)
            .engine_version(version)
            .api_version(vk::API_VERSION_1_3);

        let available_metadata = InstanceMetadata::available(&entry).ok_or(DeviceCreateError::NoSuitableVersion)?;

        if vk::api_version_minor(available_metadata.api_version) < 3 {
            return Err(DeviceCreateError::NoSuitableVersion);
        }

        let mut layers: Vec<*const i8> = Vec::new();
        let mut extensions: Vec<*const i8> = Vec::new();

        if available_metadata.has_layer(VALIDATION_LAYER) {
            layers.push(VALIDATION_LAYER.as_ptr())
        }

        let required_instance_extensions =
            ash_window::enumerate_required_extensions(window.raw_display_handle()).unwrap();

        let required_extension_names = required_instance_extensions
            .iter()
            .map(|&ptr| unsafe { CStr::from_ptr(ptr).to_owned() })
            .collect::<HashSet<_>>();

        if required_extension_names.is_subset(&available_metadata.extensions) {
            extensions.extend(required_instance_extensions);
        } else {
            let missing_extensions =
                required_extension_names.difference(&available_metadata.extensions).cloned().collect();

            return Err(DeviceCreateError::MissingInstanceExtensions(missing_extensions));
        }

        let debug_utils_supported = available_metadata.has_extension(ext::DebugUtils::name());

        if debug_utils_supported {
            extensions.push(ext::DebugUtils::name().as_ptr());
        }

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
        use vk::DebugUtilsMessageTypeFlagsEXT as MessageType;

        // MUST be the last in the instance create info pointer chain!!!
        let mut debug_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(Severity::VERBOSE | Severity::WARNING | Severity::ERROR | Severity::INFO)
            .message_type(MessageType::GENERAL | MessageType::VALIDATION | MessageType::PERFORMANCE)
            .pfn_user_callback(Some(vk_debug_log_callback));

        if debug_utils_supported {
            instance_create_info = instance_create_info.push_next(&mut debug_messenger_create_info);
        }

        let instance =
            unsafe { entry.create_instance(&instance_create_info, None) }.expect("failed to create instance");

        let layers = layers.into_iter().map(|ptr| unsafe { CStr::from_ptr(ptr).to_owned() }).collect();

        let extensions = extensions.into_iter().map(|ptr| unsafe { CStr::from_ptr(ptr).to_owned() }).collect();

        let instance_metadata = InstanceMetadata {
            api_version: available_metadata.api_version,
            layers,
            extensions,
        };

        let debug_utils_fns = if debug_utils_supported {
            Some(ext::DebugUtils::new(&entry, &instance))
        } else {
            None
        };

        let debug_messenger = debug_utils_fns.as_ref().map(|debug_utils| unsafe {
            debug_utils.create_debug_utils_messenger(&debug_messenger_create_info, None).unwrap()
        });

        let surface_fns = khr::Surface::new(&entry, &instance);

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
        }
        .unwrap();

        let required_device_extensions: &[&CStr] = &[
            khr::Swapchain::name(),
            vk::ExtIndexTypeUint8Fn::name(),
        ];

        let optional_device_extensions: &[&CStr] = &[
            ext::MeshShader::name(),
        ];

        let gpu = enumerate_gpus(&instance, &surface_fns, surface)
            .rev()
            .filter(|gpu| {
                let universal_queue =
                    gpu.queues().any(|queue| queue.ty == QueueType::Graphics && queue.supports_present);
                let required_extensions = gpu.has_all_extensions(required_device_extensions);

                universal_queue && required_extensions
            })
            .max_by_key(|gpu| {
                let mut score = 0;

                if gpu.device_type() == vk::PhysicalDeviceType::DISCRETE_GPU {
                    score += 10;
                } else if gpu.device_type() == vk::PhysicalDeviceType::INTEGRATED_GPU {
                    score += 1;
                }

                for extension in optional_device_extensions {
                    if gpu.has_extensions(extension) {
                        score += 1;
                    }
                }

                score
            })
            .ok_or(DeviceCreateError::NoSuitableGpu)?;

        let required_device_extensions = required_device_extensions.iter().map(|&ext| ext.to_owned());

        let available_optional_device_extensions =
            optional_device_extensions.iter().filter(|ext| gpu.has_extensions(ext)).map(|&ext| ext.to_owned());

        let enabled_device_extensions =
            HashSet::from_iter(required_device_extensions.chain(available_optional_device_extensions));

        let enabled_device_extension_ptrs =
            enabled_device_extensions.iter().map(|ext| ext.as_c_str().as_ptr()).collect::<Vec<_>>();

        log::info!("selected gpu: {}", gpu.name());

        let mut queue_create_infos = Vec::new();

        let graphics_queue_family = gpu
            .queues()
            .find(|queue| queue.ty == QueueType::Graphics && queue.supports_present)
            .copied()
            .unwrap();
        let async_compute_queue_family = gpu.queues().find(|queue| queue.ty == QueueType::AsyncCompute).copied();
        let async_transfer_queue_family = gpu.queues().find(|queue| queue.ty == QueueType::AsyncTransfer).copied();

        let queue_priorities = [1.0];

        queue_create_infos.push(
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family.index)
                .queue_priorities(&queue_priorities)
                .build(),
        );

        if let Some(async_comptute_queue_family) = async_compute_queue_family {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(async_comptute_queue_family.index)
                    .queue_priorities(&queue_priorities)
                    .build(),
            );
        }

        if let Some(async_transfer_queue_family) = async_transfer_queue_family {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(async_transfer_queue_family.index)
                    .queue_priorities(&queue_priorities)
                    .build(),
            );
        }

        let vulkan10_features = vk::PhysicalDeviceFeatures::builder()
            .fill_mode_non_solid(true)
            .sample_rate_shading(true)
            .multi_draw_indirect(true)
            .sampler_anisotropy(true)
            .depth_bias_clamp(true)
            .depth_clamp(true)
            .shader_int64(true)
            .shader_int16(true)
            .fragment_stores_and_atomics(true)
            .build();

        let mut vulkan11_features = vk::PhysicalDeviceVulkan11Features::builder()
            .shader_draw_parameters(true)
            .storage_buffer16_bit_access(true);

        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .buffer_device_address(true)
            .buffer_device_address_capture_replay(true)
            .shader_int8(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .draw_indirect_count(true)
            .host_query_reset(true)
            .sampler_filter_minmax(true);

        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);

        let mut index_type_uint8_features = vk::PhysicalDeviceIndexTypeUint8FeaturesEXT::builder()
            .index_type_uint8(true);

        let mut mesh_shader_features =
            vk::PhysicalDeviceMeshShaderFeaturesEXT::builder().mesh_shader(true).task_shader(true);

        let mut device_features = vk::PhysicalDeviceFeatures2::builder()
            .features(vulkan10_features)
            .push_next(&mut vulkan11_features)
            .push_next(&mut vulkan12_features)
            .push_next(&mut vulkan13_features)
            .push_next(&mut index_type_uint8_features);

        let mesh_shading_available = enabled_device_extensions.contains(ext::MeshShader::name());

        if mesh_shading_available {
            log::info!("mesh shading supported!");
            device_features = device_features.push_next(&mut mesh_shader_features);
        }

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&enabled_device_extension_ptrs)
            .push_next(&mut device_features);

        let device =
            unsafe { instance.create_device(gpu.handle, &device_create_info, None) }.expect("failed to create device");

        let device_metadata = DeviceMetadata {
            enabled_extensions: enabled_device_extensions,
        };

        // helper closure for simple debug name setting only for this function
        // let device_handle = device.handle();
        let set_debug_name = |ty: vk::ObjectType, handle: u64, name: &str| {
            if let Some(ref debug_utils) = debug_utils_fns {
                unsafe {
                    let cname = CString::new(name).unwrap(); // TODO: cache allocation?
                    let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                        .object_type(ty)
                        .object_handle(handle)
                        .object_name(cname.as_c_str());

                    debug_utils.set_debug_utils_object_name(device.handle(), &name_info).unwrap();
                }
            }
        };

        let get_queue = |family: QueueFamily| unsafe {
            Queue {
                handle: device.get_device_queue(family.index, 0),
                family,
            }
        };

        let graphics_queue = get_queue(graphics_queue_family);
        let async_compute_queue = async_compute_queue_family.map(get_queue);
        let async_transfer_queue = async_transfer_queue_family.map(get_queue);

        set_debug_name(vk::Queue::TYPE, graphics_queue.handle.as_raw(), "grahics_queue");

        if let Some(ref queue) = async_compute_queue {
            set_debug_name(vk::Queue::TYPE, queue.handle.as_raw(), "compute_queue");
        }

        if let Some(ref queue) = async_compute_queue {
            set_debug_name(vk::Queue::TYPE, queue.handle.as_raw(), "transfer_queue");
        }

        let mut queue_family_count: u32 = 1;
        let mut queue_family_indices = [graphics_queue_family.index, 0, 0];

        if let Some(queue_family) = async_compute_queue_family {
            queue_family_indices[queue_family_count as usize] = queue_family.index;
            queue_family_count += 1;
        }

        if let Some(queue_family) = async_transfer_queue_family {
            queue_family_indices[queue_family_count as usize] = queue_family.index;
            queue_family_count += 1;
        }

        let swapchain_fns = khr::Swapchain::new(&instance, &device);
        let mesh_shader_fns = mesh_shading_available.then(|| ext::MeshShader::new(&instance, &device));

        let allocator_stuff = {
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: gpu.handle,
                debug_settings: AllocatorDebugSettings::default(),
                buffer_device_address: true,
                allocation_sizes: AllocationSizes::default(),
            })
            .expect("failed to create vulkan allocator");

            let allocations = arena::Arena::new();

            Mutex::new(AllocatorStuff {
                allocator: ManuallyDrop::new(allocator),
                allocations,
            })
        };

        // descriptor stuff
        let immutable_samplers = SamplerKind::ALL.map(|sampler_kind| sampler_kind.create(&device));

        let descriptor_layouts: Vec<_> = DescriptorTableType::all_types()
            .map(|desc_type| {
                let mut descriptor_binding_flags = vec![
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                ];

                let mut set = vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: desc_type.to_vk(),
                    descriptor_count: desc_type.max_count(&gpu),
                    stage_flags: vk::ShaderStageFlags::ALL,
                    p_immutable_samplers: std::ptr::null(),
                }];

                if desc_type == DescriptorTableType::SampledImage && !immutable_samplers.is_empty() {
                    descriptor_binding_flags.push(vk::DescriptorBindingFlags::empty());

                    // Set texture binding start at the end of the immutable samplers.
                    set[0].binding = immutable_samplers.len() as u32;
                    set.push(vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        descriptor_count: immutable_samplers.len() as u32,
                        stage_flags: vk::ShaderStageFlags::ALL,
                        p_immutable_samplers: immutable_samplers.as_ptr(),
                    });
                }

                let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                    .binding_flags(&descriptor_binding_flags);

                let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&set)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut ext_flags);

                let layout = unsafe { device.create_descriptor_set_layout(&layout_create_info, None) }.unwrap();
                set_debug_name(
                    vk::DescriptorSetLayout::TYPE,
                    layout.as_raw(),
                    &format!("bindless_{}_layout", desc_type.name()),
                );
                layout
            })
            .collect();

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: 128,
        };

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap() };
        set_debug_name(
            vk::PipelineLayout::TYPE,
            pipeline_layout.as_raw(),
            "bindless_pipeline_layout",
        );

        let pool_sizes: Vec<_> = DescriptorTableType::all_types()
            .map(|desc_ty| vk::DescriptorPoolSize {
                ty: desc_ty.to_vk(),
                descriptor_count: desc_ty.max_count(&gpu),
            })
            .collect();

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(4)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_create_info, None).unwrap() };

        set_debug_name(
            vk::DescriptorPool::TYPE,
            descriptor_pool.as_raw(),
            "bindless_descriptor_pool",
        );

        let descriptor_counts: Vec<_> = DescriptorTableType::all_types().map(|ty| ty.max_count(&gpu)).collect();

        let mut variable_count =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder().descriptor_counts(&descriptor_counts);

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_layouts)
            .push_next(&mut variable_count);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        let names = [
            "buffer_descriptor_set",
            "sampled_image_descriptor_set",
            "storage_image_descriptor_index",
        ];

        for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
            set_debug_name(vk::DescriptorSet::TYPE, descriptor_set.as_raw(), names[i]);
        }

        log::info!("created device");

        Ok(Self {
            entry,

            instance,
            instance_metadata,

            debug_utils_fns,
            debug_messenger,

            surface_fns,
            surface,
            gpu,

            raw: device,
            device_metadata,

            graphics_queue,
            async_compute_queue,
            async_transfer_queue,
            queue_family_count,
            queue_family_indices,

            allocator_stuff,

            swapchain_fns,
            mesh_shader_fns,

            //descriptor stuff
            descriptor_layouts,
            pipeline_layout,

            descriptor_pool,
            descriptor_sets,

            global_descriptor_index_allocator: IndexAllocator::new(0),
            immutable_samplers,
        })
    }

    #[inline]
    pub fn get_queue(&self, ty: QueueType) -> &Queue {
        match ty {
            QueueType::Graphics => &self.graphics_queue,
            QueueType::AsyncCompute => self.async_compute_queue.as_ref().unwrap_or(&self.graphics_queue),
            QueueType::AsyncTransfer => self.async_transfer_queue.as_ref().unwrap_or(&self.graphics_queue),
        }
    }

    pub fn queue_family_indices(&self) -> &[u32] {
        &self.queue_family_indices[0..self.queue_family_count as usize]
    }

    #[inline]
    pub fn queue_submit(&self, ty: QueueType, submits: &[vk::SubmitInfo2], fence: vk::Fence) {
        unsafe { self.raw.queue_submit2(self.get_queue(ty).handle, submits, fence).unwrap() }
    }

    pub fn create_fence(&self, name: &str, signaled: bool) -> vk::Fence {
        let create_info = vk::FenceCreateInfo::builder().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        });

        let handle = unsafe { self.raw.create_fence(&create_info, None).unwrap() };
        self.set_debug_name(handle, name);
        handle
    }

    pub fn get_buffer_address(&self, buffer: vk::Buffer) -> u64 {
        unsafe { self.raw.get_buffer_device_address(&vk::BufferDeviceAddressInfo::builder().buffer(buffer)) }
    }

    pub fn set_debug_name<T: vk::Handle>(&self, handle: T, name: &str) {
        if let Some(debug_utils) = &self.debug_utils_fns {
            unsafe {
                let cname = CString::new(name).unwrap(); // TODO: cache allocation?
                let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                    .object_type(T::TYPE)
                    .object_handle(handle.as_raw())
                    .object_name(cname.as_c_str());

                debug_utils.set_debug_utils_object_name(self.raw.handle(), &name_info).unwrap();
            }
        }
    }

    pub fn allocate(
        &self,
        alloc_desc: &AllocationCreateDesc,
    ) -> (AllocIndex, vk::DeviceMemory, u64, Option<NonNull<u8>>) {
        puffin::profile_function!(alloc_desc.name);
        let mut allocator_stuff = self.allocator_stuff.lock().unwrap();
        let allocation = allocator_stuff.allocator.allocate(alloc_desc).expect("failed to allocate memory");

        let memory = unsafe { allocation.memory() };
        let offset = allocation.offset();
        let mapped_ptr = allocation.mapped_ptr().map(|ptr| ptr.cast());

        let index = allocator_stuff.allocations.insert(allocation);

        (AllocIndex(index), memory, offset, mapped_ptr)
    }

    pub fn deallocate(&self, AllocIndex(index): AllocIndex) {
        puffin::profile_function!();
        let mut allocator_stuff = self.allocator_stuff.lock().unwrap();
        if let Some(allocation) = allocator_stuff.allocations.remove(index) {
            allocator_stuff.allocator.free(allocation).expect("failed to deallocate memory");
        }
    }

    pub fn alloc_descriptor_index(&self) -> DescriptorIndex {
        let index = self.global_descriptor_index_allocator.alloc();
        index
    }

    #[track_caller]
    pub fn free_descriptor_index(&self, index: DescriptorIndex) {
        self.global_descriptor_index_allocator.free(strip_sampler(index));
    }

    pub fn bind_descriptors(&self, command_buffer: vk::CommandBuffer, bind_point: vk::PipelineBindPoint) {
        unsafe {
            self.raw.cmd_bind_descriptor_sets(
                command_buffer,
                bind_point,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            );
        }
    }

    pub fn write_storage_buffer_resource(&self, index: DescriptorIndex, handle: vk::Buffer) {
        unsafe {
            let buffer_info =
                vk::DescriptorBufferInfo::builder().buffer(handle).offset(0).range(vk::WHOLE_SIZE).build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::StorageBuffer.set_index() as usize])
                .dst_binding(0)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::StorageBuffer.to_vk())
                .buffer_info(std::slice::from_ref(&buffer_info));

            self.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }
    }

    pub fn write_sampled_image(&self, index: DescriptorIndex, handle: vk::ImageView) {
        unsafe {
            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(handle)
                .build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::SampledImage.set_index() as usize])
                .dst_binding(self.immutable_samplers.len() as u32)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::SampledImage.to_vk())
                .image_info(std::slice::from_ref(&image_info));

            self.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }
    }

    pub fn write_storage_image(&self, index: DescriptorIndex, handle: vk::ImageView) {
        unsafe {
            let image_info =
                vk::DescriptorImageInfo::builder().image_layout(vk::ImageLayout::GENERAL).image_view(handle).build();

            let write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DescriptorTableType::StorageImage.set_index() as usize])
                .dst_binding(0)
                .dst_array_element(index)
                .descriptor_type(DescriptorTableType::StorageImage.to_vk())
                .image_info(std::slice::from_ref(&image_info));

            self.raw.update_descriptor_sets(std::slice::from_ref(&write_info), &[]);
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            for sampler in self.immutable_samplers.iter() {
                self.raw.destroy_sampler(*sampler, None);
            }

            self.raw.destroy_descriptor_pool(self.descriptor_pool, None);
            self.raw.destroy_pipeline_layout(self.pipeline_layout, None);

            for descriptor_layout in &self.descriptor_layouts {
                self.raw.destroy_descriptor_set_layout(*descriptor_layout, None);
            }

            ManuallyDrop::drop(&mut self.allocator_stuff.lock().unwrap().allocator);

            self.raw.destroy_device(None);

            self.surface_fns.destroy_surface(self.surface, None);

            if let (Some(debug_utils), Some(messenger)) = (&self.debug_utils_fns, &self.debug_messenger) {
                debug_utils.destroy_debug_utils_messenger(*messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DescriptorTableType {
    StorageBuffer = 0,
    SampledImage = 1,
    StorageImage = 2,
}

impl DescriptorTableType {
    fn all_types() -> impl Iterator<Item = Self> {
        [Self::StorageBuffer, Self::SampledImage, Self::StorageImage].into_iter()
    }

    fn set_index(self) -> u32 {
        self as u32
    }

    fn from_set_index(set_index: u32) -> Self {
        match set_index {
            0 => Self::StorageBuffer,
            1 => Self::SampledImage,
            _ => panic!("invalid set index"),
        }
    }

    fn to_vk(self) -> vk::DescriptorType {
        match self {
            DescriptorTableType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            DescriptorTableType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            DescriptorTableType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DescriptorTableType::StorageBuffer => "storage_buffer",
            DescriptorTableType::SampledImage => "sampled_image",
            DescriptorTableType::StorageImage => "storage_image",
        }
    }

    fn max_count(self, gpu: &GpuInfo) -> u32 {
        let props = &gpu.properties.properties12;
        match self {
            DescriptorTableType::StorageBuffer => u32::min(
                props.max_descriptor_set_update_after_bind_storage_buffers,
                props.max_per_stage_descriptor_update_after_bind_storage_buffers,
            ),
            DescriptorTableType::SampledImage => u32::min(
                props.max_descriptor_set_update_after_bind_sampled_images,
                props.max_per_stage_descriptor_update_after_bind_sampled_images,
            ),
            DescriptorTableType::StorageImage => u32::min(
                props.max_descriptor_set_update_after_bind_storage_images,
                props.max_per_stage_descriptor_update_after_bind_storage_images,
            ),
        }
    }
}

pub type DescriptorIndex = u32;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerKind {
    LinearClamp = 0,
    LinearRepeat = 1,
    NearestClamp = 2,
    NearestRepeat = 3,
    ShadowComparison = 4,
    ShadowDepth = 5,
    ReduceMin = 6,
}

const SAMPLER_COUNT: usize = 7;

impl SamplerKind {
    pub const ALL: [SamplerKind; SAMPLER_COUNT] = [
        Self::LinearClamp,
        Self::LinearRepeat,
        Self::NearestClamp,
        Self::NearestRepeat,
        Self::ShadowComparison,
        Self::ShadowDepth,
        Self::ReduceMin,
    ];

    fn create(self, device: &ash::Device) -> vk::Sampler {
        match self {
            SamplerKind::LinearClamp => {
                let create_info = vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(true)
                    .max_anisotropy(16.0)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE);

                unsafe { device.create_sampler(&create_info, None).unwrap() }
            }
            SamplerKind::LinearRepeat => {
                let create_info = vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(true)
                    .max_anisotropy(16.0)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE);

                unsafe { device.create_sampler(&create_info, None).unwrap() }
            }
            SamplerKind::NearestClamp => {
                let create_info = vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(true)
                    .max_anisotropy(16.0)
                    .min_filter(vk::Filter::NEAREST)
                    .mag_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE);

                unsafe { device.create_sampler(&create_info, None).unwrap() }
            }
            SamplerKind::NearestRepeat => {
                let create_info = vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(true)
                    .max_anisotropy(16.0)
                    .min_filter(vk::Filter::NEAREST)
                    .mag_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE);

                unsafe { device.create_sampler(&create_info, None).unwrap() }
            }
            SamplerKind::ShadowComparison => {
                let create_info = vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::GREATER_OR_EQUAL);

                unsafe { device.create_sampler(&create_info, None).unwrap() }
            }
            SamplerKind::ShadowDepth => {
                let create_info = vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_filter(vk::Filter::NEAREST)
                    .mag_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE);

                unsafe { device.create_sampler(&create_info, None).unwrap() }
            }
            SamplerKind::ReduceMin => {
                let mut reduction_mode =
                    vk::SamplerReductionModeCreateInfo::builder().reduction_mode(vk::SamplerReductionMode::MIN);

                let create_info = vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .push_next(&mut reduction_mode);

                unsafe { device.create_sampler(&create_info, None).unwrap() }
            }
        }
    }
}

pub fn strip_sampler(index: DescriptorIndex) -> DescriptorIndex {
    index << 8 >> 8
}

pub fn descriptor_index_with_sampler(index: DescriptorIndex, sampler: SamplerKind) -> DescriptorIndex {
    assert!(index >> 24 == 0);
    index | ((sampler as u32) << 24)
}

unsafe extern "system" fn vk_debug_log_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }

    // use vk::DebugUtilsMessageTypeFlagsEXT as MessageType;
    // let target = match message_type {
    //     MessageType::GENERAL => "vk_general",
    //     MessageType::PERFORMANCE => "vk_performance",
    //     MessageType::VALIDATION => "vk_validation",
    //     _ => "vk_unknown",
    // };

    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
    let level = match message_severity {
        Severity::WARNING => log::Level::Warn,
        Severity::ERROR => log::Level::Error,
        Severity::INFO => log::Level::Trace, // INFO has too much clutter
        Severity::VERBOSE => log::Level::Trace,
        _ => log::Level::Debug,
    };

    let message_cstr = CStr::from_ptr((*p_callback_data).p_message);
    let Ok(message) = message_cstr.to_str() else {
        log::error!("failed to parse debug callback message, displaying cstr...");
        log::log!(target: "vulkan", level, "{:?}", message_cstr);
        return vk::FALSE;
    };

    if message.contains("UNASSIGNED-DEBUG-PRINTF") {
        let message = message.split("|").last().unwrap().trim();
        log::debug!(target: "vulkan", "shader: {message}");
        return vk::FALSE;
    }

    log::log!(target: "vulkan", level, "{}", message);

    vk::FALSE
}
