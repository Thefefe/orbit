use ash::{
    vk,
    extensions::{khr, ext},
};

use std::{
    collections::HashSet,
    ffi::{c_void, CStr, CString},
    mem::ManuallyDrop,
    sync::{Mutex}, ptr::NonNull,
};

use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc, Allocation, AllocationCreateDesc},
    AllocatorDebugSettings,
};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

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

#[derive(Debug, Clone, Copy)]
pub struct QueueFamily {
    pub index: u32,
    pub flags: vk::QueueFlags,
    pub count: u32,
    pub surface_support: bool,
}

impl QueueFamily {
    #[inline]
    pub fn graphics(&self) -> bool {
        self.flags.contains(vk::QueueFlags::GRAPHICS)
    }

    #[inline]
    pub fn compute(&self) -> bool {
        self.flags.contains(vk::QueueFlags::COMPUTE)
    }

    #[inline]
    pub fn transfer(&self) -> bool {
        self.flags.contains(vk::QueueFlags::TRANSFER)
    }

    #[inline]
    pub fn present(&self) -> bool {
        self.surface_support
    }

    #[inline]
    pub fn graphics_present(&self) -> bool {
        self.graphics() && self.present()
    }

    #[inline]
    pub fn dedicated_compute(&self) -> bool {
        self.compute() && !self.graphics()
    }

    #[inline]
    pub fn dedicated_transfer(&self) -> bool {
        self.transfer() && !self.graphics() && !self.compute()
    }

    #[inline]
    pub fn universal(&self) -> bool {
        self.graphics() && self.compute() && self.transfer() && self.present()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SurfaceInfo {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub formats: Vec<vk::SurfaceFormatKHR>,
}

impl SurfaceInfo {
    unsafe fn new(surface_fns: &khr::Surface, physical_device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> Self {
        let capabilities = surface_fns.get_physical_device_surface_capabilities(physical_device, surface).unwrap();
        let present_modes = surface_fns.get_physical_device_surface_present_modes(physical_device, surface).unwrap();
        let formats = surface_fns.get_physical_device_surface_formats(physical_device, surface).unwrap();

        Self {
            capabilities,
            present_modes,
            formats,
        }
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
                    let flags = queue_family.queue_flags;
                    let count = queue_family.queue_count;
                    let surface_support =
                        surface_fns.get_physical_device_surface_support(handle, index, surface).unwrap();

                    QueueFamily {
                        index,
                        flags,
                        count,
                        surface_support,
                    }
                })
                .collect();

            let extensions = instance
                .enumerate_device_extension_properties(handle)
                .unwrap()
                .into_iter()
                .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()).to_owned())
                .collect();

            let surface_info = if queue_families.iter().any(|queue| queue.present()) {
                SurfaceInfo::new(surface_fns, handle, surface)
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
            _ => None
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
            DeviceCreateError::LoadError(err)       => write!(f, "vulkan loading error: {err}"),
            DeviceCreateError::NoSuitableVersion    => write!(f, "vulkan 1.3 required"),
            DeviceCreateError::MissingInstanceExtensions(ext) => write!(f, "missing instance extensions: {ext:?}"),
            DeviceCreateError::NoSuitableGpu        => write!(f, "no suitable graphics device found"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AllocIndex(thunderdome::Index);

struct AllocatorStuff {
    allocator: ManuallyDrop<Allocator>,
    allocations: thunderdome::Arena<Allocation>,
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

    pub queue_family_index: u32,
    pub queue: vk::Queue,

    allocator_stuff: Mutex<AllocatorStuff>,

    pub swapchain_fns: khr::Swapchain,
}

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
            .message_severity(Severity::VERBOSE | Severity::WARNING | Severity::ERROR)
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

        let required_device_extensions: &[&CStr] = &[khr::Swapchain::name()];

        let optional_device_extensions: &[&CStr] = &[];

        let gpu = enumerate_gpus(&instance, &surface_fns, surface)
            .rev()
            .filter(|gpu| {
                let universal_queue = gpu.queues().any(|queue| queue.universal());
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

        let universal_queue_family = gpu.queues().find(|queue| queue.universal()).copied().unwrap();

        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(universal_queue_family.index)
            .queue_priorities(&[1.0]);

        let vulkan10_features = vk::PhysicalDeviceFeatures::builder().build();

        let mut vulkan11_features = vk::PhysicalDeviceVulkan11Features::builder().build();

        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .buffer_device_address(true)
            .build();

        let mut vulkan13_features =
            vk::PhysicalDeviceVulkan13Features::builder().dynamic_rendering(true).synchronization2(true).build();

        let mut device_features = vk::PhysicalDeviceFeatures2::builder()
            .features(vulkan10_features)
            .push_next(&mut vulkan11_features)
            .push_next(&mut vulkan12_features)
            .push_next(&mut vulkan13_features);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&enabled_device_extension_ptrs)
            .push_next(&mut device_features);

        let device =
            unsafe { instance.create_device(gpu.handle, &device_create_info, None) }.expect("failed to create device");

        let device_metadata = DeviceMetadata {
            enabled_extensions: enabled_device_extensions,
        };

        let queue_family_index = universal_queue_family.index;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let swapchain_fns = khr::Swapchain::new(&instance, &device);

        let allocator_stuff = {
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: gpu.handle,
                debug_settings: AllocatorDebugSettings::default(),
                buffer_device_address: true,
            }).expect("failed to create vulkan allocator");

            let allocations = thunderdome::Arena::new();

            Mutex::new(AllocatorStuff {
                allocator: ManuallyDrop::new(allocator),
                allocations,
            })
        };

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

            queue_family_index,
            queue,

            allocator_stuff,

            swapchain_fns,
        })
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

    pub fn create_semaphore(&self, name: &str) -> vk::Semaphore {
        let handle =  unsafe { self.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() };
        self.set_debug_name(handle, name);
        handle
    }

    pub fn get_buffer_address(&self, buffer: vk::Buffer) -> u64 {
        unsafe { self.raw.get_buffer_device_address(&vk::BufferDeviceAddressInfo::builder().buffer(buffer)) }
    }

    #[inline]
    pub fn set_debug_name<T: vk::Handle>(&self, handle: T, name: &str) {
        if let Some(debug_utils) = &self.debug_utils_fns {
            unsafe {
                let cname = CString::new(name).unwrap();
                let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                    .object_type(T::TYPE)
                    .object_handle(handle.as_raw())
                    .object_name(cname.as_c_str());

                debug_utils.set_debug_utils_object_name(self.raw.handle(), &name_info)
                    .unwrap();
            }
        }
    }

    pub fn allocate(
        &self,
        alloc_desc: &AllocationCreateDesc,
    ) -> (AllocIndex, vk::DeviceMemory, u64, Option<NonNull<u8>>) {
        let mut allocator_stuff = self.allocator_stuff.lock().unwrap();
        let allocation = allocator_stuff.allocator.allocate(alloc_desc)
            .expect("failed to allocate memory");

        let memory = unsafe { allocation.memory() };
        let offset = allocation.offset();
        let mapped_ptr = allocation.mapped_ptr().map(|ptr| ptr.cast());

        let index = allocator_stuff.allocations.insert(allocation);

        (AllocIndex(index), memory, offset, mapped_ptr)
    }

    pub fn deallocate(&self, AllocIndex(index): AllocIndex) {
        let mut allocator_stuff = self.allocator_stuff.lock().unwrap();
        if let Some(allocation) = allocator_stuff.allocations.remove(index) {
            allocator_stuff.allocator.free(allocation).expect("failed to deallocate memory");
        }
    }

    pub fn destroy(&mut self) {
        unsafe {
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

unsafe extern "system" fn vk_debug_log_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
    use vk::DebugUtilsMessageTypeFlagsEXT as MessageType;

    let level = match message_severity {
        Severity::VERBOSE => log::Level::Trace,
        Severity::WARNING => log::Level::Warn,
        Severity::ERROR => log::Level::Error,
        Severity::INFO => log::Level::Info,
        _ => log::Level::Debug,
    };
    let target = match message_type {
        MessageType::GENERAL => "vk_general",
        MessageType::PERFORMANCE => "vk_performance",
        MessageType::VALIDATION => "vk_validation",
        _ => "vk_unknown",
    };
    let message_cstr = CStr::from_ptr((*p_callback_data).p_message);

    if let Ok(message) = message_cstr.to_str() {
        log::log!(target: target, level, "{}", message);
    } else {
        log::error!("failed to parse debug callback message, displaying cstr...");
        log::log!(target: target, level, "{:?}", message_cstr);
    }

    vk::FALSE
}
