use crate::{graphics, Camera};

#[derive(Debug, Clone, Copy)]
pub struct ClusterSettings {
    px_size_power: u32,
    z_slice_count: u32,
    far_plane: f32,
}

impl Default for ClusterSettings {
    fn default() -> Self {
        Self {
            px_size_power: 3, // 2^3 = 8
            z_slice_count: 24,
            far_plane: 100.0,
        }
    }
}

impl ClusterSettings {
    pub fn px_size(&self) -> u32 {
        u32::pow(2, self.px_size_power)
    }

    pub fn edit(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("pixel size");
            ui.add(
                egui::Slider::new(&mut self.px_size_power, 2..=6)
                    .custom_formatter(|n, _| u32::pow(2, n as u32).to_string()),
            );
        });
        ui.horizontal(|ui| {
            ui.label("z slice count");
            ui.add(egui::DragValue::new(&mut self.z_slice_count).clamp_range(1..=32));
        });
        ui.horizontal(|ui| {
            ui.label("z far plane");
            ui.add(egui::DragValue::new(&mut self.far_plane).clamp_range(10.0..=200.0));
        });
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuClusterGridInfo {
    z_scale: f32,
    z_bias: f32,
    z_slice_count: u32,
}

impl GpuClusterGridInfo {
    pub fn new(settings: &ClusterSettings, z_near: f32) -> Self {
        let far = settings.far_plane;
        let near = z_near;
        let num_slices = settings.z_slice_count as f32;
        let log_f_n = f32::log2(far / near);
        Self {
            z_scale: num_slices / log_f_n,
            z_bias: -((num_slices * near.log2()) / log_f_n),
            z_slice_count: settings.z_slice_count,
        }
    }
}

fn generate_clusters(
    context: &mut graphics::Context,
    settings: &ClusterSettings,
    camera: &Camera,
) {
    
}