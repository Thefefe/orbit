use std::{collections::HashMap, path::Path, borrow::Cow};

use ash::vk;
use glam::{Vec4, Mat4, Vec3, Vec3A};
use gpu_allocator::MemoryLocation;
use image::EncodableLayout;

use crate::{
    assets::{sep_vertex_merge, GpuAssetStore, MaterialData, MeshData, ModelHandle, Submesh, TextureHandle, Aabb},
    render,
    scene::{EntityData, SceneBuffer, Transform},
};

fn dxgi_format_to_vk(format : ddsfile::DxgiFormat) -> Option<vk::Format> {
    let vk_format = match format {
        ddsfile::DxgiFormat::R32G32B32A32_Typeless      => vk::Format::R32G32B32A32_SINT,
        ddsfile::DxgiFormat::R32G32B32A32_Float         => vk::Format::R32G32B32A32_SFLOAT,
        ddsfile::DxgiFormat::R32G32B32A32_UInt          => vk::Format::R32G32B32A32_UINT,
        ddsfile::DxgiFormat::R32G32B32A32_SInt          => vk::Format::R32G32B32A32_UINT,
        ddsfile::DxgiFormat::R32G32B32_Typeless         => vk::Format::R32G32B32_SINT,
        ddsfile::DxgiFormat::R32G32B32_Float            => vk::Format::R32G32B32_SFLOAT,
        ddsfile::DxgiFormat::R32G32B32_UInt             => vk::Format::R32G32B32_UINT,
        ddsfile::DxgiFormat::R32G32B32_SInt             => vk::Format::R32G32B32_SINT,
        ddsfile::DxgiFormat::R16G16B16A16_Typeless      => vk::Format::R16G16B16A16_SINT,
        ddsfile::DxgiFormat::R16G16B16A16_Float         => vk::Format::R16G16B16A16_SFLOAT,
        ddsfile::DxgiFormat::R16G16B16A16_UNorm         => vk::Format::R16G16B16A16_UNORM,
        ddsfile::DxgiFormat::R16G16B16A16_UInt          => vk::Format::R16G16B16A16_UINT,
        ddsfile::DxgiFormat::R16G16B16A16_SNorm         => vk::Format::R16G16B16A16_SNORM,
        ddsfile::DxgiFormat::R16G16B16A16_SInt          => vk::Format::R16G16B16A16_SINT,
        ddsfile::DxgiFormat::R32G32_Typeless            => vk::Format::R32G32_SINT,
        ddsfile::DxgiFormat::R32G32_Float               => vk::Format::R32G32_SFLOAT,
        ddsfile::DxgiFormat::R32G32_UInt                => vk::Format::R32G32_UINT,
        ddsfile::DxgiFormat::R32G32_SInt                => vk::Format::R32G32_SINT,
        ddsfile::DxgiFormat::R10G10B10A2_Typeless       => vk::Format::A2R10G10B10_SINT_PACK32,
        ddsfile::DxgiFormat::R10G10B10A2_UNorm          => vk::Format::A2R10G10B10_UNORM_PACK32,
        ddsfile::DxgiFormat::R10G10B10A2_UInt           => vk::Format::A2R10G10B10_UINT_PACK32,
        ddsfile::DxgiFormat::R8G8B8A8_Typeless          => vk::Format::R8G8B8A8_SINT,
        ddsfile::DxgiFormat::R8G8B8A8_UNorm             => vk::Format::R8G8B8A8_UNORM,
        ddsfile::DxgiFormat::R8G8B8A8_UNorm_sRGB        => vk::Format::R8G8B8A8_SRGB,
        ddsfile::DxgiFormat::R8G8B8A8_UInt              => vk::Format::R8G8B8A8_UINT,
        ddsfile::DxgiFormat::R8G8B8A8_SNorm             => vk::Format::R8G8B8A8_SNORM,
        ddsfile::DxgiFormat::R8G8B8A8_SInt              => vk::Format::R8G8B8A8_SINT,
        ddsfile::DxgiFormat::R16G16_Typeless            => vk::Format::R16G16_SINT,
        ddsfile::DxgiFormat::R16G16_Float               => vk::Format::R16G16_SFLOAT,
        ddsfile::DxgiFormat::R16G16_UNorm               => vk::Format::R16G16_UNORM,
        ddsfile::DxgiFormat::R16G16_UInt                => vk::Format::R16G16_UINT,
        ddsfile::DxgiFormat::R16G16_SNorm               => vk::Format::R16G16_SNORM,
        ddsfile::DxgiFormat::R16G16_SInt                => vk::Format::R16G16_SINT,
        ddsfile::DxgiFormat::R32_Typeless               => vk::Format::R32_SINT,
        ddsfile::DxgiFormat::D32_Float                  => vk::Format::D32_SFLOAT,
        ddsfile::DxgiFormat::R32_Float                  => vk::Format::R32_SFLOAT,
        ddsfile::DxgiFormat::R32_UInt                   => vk::Format::R32_UINT,
        ddsfile::DxgiFormat::R32_SInt                   => vk::Format::R32_SINT,
        ddsfile::DxgiFormat::D24_UNorm_S8_UInt          => vk::Format::D24_UNORM_S8_UINT,
        ddsfile::DxgiFormat::R8G8_Typeless              => vk::Format::R8G8_SINT,
        ddsfile::DxgiFormat::R8G8_UNorm                 => vk::Format::R8G8_UNORM,
        ddsfile::DxgiFormat::R8G8_UInt                  => vk::Format::R8G8_UINT,
        ddsfile::DxgiFormat::R8G8_SNorm                 => vk::Format::R8G8_SNORM,
        ddsfile::DxgiFormat::R8G8_SInt                  => vk::Format::R8G8_SINT,
        ddsfile::DxgiFormat::R16_Typeless               => vk::Format::R16_SINT,
        ddsfile::DxgiFormat::R16_Float                  => vk::Format::R16_SFLOAT,
        ddsfile::DxgiFormat::D16_UNorm                  => vk::Format::D16_UNORM,
        ddsfile::DxgiFormat::R16_UNorm                  => vk::Format::R16_UNORM,
        ddsfile::DxgiFormat::R16_UInt                   => vk::Format::R16_UINT,
        ddsfile::DxgiFormat::R16_SNorm                  => vk::Format::R16_SNORM,
        ddsfile::DxgiFormat::R16_SInt                   => vk::Format::R16_SINT,
        ddsfile::DxgiFormat::R8_Typeless                => vk::Format::R8_SINT,
        ddsfile::DxgiFormat::R8_UNorm                   => vk::Format::R8_UNORM,
        ddsfile::DxgiFormat::R8_UInt                    => vk::Format::R8_UINT,
        ddsfile::DxgiFormat::R8_SNorm                   => vk::Format::R8_SNORM,
        ddsfile::DxgiFormat::R8_SInt                    => vk::Format::R8_SINT,
        ddsfile::DxgiFormat::R9G9B9E5_SharedExp         => vk::Format::E5B9G9R9_UFLOAT_PACK32,
        ddsfile::DxgiFormat::BC1_Typeless               => vk::Format::BC1_RGBA_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC1_UNorm                  => vk::Format::BC1_RGBA_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC1_UNorm_sRGB             => vk::Format::BC1_RGBA_SRGB_BLOCK,
        ddsfile::DxgiFormat::BC2_Typeless               => vk::Format::BC2_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC2_UNorm                  => vk::Format::BC2_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC2_UNorm_sRGB             => vk::Format::BC2_SRGB_BLOCK,
        ddsfile::DxgiFormat::BC3_Typeless               => vk::Format::BC3_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC3_UNorm                  => vk::Format::BC3_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC3_UNorm_sRGB             => vk::Format::BC3_SRGB_BLOCK,
        ddsfile::DxgiFormat::BC4_Typeless               => vk::Format::BC4_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC4_UNorm                  => vk::Format::BC4_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC4_SNorm                  => vk::Format::BC4_SNORM_BLOCK,
        ddsfile::DxgiFormat::BC5_Typeless               => vk::Format::BC5_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC5_UNorm                  => vk::Format::BC5_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC5_SNorm                  => vk::Format::BC5_SNORM_BLOCK,
        ddsfile::DxgiFormat::B5G6R5_UNorm               => vk::Format::B5G6R5_UNORM_PACK16,
        ddsfile::DxgiFormat::B5G5R5A1_UNorm             => vk::Format::B5G5R5A1_UNORM_PACK16,
        ddsfile::DxgiFormat::B8G8R8A8_UNorm             => vk::Format::B8G8R8A8_UNORM,
        ddsfile::DxgiFormat::B8G8R8A8_Typeless          => vk::Format::B8G8R8A8_SINT,
        ddsfile::DxgiFormat::B8G8R8A8_UNorm_sRGB        => vk::Format::B8G8R8A8_SRGB,
        ddsfile::DxgiFormat::BC6H_Typeless              => vk::Format::BC6H_SFLOAT_BLOCK,
        ddsfile::DxgiFormat::BC6H_UF16                  => vk::Format::BC6H_UFLOAT_BLOCK,
        ddsfile::DxgiFormat::BC6H_SF16                  => vk::Format::BC6H_SFLOAT_BLOCK,
        ddsfile::DxgiFormat::BC7_Typeless               => vk::Format::BC7_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC7_UNorm                  => vk::Format::BC7_UNORM_BLOCK,
        ddsfile::DxgiFormat::BC7_UNorm_sRGB             => vk::Format::BC7_SRGB_BLOCK,
        _ => return None,
    };
    Some(vk_format)
}

pub fn mip_levels_from_size(max_size: u32) -> u32 {
    u32::max(1, f32::floor(f32::log2(max_size as f32)) as u32 + 1) 
}

pub fn next_mip_size(prev: u32) -> u32 {
    if prev > 1 { prev / 2 } else { 1 }
}

// taken from https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-file-layout-for-textures
fn tex_level_size(width: usize, height: usize, min_mip_size: usize) -> usize {
    usize::max(1, (width + 3) / 4 ) * usize::max(1, (height + 3) / 4 ) * min_mip_size
}

pub fn upload_dds_image(context: &render::Context, name: Cow<'static, str>, bin: &[u8]) -> render::Image {
    let dds = ddsfile::Dds::read(bin).unwrap();

    let dxgi_format = dds.get_dxgi_format().unwrap();
    let format = dxgi_format_to_vk(dxgi_format).unwrap();
    let width = dds.get_width();
    let height = dds.get_height();
    let mip_levels = dds.get_num_mipmap_levels().max(1);
    assert!(mip_levels >= 1);

    let image = context.create_image(name, &render::ImageDesc {
        ty: render::ImageType::Single2D,
        format,
        dimensions: [width, height, 1],
        mip_levels,
        samples: render::MultisampleCount::None,
        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        aspect: vk::ImageAspectFlags::COLOR,
    });

    let data = dds.get_data(0).unwrap();

    let scratch_buffer = context.create_buffer_init("scratch_buffer", &render::BufferDesc {
        size: data.len(),
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        memory_location: MemoryLocation::CpuToGpu,
    }, data);

    let min_mip_size = dds.get_min_mipmap_size_in_bytes();

    context.record_and_submit(|cmd| {
        use render::AccessKind;
        cmd.barrier(&[], &[render::image_barrier(&image, AccessKind::None, AccessKind::TransferWrite)], &[]);

        let mut buffer_offset = 0;
        let mut dimensions = [width, height];

        for mip_level in 0..mip_levels {
            let [width, height] = dimensions;

            cmd.copy_buffer_to_image(&scratch_buffer, &image, &[vk::BufferImageCopy {
                buffer_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: image.subresource_layers(mip_level, 0..1),
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width, height, depth: 1 },
            }]);

            buffer_offset += tex_level_size(width as usize, height as usize, min_mip_size as usize) as u64;
            dimensions = dimensions.map(next_mip_size);
        }


        cmd.barrier(&[], &[render::image_barrier(&image, AccessKind::TransferWrite, AccessKind::AllGraphicsRead)], &[]);
    });

    context.destroy_buffer(&scratch_buffer);

    image
}

pub fn load_image_data(path: &str) -> image::ImageResult<(Vec<u8>, image::ImageFormat)> {
    let data = std::fs::read(&path)?;
    let format = image::ImageFormat::from_path(&path)?;
    Ok((data, format))
}

fn image_data_from_gltf<'a>(
    base_path: &Path,
    source: gltf::image::Source,
    buffers: &'a [Vec<u8>],
) -> image::ImageResult<(Cow<'a, [u8]>, image::ImageFormat)> {
    match source {
        gltf::image::Source::View { view, mime_type } => {
            let buffer = &buffers[view.buffer().index()];
            let data = &buffer[view.offset()..view.offset() + view.length()];

            let format = image::ImageFormat::from_mime_type(mime_type)
                .unwrap_or_else(|| image::guess_format(data).unwrap());

            Ok((data.into(), format))
        }
        gltf::image::Source::Uri { uri, .. } => {
            let source_path = Path::new(uri);
            let source_path = if source_path.is_relative() {
                base_path.join(source_path)
            } else {
                source_path.to_path_buf()
            };
            let data = std::fs::read(&source_path)?;
            let format = image::ImageFormat::from_path(&source_path)?;

            Ok((data.into(), format))
        }
    }
}

pub fn upload_image_and_generate_mipmaps(
    context: &render::Context,
    name: Cow<'static, str>,
    image_data: image::DynamicImage,
    srgb: bool,
    hdr: bool,
) -> render::Image {
    let (width, height) = (image_data.width(), image_data.height());
    let max_size = u32::max(width, height);
    let mip_levels = u32::max(1, f32::floor(f32::log2(max_size as f32)) as u32 + 1);
    
    let image = context.create_image(name, &render::ImageDesc {
        ty: render::ImageType::Single2D,
        format: match (srgb, hdr) {
            (true, false)  => vk::Format::R8G8B8A8_SRGB,
            (false, false) => vk::Format::R8G8B8A8_UNORM,
            (_, true)      => vk::Format::R32G32B32A32_SFLOAT, // the `image` crate doesn't support 16 bit floats
        },
        dimensions: [width, height, 1],
        mip_levels,
        samples: render::MultisampleCount::None,
        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        aspect: vk::ImageAspectFlags::COLOR,
    });

    let scratch_buffer = if hdr {
        let image = image_data.into_rgba32f();
        context.create_buffer_init(
            "scratch_buffer",
            &render::BufferDesc {
                size: image.as_bytes().len(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: MemoryLocation::CpuToGpu,
            },
            image.as_bytes(),
        )
    } else {
        let image = image_data.into_rgba8();
        context.create_buffer_init(
            "scratch_buffer",
            &render::BufferDesc {
                size: image.len(),
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: MemoryLocation::CpuToGpu,
            },
            image.as_bytes(),
        )
    };

    context.record_and_submit(|cmd| {
        use render::AccessKind;

        cmd.barrier(&[], &[render::image_barrier(&image, AccessKind::None, AccessKind::TransferWrite)], &[]);
        cmd.copy_buffer_to_image(
            &scratch_buffer,
            &image,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: image.subresource_layers(0, 0..1),
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: image.width(),
                    height: image.height(),
                    depth: 1,
                },
            }],
        );

        if mip_levels > 1 {
            cmd.generate_mipmaps(&image, 0..1, AccessKind::TransferWrite, AccessKind::AllGraphicsRead);
        } else {
            cmd.barrier(&[], &[render::image_barrier(
                &image,
                AccessKind::TransferWrite,
                AccessKind::AllGraphicsRead,
            )], &[]);
        }
    });

    context.destroy_buffer(&scratch_buffer);

    image
}

pub fn load_image(
    context: &render::Context,
    name: Cow<'static, str>,
    image_binary: &[u8],
    image_format: image::ImageFormat,
    srgb: bool,
    hdr: bool,
) -> render::Image {
    match image_format {
        image::ImageFormat::Dds => {
            upload_dds_image(context, name, &image_binary)
        },
        format => {
            // https://github.com/image-rs/image/issues/1936
            let image_data = if image_format == image::ImageFormat::Hdr {
                let hdr_decoder = image::codecs::hdr::HdrDecoder::new(std::io::BufReader::new(image_binary)).unwrap();
                let width = hdr_decoder.metadata().width;
                let height = hdr_decoder.metadata().height;
                let buffer = hdr_decoder.read_image_hdr().ok().unwrap();
                image::DynamicImage::ImageRgb32F(image::ImageBuffer::from_vec(
                    width,
                    height,
                    buffer.into_iter().flat_map(|c| vec![c[0], c[1], c[2]]).collect(),
                ).unwrap())
            } else {
                image::load_from_memory_with_format(&image_binary, format).unwrap()
            };
            upload_image_and_generate_mipmaps(context, name, image_data, srgb, hdr)
        },
    }
}

pub fn load_gltf(
    path: &str,
    context: &render::Context,
    asset_store: &mut GpuAssetStore,
    scene: &mut SceneBuffer,
) -> gltf::Result<()> {
    let path = Path::new(path);
    let base_path = path.parent().unwrap_or(Path::new(""));

    let mut document = gltf::Gltf::open(path)?;

    let mut blob = document.blob.take();
    let mut buffers = Vec::new();
    for buffer in document.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => blob.take().unwrap(),
            gltf::buffer::Source::Uri(source_path) => {
                let source_path = Path::new(source_path);
                let source_path = if source_path.is_relative() {
                    base_path.join(source_path)
                } else {
                    source_path.to_path_buf()
                };
                std::fs::read(source_path).unwrap()
            }
        };
        buffers.push(data);
    }

    let mut texture_lookup_table: HashMap<usize, TextureHandle> = HashMap::new();
    let mut get_texture = |assets: &mut GpuAssetStore, gltf_texture: gltf::Texture, srgb: bool| -> TextureHandle {
        let index = gltf_texture.index();
        if let Some(texture_id) = texture_lookup_table.get(&index) {
            return *texture_id;
        }

        let (image_binary, image_format) = image_data_from_gltf(base_path, gltf_texture.source().source(), &buffers)
            .unwrap();
        let mut texture =
            load_image(context, format!("gltf_image_{index}").into(), &image_binary, image_format, srgb, false);
        
        use gltf::texture::{MagFilter, WrappingMode};
        let mag_filter = gltf_texture.sampler().mag_filter().unwrap_or(MagFilter::Linear);
        let wrapping_mode = gltf_texture.sampler().wrap_s();
        let sampler_flags = match (mag_filter, wrapping_mode) {
            (MagFilter::Nearest, WrappingMode::ClampToEdge      ) => render::SamplerKind::NearestClamp,
            (MagFilter::Nearest, WrappingMode::MirroredRepeat   ) => render::SamplerKind::NearestRepeat,
            (MagFilter::Nearest, WrappingMode::Repeat           ) => render::SamplerKind::NearestRepeat,
            (MagFilter::Linear,  WrappingMode::ClampToEdge      ) => render::SamplerKind::LinearClamp,
            (MagFilter::Linear,  WrappingMode::MirroredRepeat   ) => render::SamplerKind::LinearRepeat,
            (MagFilter::Linear,  WrappingMode::Repeat           ) => render::SamplerKind::LinearRepeat,
        };
        texture.set_sampler_flags(sampler_flags);
            let handle = assets.import_texture(texture);
            texture_lookup_table.insert(index, handle);
            handle
        };

    let mut material_lookup_table = Vec::new();
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        
        let base_texture = pbr.base_color_texture().map(|tex| get_texture(asset_store, tex.texture(), true));
        let normal_texture = material.normal_texture().map(|tex| get_texture(asset_store, tex.texture(), false));
        let metallic_roughness_texture = pbr
            .metallic_roughness_texture()
            .map(|tex| get_texture(asset_store, tex.texture(), false));
        let occulusion_texture = material
            .occlusion_texture()
            .map(|tex| get_texture(asset_store, tex.texture(), false));
        let emissive_texture = material.emissive_texture().map(|tex| get_texture(asset_store, tex.texture(), true));

        let base_color = Vec4::from_array(pbr.base_color_factor());
        let metallic_factor = pbr.metallic_factor();
        let roughness_factor = pbr.roughness_factor();
        let occulusion_factor = material.occlusion_texture().map_or(0.0, |tex| tex.strength());
        let emissive_factor = Vec3::from_array(material.emissive_factor());

        let handle = asset_store.add_material(
            context,
            MaterialData {
                base_color,
                emissive_factor,
                metallic_factor,
                roughness_factor,
                occulusion_factor,
                
                base_texture,
                normal_texture,
                metallic_roughness_texture,
                occulusion_texture,
                emissive_texture,
                
            },
        );
        material_lookup_table.push(handle);
    }

    let mut model_lookup_table = Vec::new();
    let mut mesh_data = MeshData::new();
    let mut submeshes: Vec<Submesh> = Vec::new();
    for mesh in document.meshes() {
        submeshes.clear();
        for (prim_index, primitive) in mesh.primitives().enumerate() {
            mesh_data.clear();

            let material_index = material_lookup_table[primitive.material().index().unwrap()];

            assert_eq!(primitive.mode(), gltf::mesh::Mode::Triangles);
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let name = mesh.name().unwrap_or("unnamed");

            let have_uvs = reader.read_tex_coords(0).is_some();
            let mut have_normals = reader.read_normals().is_some();
            let have_tangents = reader.read_tangents().is_some();

            if !have_uvs {
                log::warn!("model '{name}' primitive {prim_index} has no uv coordinates");
            }
            if !have_normals {
                log::warn!("model '{name}' primitive {prim_index} has no normals");
            }
            if !have_tangents {
                log::warn!("model '{name}' primitive {prim_index} has no tangents");
            }

            let vertices = sep_vertex_merge(
                reader.read_positions().unwrap(),
                reader.read_tex_coords(0).unwrap().into_f32(),
                reader.read_normals().unwrap(),
                reader.read_tangents().into_iter().flatten(),
            );
            let indices = reader.read_indices().unwrap().into_u32();

            mesh_data.vertices.extend(vertices);
            mesh_data.indices.extend(indices);

            if !have_normals {
                log::info!("generating normals for model '{name}' primitive {prim_index}...");
                mesh_data.compute_normals();
                have_normals = true;
            }

            if !have_tangents {
                if have_normals && have_uvs {
                    log::info!("generating tangents for model '{name}' primitive {prim_index}...");
                    mesh_data.compute_tangents();
                } else {
                    log::warn!("can't generate tangents for '{name}' primitive {prim_index}, becouse it hase no uv coordinates");
                }
            }

            let bounding_box = primitive.bounding_box();
            mesh_data.aabb = Aabb::from_arrays(bounding_box.min, bounding_box.max);

            let mut sphere_radius_sqr: f32 = 0.0;
            for vertex in mesh_data.vertices.iter() {
                let position = Vec3A::from(vertex.position);
                sphere_radius_sqr = sphere_radius_sqr.max(position.length_squared())
            }
            mesh_data.sphere_radius = sphere_radius_sqr.sqrt();
            
            let mesh_handle = asset_store.add_mesh(context, &mesh_data);
            submeshes.push(Submesh {
                mesh_handle,
                material_index,
            });
        }
        let model_handle = asset_store.add_model(submeshes.as_slice());
        model_lookup_table.push(model_handle);
    }

    fn add_gltf_node(
        scene: &mut SceneBuffer,
        model_lookup_table: &[ModelHandle],
        node: gltf::Node,
        parent: Option<&Mat4>,
    ) {
        let mut transform_matrix = Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(parent) = parent {
            transform_matrix = parent.mul_mat4(&transform_matrix);
        }
        let transform = Transform::from_mat4(transform_matrix);

        let model = node.mesh().map(|gltf_mesh| model_lookup_table[gltf_mesh.index()]);
        let name = node.name().map(|str| str.to_owned());

        scene.add_entity(EntityData { name, transform, model });

        for child in node.children() {
            add_gltf_node(scene, model_lookup_table, child, Some(&transform_matrix));
        }
    }

    for node in document.scenes().next().unwrap().nodes() {
        add_gltf_node(scene, &model_lookup_table, node, None);
    }

    Ok(())
}
