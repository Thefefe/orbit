use std::{collections::HashMap, path::Path};

use ash::vk;
use glam::{Vec4, Mat4, Vec3};
use gpu_allocator::MemoryLocation;

use crate::{
    assets::{sep_vertex_merge, GpuAssetStore, MaterialData, MeshData, ModelHandle, Submesh, TextureHandle},
    render,
    scene::{EntityData, SceneBuffer, Transform},
};

fn load_image_data(
    base_path: &Path,
    source: gltf::image::Source,
    buffers: &[Vec<u8>],
) -> Result<image::RgbaImage, image::ImageError> {
    match source {
        gltf::image::Source::View { view, mime_type } => {
            let buffer = &buffers[view.buffer().index()];
            let data = &buffer[view.offset()..view.offset() + view.length()];

            let format = match mime_type {
                "image/png" => image::ImageFormat::Png,
                "image/jpeg" => image::ImageFormat::Jpeg,
                _ => image::guess_format(data)?,
            };

            let image = image::load_from_memory_with_format(data, format)?;

            Ok(image.to_rgba8())
        }
        gltf::image::Source::Uri { uri, .. } => {
            let source_path = Path::new(uri);
            let source_path = if source_path.is_relative() {
                base_path.join(source_path)
            } else {
                source_path.to_path_buf()
            };
            let image = image::open(source_path)?;
            Ok(image.to_rgba8())
        }
    }
}

fn load_texture(
    texture: gltf::Texture,
    base_path: &Path,
    buffers: &[Vec<u8>],
    context: &render::Context,
    srgb: bool,
) -> render::Image {
    let image = texture.source();
    let image_index = image.index();
    let image_data = load_image_data(base_path, image.source(), &buffers).unwrap();

    let (width, height) = image_data.dimensions();
    let max_size = u32::max(width, height);
    let mip_levels = f32::floor(f32::log2(max_size as f32)) as u32 + 1;

    let mut image = context.create_image(
        format!("gltf_image_{image_index}"),
        &render::ImageDesc {
            format: if srgb {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            },
            width,
            height,
            mip_levels,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            aspect: vk::ImageAspectFlags::COLOR,
        },
    );

    let sampler_flags = {
        let mut flags = render::SamplerFlags::empty();

        if texture.sampler().min_filter() == Some(gltf::texture::MinFilter::Nearest) {
            flags |= render::SamplerFlags::NEAREST;
        }

        if texture.sampler().wrap_s() == gltf::texture::WrappingMode::Repeat
            || texture.sampler().wrap_t() == gltf::texture::WrappingMode::Repeat
        {
            flags |= render::SamplerFlags::REPEAT;
        }

        flags
    };
    image.set_sampler_flags(sampler_flags);

    let scratch_buffer = context.create_buffer_init(
        "scratch_buffer",
        &render::BufferDesc {
            size: image_data.len(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &image_data,
    );

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

        let mut mip_width = width;
        let mut mip_height = height;

        for (last_level, current_level) in (1..mip_levels).enumerate() {
            let last_level = last_level as u32;
            let current_level = current_level as u32;

            let new_mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
            let new_mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };

            cmd.barrier(&[], &[render::image_subresource_barrier(
                &image,
                last_level..last_level + 1, 
                ..,
                AccessKind::TransferWrite,
                AccessKind::TransferRead
            )], &[]);

            let blit_region = vk::ImageBlit {
                src_subresource: image.subresource_layers(last_level, ..),
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width as i32,
                        y: mip_height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: image.subresource_layers(current_level, ..),
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: new_mip_width as i32,
                        y: new_mip_height as i32,
                        z: 1,
                    },
                ],
            };

            cmd.blit_image(
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
        
        cmd.barrier(&[], &[render::image_subresource_barrier(
            &image,
            mip_levels - 1..mip_levels, 
            ..,
            AccessKind::TransferWrite,
            AccessKind::AllGraphicsRead,
        )], &[]);

        cmd.barrier(&[], &[render::image_subresource_barrier(
            &image,
            ..mip_levels - 1, 
            ..,
            AccessKind::TransferRead,
            AccessKind::AllGraphicsRead,
        )], &[]);
    });

    context.destroy_buffer(&scratch_buffer);

    image
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
    let mut get_texture = |assets: &mut GpuAssetStore, texture: gltf::Texture, srgb: bool| -> TextureHandle {
        let index = texture.index();
        if let Some(texture_id) = texture_lookup_table.get(&index) {
            return *texture_id;
        }

        let texture = load_texture(texture, base_path, &buffers, context, srgb);
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

            if !have_tangents && have_normals && have_uvs {
                log::info!("generating tangents for model '{name}' primitive {prim_index}...");
                mesh_data.compute_tangents();
            } else {
                log::warn!("can't generate tangents for '{name}' primitive {prim_index}, becouse it hase no uv coordinates");
            }

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
