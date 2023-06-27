use std::path::Path;

use ash::vk;
use glam::{Vec3, Quat, Vec4};

use crate::{render, scene::{SceneBuffer, Transform, EntityData}, assets::{GpuAssetStore, MeshData, sep_vertex_merge, ModelHandle, Submesh, MaterialData}};

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
        },
        gltf::image::Source::Uri { uri, .. } => {
            let source_path = Path::new(uri);
            let source_path = if source_path.is_relative() {
                base_path.join(source_path)
            } else {
                source_path.to_path_buf()
            };
            let image = image::open(source_path)?;
            Ok(image.to_rgba8())
        },
    }
}

pub fn load_gltf(
    path: &str,
    context: &render::Context,
    asset_store: &mut GpuAssetStore,
    scene: &mut SceneBuffer
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
            },
        };
        buffers.push(data);
    }

    let mut texture_lookup_table = Vec::new();
    for texture in document.textures() {
        let image = texture.source();
        let image_index = image.index();
        let image_data = load_image_data(base_path, image.source(), &buffers).unwrap();

        let mut image = context.create_image(&format!("gltf_image_{image_index}"), &render::ImageDesc {
            format: vk::Format::R8G8B8A8_UNORM,
            width: image_data.width(),
            height: image_data.height(),
            mip_levels: 1,
            samples: render::MultisampleCount::None,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
        });

        let sampler_flags = {
            let mut flags = render::SamplerFlags::empty();
            
            if texture.sampler().min_filter() == Some(gltf::texture::MinFilter::Nearest) {
                flags |= render::SamplerFlags::NEAREST;
            }

            if texture.sampler().wrap_s() == gltf::texture::WrappingMode::Repeat || 
               texture.sampler().wrap_t() == gltf::texture::WrappingMode::Repeat {
                flags |= render::SamplerFlags::REPEAT;
            }

            flags
        };
        image.set_sampler_flags(sampler_flags);

        context.immediate_write_image(
            &image,
            0,
            0..1,
            render::AccessKind::None,
            Some(render::AccessKind::FragmentShaderRead),
            &image_data,
            None
        );

        let handle = asset_store.import_texture(image);
        texture_lookup_table.push(handle);
    }

    let mut material_lookup_table = Vec::new();
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let color = pbr.base_color_factor();
        let base_texture = pbr.base_color_texture().map(|tex| texture_lookup_table[tex.texture().index()]);
        
        let handle = asset_store.add_material(context, MaterialData {
            base_color: Vec4::from_array(color),
            base_texture,
            normal_texture: None,
        });
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

            if reader.read_tex_coords(0).is_none() {
                log::warn!("model '{name}' primitive {prim_index} has no uv coordinates");
            }
            if reader.read_normals().is_none() {
                log::warn!("model '{name}' primitive {prim_index} has no normals");
            }
            if reader.read_tangents().is_none() {
                log::warn!("model '{name}' primitive {prim_index} has no tangents");
            }

            let vertices = sep_vertex_merge(
                reader.read_positions().unwrap(),
                reader.read_tex_coords(0).unwrap().into_f32(),
                reader.read_normals().unwrap(),
                reader.read_tangents().into_iter().flatten()
            );
            let indices = reader.read_indices().unwrap().into_u32();
            
            mesh_data.vertices.extend(vertices);
            mesh_data.indices.extend(indices);

            let mesh_handle = asset_store.add_mesh(context, &mesh_data);
            submeshes.push(Submesh { mesh_handle, material_index });
        }
        let model_handle = asset_store.add_model(submeshes.as_slice());
        model_lookup_table.push(model_handle);
    }

    fn add_gltf_node(
        scene: &mut SceneBuffer,
        model_lookup_table: &[ModelHandle],
        node: gltf::Node,
        parent: Option<Transform>
    ) {
        let (position, orientation, scale) = node.transform().decomposed();
        let mut transform = Transform {
            position: Vec3::from_array(position),
            orientation: Quat::from_array(orientation),
            scale: Vec3::from_array(scale),
        };

        if let Some(parent) = parent {
            parent.transform(&mut transform);
        }

        let model = node.mesh().map(|gltf_mesh| model_lookup_table[gltf_mesh.index()]);
        scene.add_entity(EntityData {
            transform,
            model,
        });

        for child in node.children() {
            add_gltf_node(scene, model_lookup_table, child, Some(transform));
        }
    }

    for node in document.nodes() {
        add_gltf_node(scene, &model_lookup_table, node, None);       
    }

    Ok(())
}