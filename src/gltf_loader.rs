use glam::{Vec3, Quat};

use crate::{render, scene::{SceneBuffer, Transform, EntityData}, assets::{GpuAssetStore, MeshData, sep_vertex_merge, ModelHandle}};


pub fn load_gltf(
    path: &str,
    context: &render::Context,
    asset_store: &mut GpuAssetStore,
    scene: &mut SceneBuffer
) -> gltf::Result<()> {
    let (document, buffers, _) = gltf::import(path)?;

    let mut model_lookup_table = Vec::new();
    let mut mesh_data = MeshData::new();
    let mut submeshes = Vec::new();
    for mesh in document.meshes() {
        submeshes.clear();
        for (prim_index, primitive) in mesh.primitives().enumerate() {
            mesh_data.clear();

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

            let mesh = asset_store.add_mesh(context, &mesh_data);
            submeshes.push(mesh);
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