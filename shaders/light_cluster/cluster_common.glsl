uint cluster_volume_index_from_id(uvec3 cluster_count, uvec3 id) {
    // TODO: optimize data layout
    return id.x + id.y * cluster_count.x + id.z * cluster_count.x * cluster_count.y;
}

uvec3 cluster_id_from_volume_index(uvec3 cluster_count, uint index) {
    uint z = index / (cluster_count.x * cluster_count.y);
    index -= z * cluster_count.x * cluster_count.y;
    uint y = index / cluster_count.x;
    index -= y * cluster_count.x;
    return uvec3(index, y, z);
}

uint tile_id_to_tile_index(uvec2 tile_counts, uvec2 tile_id) {
    return tile_id.x + tile_id.y * tile_counts.x;
}

uint linear_z_to_depth_slice(float z_scale, float z_bias, float z) {
    return uint(log2(z) * z_scale + z_bias);
}

float depth_slice_to_linear_near_z(uint slice_count, float near, float far, uint slice) {
    return near * pow(far / near, slice / float(slice_count));
}

float attenuation(float dist, float intensity, float luminance_cutoff, float outer_radius) {
    float d2 = dist*dist;
    return max((intensity / d2) - luminance_cutoff * d2 / (outer_radius * outer_radius), 0);
}

bool is_cluster_active(uint cluster_mask_buffer, uvec3 cluster_count, uvec3 cluster_id) {
    uint tile_index = tile_id_to_tile_index(cluster_count.xy, cluster_id.xy);
    return bool(GetBuffer(TileDepthSliceMask, cluster_mask_buffer).masks[tile_index] & (1 << cluster_id.z));
}