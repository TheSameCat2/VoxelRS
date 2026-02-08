#[cfg(test)]
mod tests {
    use crate::parallel_terrain::{generate_terrain_parallel, TerrainRegion};
    use crate::terrain::TerrainConfig;
    use crate::voxel::VoxelPosition;
    use std::time::Instant;

    #[test]
    fn test_multithreaded_terrain_generation() {
        // Create terrain region
        let region = TerrainRegion::new(-8, 8, -8, 8, 20);

        // Create config with multithreading enabled
        let config = TerrainConfig {
            enable_multithreading: true,
            num_threads: 36,
            ..Default::default()
        };

        // Test multithreaded generation
        let start = Instant::now();

        let (world, _metrics) = generate_terrain_parallel(&region, &config, config.num_threads);

        let elapsed = start.elapsed();
        println!("Multithreaded terrain generation took: {:?}", elapsed);

        // Verify that terrain was generated
        let min_y = config.min_surface_height - 10;
        let voxel_count = (region.min_x..region.max_x)
            .flat_map(|x| {
                let world_ref = &world;
                (region.min_z..region.max_z).flat_map(move |z| {
                    (min_y..region.max_y)
                        .map(move |y| world_ref.get_voxel(&VoxelPosition::new(x, y, z)))
                })
            })
            .filter(|v| v.is_some())
            .count();

        println!("Generated voxels: {}", voxel_count);
        assert!(voxel_count > 0);
    }
}
