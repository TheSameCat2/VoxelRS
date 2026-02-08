/// Multithreaded terrain generation implementation.
///
/// This module provides parallel terrain generation capabilities using Rayon
/// to significantly improve performance of terrain generation operations.
/// It splits terrain into independent regions that can be processed in parallel.
use crate::terrain::TerrainConfig;
use crate::voxel::{Voxel, VoxelPosition, VoxelType, VoxelWorld, CHUNK_SIZE};
use bevy::prelude::Color;
use bevy::prelude::*;
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;
use tracing::{debug, instrument};

/// Represents a region of terrain to be generated.
///
/// This is used to split terrain generation into smaller chunks
/// that can be processed in parallel.
#[derive(Debug, Clone)]
pub struct TerrainRegion {
    /// Minimum X coordinate (inclusive)
    pub min_x: i32,
    /// Maximum X coordinate (exclusive)
    pub max_x: i32,
    /// Minimum Z coordinate (inclusive)
    pub min_z: i32,
    /// Maximum Z coordinate (exclusive)
    pub max_z: i32,
    /// Maximum Y coordinate (exclusive)
    pub max_y: i32,
}

impl TerrainRegion {
    /// Creates a new terrain region.
    pub fn new(min_x: i32, max_x: i32, min_z: i32, max_z: i32, max_y: i32) -> Self {
        Self {
            min_x,
            max_x,
            min_z,
            max_z,
            max_y,
        }
    }

    /// Gets the width of this region.
    pub fn width(&self) -> i32 {
        self.max_x - self.min_x
    }

    /// Gets the depth of this region.
    pub fn depth(&self) -> i32 {
        self.max_z - self.min_z
    }

    /// Gets the height of this region.
    pub fn height(&self, min_surface_height: i32) -> i32 {
        self.max_y - (min_surface_height - 10)
    }

    /// Splits this region into smaller sub-regions for parallel processing.
    ///
    /// # Arguments
    ///
    /// * `num_regions` - Number of regions to split into
    ///
    /// # Returns
    ///
    /// A vector of sub-regions
    pub fn split_into_regions(
        &self,
        num_regions: usize,
        _min_surface_height: i32,
    ) -> Vec<TerrainRegion> {
        let mut regions = Vec::with_capacity(num_regions);

        // Calculate optimal grid dimensions
        let cols = (num_regions as f32).sqrt().ceil() as usize;
        let rows = (num_regions as f32 / cols as f32).ceil() as usize;

        let region_width = self.width() / cols as i32;
        let region_depth = self.depth() / rows as i32;

        for row in 0..rows {
            for col in 0..cols {
                if regions.len() >= num_regions {
                    break;
                }

                let min_x = self.min_x + col as i32 * region_width;
                let max_x = if col == cols - 1 {
                    self.max_x
                } else {
                    min_x + region_width
                };

                let min_z = self.min_z + row as i32 * region_depth;
                let max_z = if row == rows - 1 {
                    self.max_z
                } else {
                    min_z + region_depth
                };

                regions.push(TerrainRegion::new(min_x, max_x, min_z, max_z, self.max_y));
            }
        }

        regions
    }
}

/// Thread-safe terrain generator for parallel processing.
///
/// This struct contains the actual terrain generation logic
/// that can be safely called from multiple threads.
#[derive(Clone, Debug)]
pub struct ParallelTerrainGenerator {
    /// Perlin noise generator for height (thread-safe clone)
    height_noise: Perlin,
    /// Perlin noise generator for caves (thread-safe clone)
    cave_noise: Perlin,
    /// Perlin noise generator for ores (thread-safe clone)
    ore_noise: Perlin,
    /// Configuration for terrain generation
    config: TerrainConfig,
}

impl ParallelTerrainGenerator {
    /// Creates a new parallel terrain generator.
    pub fn new(config: TerrainConfig) -> Self {
        let height_noise = Perlin::new(config.seed);
        let cave_noise = Perlin::new(config.seed.wrapping_add(1));
        let ore_noise = Perlin::new(config.seed.wrapping_add(2));

        Self {
            height_noise,
            cave_noise,
            ore_noise,
            config,
        }
    }

    /// Generates terrain for a single region.
    ///
    /// This is the core terrain generation logic that operates
    /// on a single region and can be called from any thread.
    ///
    /// # Arguments
    ///
    /// * `world` - Thread-safe voxel world
    /// * `region` - The region to generate
    ///
    /// # Returns
    ///
    /// Metrics about the generated terrain
    #[instrument(skip(world, self), fields(
        min_x = region.min_x,
        max_x = region.max_x,
        min_z = region.min_z,
        max_z = region.max_z,
        max_y = region.max_y,
        region_width = region.width(),
        region_depth = region.depth()
    ))]
    pub fn generate_region(
        &self,
        world: &mut VoxelWorld,
        region: &TerrainRegion,
    ) -> TerrainGenerationMetrics {
        let _profiler = crate::profiling::Profiler::new("parallel_terrain_region");
        let mut voxels_generated = 0;
        let mut caves_generated = 0;
        let mut trees_generated = 0;

        let chunk_size = CHUNK_SIZE as i32;
        let min_cx = region.min_x.div_euclid(chunk_size);
        let max_cx = (region.max_x - 1).div_euclid(chunk_size);
        let min_cz = region.min_z.div_euclid(chunk_size);
        let max_cz = (region.max_z - 1).div_euclid(chunk_size);
        let min_cy = (self.config.min_surface_height - 10).div_euclid(chunk_size);
        let max_cy = (region.max_y - 1).div_euclid(chunk_size);

        for cx in min_cx..=max_cx {
            for cz in min_cz..=max_cz {
                // Pre-calculate heights for this chunk column
                let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
                for lz in 0..CHUNK_SIZE {
                    let z = cz * chunk_size + lz as i32;
                    if z < region.min_z || z >= region.max_z {
                        continue;
                    }
                    for lx in 0..CHUNK_SIZE {
                        let x = cx * chunk_size + lx as i32;
                        if x < region.min_x || x >= region.max_x {
                            continue;
                        }
                        heights[lz * CHUNK_SIZE + lx] = self.get_height(x, z);
                    }
                }

                for cy in min_cy..=max_cy {
                    let chunk_pos = crate::voxel::ChunkPosition::new(cx, cy, cz);
                    let chunk = world.get_chunk_mut(&chunk_pos);

                    for lz in 0..CHUNK_SIZE {
                        let z = cz * chunk_size + lz as i32;
                        if z < region.min_z || z >= region.max_z {
                            continue;
                        }
                        for lx in 0..CHUNK_SIZE {
                            let x = cx * chunk_size + lx as i32;
                            if x < region.min_x || x >= region.max_x {
                                continue;
                            }

                            let height = heights[lz * CHUNK_SIZE + lx];

                            for ly in 0..CHUNK_SIZE {
                                let y = cy * chunk_size + ly as i32;
                                if y >= region.max_y {
                                    continue;
                                }

                                let pos = VoxelPosition::new(lx as i32, ly as i32, lz as i32);

                                if y < height {
                                    // Below ground level
                                    if self.is_cave(x, y, z) {
                                        if y < self.config.water_level {
                                            let voxel = Voxel::new(
                                                VoxelType::Destructible,
                                                Color::srgba(0.2, 0.4, 0.8, 0.7),
                                            );
                                            chunk.set_voxel(pos, voxel);
                                            voxels_generated += 1;
                                        }
                                        caves_generated += 1;
                                        continue;
                                    }

                                    let voxel = if y < height - 3 {
                                        if y < 0 {
                                            self.generate_stone()
                                        } else {
                                            self.generate_dirt()
                                        }
                                    } else if y < height - 1 {
                                        self.generate_dirt()
                                    } else if y == height - 1 {
                                        if height <= self.config.water_level + 2 {
                                            Voxel::new(VoxelType::Solid, Color::srgb(0.9, 0.8, 0.6))
                                        } else {
                                            self.generate_grass()
                                        }
                                    } else {
                                        Voxel::new(VoxelType::Air, Color::BLACK)
                                    };

                                    chunk.set_voxel(pos, voxel);
                                } else if y < self.config.water_level {
                                    let voxel = Voxel::new(
                                        VoxelType::Destructible,
                                        Color::srgba(0.2, 0.4, 0.8, 0.7),
                                    );
                                    chunk.set_voxel(pos, voxel);
                                    voxels_generated += 1;
                                }
                            }
                        }
                    }
                    chunk.generated = true;
                }

                // Generate trees on surface for this chunk column
                for lz in 0..CHUNK_SIZE {
                    let z = cz * chunk_size + lz as i32;
                    if z < region.min_z || z >= region.max_z {
                        continue;
                    }
                    for lx in 0..CHUNK_SIZE {
                        let x = cx * chunk_size + lx as i32;
                        if x < region.min_x || x >= region.max_x {
                            continue;
                        }
                        let height = heights[lz * CHUNK_SIZE + lx];
                        if height > self.config.tree_level && self.should_generate_tree(x, z) {
                            self.generate_tree_in_region(world, x, height, z);
                            trees_generated += 1;
                        }
                    }
                }
            }
        }

        // Generate ores for this region
        let ore_metrics = self.generate_ores_in_region(world, region);

        TerrainGenerationMetrics {
            voxels_generated,
            caves_generated,
            trees_generated,
            ores_generated: ore_metrics.ores_generated,
        }
    }

    /// Generates ores for a specific region.
    ///
    /// This is separated to allow for potential future parallelization
    /// of ore generation itself.
    ///
    /// # Arguments
    ///
    /// * `world` - Thread-safe voxel world
    /// * `region` - The region to generate ores in
    ///
    /// # Returns
    ///
    /// Metrics about ore generation
    #[instrument(skip(world, self), fields(
        min_x = region.min_x,
        max_x = region.max_x,
        min_y = self.config.min_surface_height - 10,
        max_y = region.max_y,
        min_z = region.min_z,
        max_z = region.max_z,
        volume = (region.max_x - region.min_x) * (region.max_y - (self.config.min_surface_height - 10)) * (region.max_z - region.min_z)
    ))]
    fn generate_ores_in_region(
        &self,
        world: &mut VoxelWorld,
        region: &TerrainRegion,
    ) -> OreGenerationMetrics {
        let _profiler = crate::profiling::Profiler::new("parallel_ore_generation");
        let mut ores_generated = 0;

        let min_y = self.config.min_surface_height - 10;
        let chunk_size = CHUNK_SIZE as i32;
        let min_cx = region.min_x.div_euclid(chunk_size);
        let max_cx = (region.max_x - 1).div_euclid(chunk_size);
        let min_cz = region.min_z.div_euclid(chunk_size);
        let max_cz = (region.max_z - 1).div_euclid(chunk_size);
        let min_cy = min_y.div_euclid(chunk_size);
        let max_cy = (region.max_y - 1).div_euclid(chunk_size);

        for cx in min_cx..=max_cx {
            for cz in min_cz..=max_cz {
                for cy in min_cy..=max_cy {
                    let chunk_pos = crate::voxel::ChunkPosition::new(cx, cy, cz);
                    let chunk = world.get_chunk_mut(&chunk_pos);

                    for lz in 0..CHUNK_SIZE {
                        let z = cz * chunk_size + lz as i32;
                        if z < region.min_z || z >= region.max_z {
                            continue;
                        }
                        for ly in 0..CHUNK_SIZE {
                            let y = cy * chunk_size + ly as i32;
                            if y < min_y || y >= region.max_y {
                                continue;
                            }
                            for lx in 0..CHUNK_SIZE {
                                let x = cx * chunk_size + lx as i32;
                                if x < region.min_x || x >= region.max_x {
                                    continue;
                                }

                                let pos = VoxelPosition::new(lx as i32, ly as i32, lz as i32);

                                // Skip air voxels
                                if chunk.get_voxel(pos).voxel_type == VoxelType::Air {
                                    continue;
                                }

                                // Use noise to determine ore placement
                                let nx = x as f64 * self.config.scale * 5.0;
                                let ny = y as f64 * self.config.scale * 5.0;
                                let nz = z as f64 * self.config.scale * 5.0;

                                let ore_value = self.ore_noise.get([nx, ny, nz]);

                                // Different ores at different depths
                                let ore_voxel = if y < 10 && ore_value > 0.85 {
                                    Voxel::new(VoxelType::Magic, Color::srgb(0.8, 0.2, 0.2))
                                } else if y < 20 && ore_value > 0.8 {
                                    Voxel::new(VoxelType::Magic, Color::srgb(0.2, 0.2, 0.8))
                                } else if y < 30 && ore_value > 0.75 {
                                    Voxel::new(VoxelType::Magic, Color::srgb(0.2, 0.8, 0.2))
                                } else {
                                    continue;
                                };

                                chunk.set_voxel(pos, ore_voxel);
                                ores_generated += 1;
                            }
                        }
                    }
                }
            }
        }

        OreGenerationMetrics { ores_generated }
    }

    /// Gets terrain height at given XZ coordinates.
    fn get_height(&self, x: i32, z: i32) -> i32 {
        let nx = x as f64 * self.config.scale;
        let nz = z as f64 * self.config.scale;

        // Use multiple octaves of noise for more interesting terrain
        let height = self.height_noise.get([nx, nz]) * 10.0
            + self.height_noise.get([nx * 2.0, nz * 2.0]) * 5.0
            + self.height_noise.get([nx * 4.0, nz * 4.0]) * 2.5;

        // Add continent-style variation for larger features
        let continent = self.height_noise.get([nx * 0.1, nz * 0.1]) * 20.0;
        let total_height = height + continent;

        // Normalize to [0, 1] range using a more precise normalization
        // The theoretical range of our noise is approximately [-37.5, 37.5]
        let normalized = (total_height + 37.5) / 75.0;
        let clamped = normalized.clamp(0.0, 1.0);

        // Apply a curve to create more flat areas and steeper mountains
        let curved = if clamped < 0.4 {
            // Flatten low areas (potential water/ocean)
            clamped * 0.5
        } else if clamped > 0.7 {
            // Exaggerate high areas (mountains)
            0.7 + (clamped - 0.7).powf(1.5) * 1.0
        } else {
            // Moderate slope for mid-range
            clamped
        };

        // Map to our desired surface height range
        let range = self.config.max_surface_height - self.config.min_surface_height;
        let surface_height = self.config.min_surface_height + (curved * range as f64) as i32;

        surface_height
    }

    /// Determines if a position should be a cave.
    fn is_cave(&self, x: i32, y: i32, z: i32) -> bool {
        // Don't create caves near surface to ensure walkable terrain
        let surface_height = self.get_height(x, z);
        if y >= surface_height - 3 {
            return false;
        }

        let nx = x as f64 * self.config.scale * 2.0;
        let ny = y as f64 * self.config.scale * 2.0;
        let nz = z as f64 * self.config.scale * 2.0;

        // Use 3D noise to create caves
        let cave_value = self.cave_noise.get([nx, ny, nz]);
        cave_value > 0.5
    }

    /// Determines if a tree should be generated at given position.
    fn should_generate_tree(&self, x: i32, z: i32) -> bool {
        let nx = x as f64 * self.config.scale * 10.0;
        let nz = z as f64 * self.config.scale * 10.0;

        // Use noise to determine tree placement
        let tree_value = self.height_noise.get([nx, nz]);
        tree_value > 0.9
    }

    /// Generates a tree at given position (for parallel use).
    fn generate_tree_in_region(&self, world: &mut VoxelWorld, x: i32, y: i32, z: i32) {
        // Generate trunk
        for i in 0..5 {
            let trunk_pos = VoxelPosition::new(x, y + i, z);
            let trunk_voxel = Voxel::new(
                VoxelType::Destructible,
                Color::srgb(0.5, 0.3, 0.1), // Brown
            );
            world.set_voxel(&trunk_pos, trunk_voxel);
        }

        // Generate leaves (simple cube shape)
        for dx in -2i32..=2 {
            for dy in 3i32..=6 {
                for dz in -2i32..=2 {
                    // Skip the trunk area
                    if dx.abs() <= 1 && dz.abs() <= 1 && dy <= 5 {
                        continue;
                    }

                    let leaf_pos = VoxelPosition::new(x + dx, y + dy, z + dz);
                    let leaf_voxel = Voxel::new(
                        VoxelType::Destructible,
                        Color::srgb(0.1, 0.7, 0.1), // Green
                    );
                    world.set_voxel(&leaf_pos, leaf_voxel);
                }
            }
        }
    }

    /// Generates a grass voxel.
    fn generate_grass(&self) -> Voxel {
        Voxel::new(VoxelType::Solid, Color::srgb(0.1, 0.7, 0.1))
    }

    /// Generates a dirt voxel.
    fn generate_dirt(&self) -> Voxel {
        Voxel::new(VoxelType::Solid, Color::srgb(0.6, 0.4, 0.2))
    }

    /// Generates a stone voxel.
    fn generate_stone(&self) -> Voxel {
        Voxel::new(VoxelType::Solid, Color::srgb(0.5, 0.5, 0.5))
    }
}

/// Metrics for terrain generation operations.
#[derive(Debug, Default)]
pub struct TerrainGenerationMetrics {
    /// Number of voxels generated
    pub voxels_generated: u32,
    /// Number of caves generated
    pub caves_generated: u32,
    /// Number of trees generated
    pub trees_generated: u32,
    /// Number of ores generated
    pub ores_generated: u32,
}

impl TerrainGenerationMetrics {
    /// Creates a new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds another metrics instance to this one.
    pub fn add(&mut self, other: &TerrainGenerationMetrics) {
        self.voxels_generated += other.voxels_generated;
        self.caves_generated += other.caves_generated;
        self.trees_generated += other.trees_generated;
        self.ores_generated += other.ores_generated;
    }
}

/// Metrics for ore generation operations.
#[derive(Debug)]
struct OreGenerationMetrics {
    /// Number of ores generated
    ores_generated: u32,
}

/// Generates terrain for multiple regions in parallel.
///
/// This is the main entry point for parallel terrain generation.
/// It splits the terrain into regions and processes them in parallel.
///
/// # Arguments
///
/// * `region` - The main region to generate
/// * `config` - Terrain configuration
/// * `num_threads` - Number of threads to use (0 = auto)
///
/// # Returns
///
/// Combined world and metrics for all regions
#[instrument(skip(config), fields(
    min_x = region.min_x,
    max_x = region.max_x,
    min_z = region.min_z,
    max_z = region.max_z,
    max_y = region.max_y,
    world_size = region.width(),
    height_range = region.height(config.min_surface_height),
    num_threads
))]
pub fn generate_terrain_parallel(
    region: &TerrainRegion,
    config: &TerrainConfig,
    num_threads: usize,
) -> (VoxelWorld, TerrainGenerationMetrics) {
    let _profiler = crate::profiling::Profiler::new("generate_terrain_parallel");

    // Determine the actual number of threads that will be used
    let actual_num_threads = if num_threads == 0 {
        // For auto-detect, use the number of available CPU cores
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) // Fallback to 4 if detection fails
    } else {
        num_threads
    };

    let generator = ParallelTerrainGenerator::new(config.clone());

    // Split region into sub-regions for parallel processing
    let sub_regions =
        region.split_into_regions(actual_num_threads.max(1), config.min_surface_height);

    debug!(
        num_threads = num_threads,
        actual_num_threads = actual_num_threads,
        sub_regions_count = sub_regions.len(),
        "Split terrain into sub-regions for parallel processing"
    );

    // Process regions in parallel
    let results: Vec<(VoxelWorld, TerrainGenerationMetrics)> = if num_threads == 0 {
        // Use global Rayon thread pool for auto-detect
        sub_regions
            .par_iter()
            .map(|sub_region| {
                let mut local_world = VoxelWorld::new();
                let metrics = generator.generate_region(&mut local_world, sub_region);
                (local_world, metrics)
            })
            .collect()
    } else {
        // Use custom thread pool if a specific number of threads is requested
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

        pool.install(|| {
            sub_regions
                .par_iter()
                .map(|sub_region| {
                    let mut local_world = VoxelWorld::new();
                    let metrics = generator.generate_region(&mut local_world, sub_region);
                    (local_world, metrics)
                })
                .collect()
        })
    };

    // Combine results from all regions
    let mut combined_world = VoxelWorld::new();
    let mut total_metrics = TerrainGenerationMetrics::new();
    for (local_world, metrics) in results {
        combined_world.merge(local_world);
        total_metrics.add(&metrics);
    }

    debug!(
        voxels_generated = total_metrics.voxels_generated,
        caves_generated = total_metrics.caves_generated,
        trees_generated = total_metrics.trees_generated,
        ores_generated = total_metrics.ores_generated,
        "Parallel terrain generation completed"
    );

    (combined_world, total_metrics)
}
