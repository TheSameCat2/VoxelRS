/// Terrain generation implementation.
///
/// This module provides terrain generation functionality for the voxel game.
/// It includes both single-threaded and multithreaded implementations.
use crate::voxel::{Voxel, VoxelPosition, VoxelType, VoxelWorld, CHUNK_SIZE};
use bevy::prelude::*;
use noise::{NoiseFn, Perlin};
use tracing::{debug, instrument};

/// Configuration for terrain generation.
///
/// This struct contains all parameters needed to generate terrain.
/// It can be customized to create different types of landscapes.
#[derive(Resource, Clone, Debug)]
pub struct TerrainConfig {
    /// Seed for noise functions
    pub seed: u32,
    /// Scale of terrain (higher = larger features)
    pub scale: f64,
    /// Height multiplier for terrain
    pub height_multiplier: f32,
    /// Water level (sea level)
    pub water_level: i32,
    /// Snow level
    pub snow_level: i32,
    /// Tree level
    pub tree_level: i32,
    /// Minimum surface height (inclusive)
    pub min_surface_height: i32,
    /// Maximum surface height (inclusive)
    pub max_surface_height: i32,
    /// Enable greedy meshing for performance optimization
    pub enable_greedy_meshing: bool,
    /// Enable multithreaded terrain generation
    pub enable_multithreading: bool,
    /// Number of threads to use for terrain generation (0 = auto)
    pub num_threads: usize,
    /// Distance (in chunks) to generate around the player
    pub generation_distance: i32,
    /// Distance (in chunks) to keep chunks visible
    pub render_distance: i32,
    /// Initial world size (in voxels) to generate at startup
    pub initial_world_size: i32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            scale: 0.05,
            height_multiplier: 10.0,
            water_level: 0,
            snow_level: 30,
            tree_level: 5,
            min_surface_height: -15,
            max_surface_height: 20,
            enable_greedy_meshing: true,
            enable_multithreading: true,
            num_threads: 0, // 0 = auto-detect
            generation_distance: 6,
            render_distance: 5,
            initial_world_size: 128,
        }
    }
}

/// Terrain generator using noise functions.
///
/// This struct contains the noise generators and methods to create
/// different types of terrain for our voxel world.
#[derive(Debug)]
pub struct TerrainGenerator {
    /// Perlin noise generator for height
    height_noise: Perlin,
    /// Perlin noise generator for caves
    cave_noise: Perlin,
    /// Perlin noise generator for ores
    ore_noise: Perlin,
    /// Configuration for terrain generation
    config: TerrainConfig,
}

impl TerrainGenerator {
    /// Creates a new terrain generator with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for terrain generation
    ///
    /// # Returns
    ///
    /// A new TerrainGenerator instance
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

    /// Generates terrain for the given region.
    ///
    /// This function fills a voxel world with terrain for the specified
    /// region. It generates height maps, caves, and ore deposits.
    ///
    /// # Arguments
    ///
    /// * `world` - The voxel world to fill with terrain
    /// * `min_x` - Minimum X coordinate (inclusive)
    /// * `max_x` - Maximum X coordinate (exclusive)
    /// * `min_z` - Minimum Z coordinate (inclusive)
    /// * `max_z` - Maximum Z coordinate (exclusive)
    /// * `max_y` - Maximum Y coordinate (exclusive)
    #[instrument(skip(world), fields(
        min_x, max_x, min_z, max_z, max_y,
        world_size = max_x - min_x,
        height_range = max_y - (self.config.min_surface_height - 10)
    ))]
    pub fn generate_terrain(
        &self,
        world: &mut VoxelWorld,
        min_x: i32,
        max_x: i32,
        min_z: i32,
        max_z: i32,
        max_y: i32,
    ) {
        let _profiler = crate::profiling::Profiler::new("terrain_generate_region");
        let mut voxels_generated = 0;
        let mut caves_generated = 0;
        let mut trees_generated = 0;

        for x in min_x..max_x {
            for z in min_z..max_z {
                // Generate height map
                let height = self.get_height(x, z);

                // Generate terrain column
                for y in (self.config.min_surface_height - 10)..max_y {
                    let pos = VoxelPosition::new(x, y, z);

                    if y < height {
                        // Below ground level
                        if self.is_cave(x, y, z) {
                            // Cave - leave as air (or water below sea level)
                            if y < self.config.water_level {
                                // Water in caves below sea level
                                let voxel = Voxel::new(
                                    VoxelType::Destructible,
                                    Color::srgba(0.2, 0.4, 0.8, 0.7),
                                );
                                world.set_voxel(&pos, voxel);
                                voxels_generated += 1;
                            }
                            caves_generated += 1;
                            continue;
                        }

                        // Generate different materials based on depth
                        let voxel = if y < height - 3 {
                            // Underground - stone/dirt mix
                            if y < 0 {
                                self.generate_stone()
                            } else {
                                self.generate_dirt()
                            }
                        } else if y < height - 1 {
                            // Near surface - dirt
                            self.generate_dirt()
                        } else if y == height - 1 {
                            // Surface - grass or sand
                            if height <= self.config.water_level + 2 {
                                // Beach areas near water
                                Voxel::new(VoxelType::Solid, Color::srgb(0.9, 0.8, 0.6))
                            } else {
                                self.generate_grass()
                            }
                        } else {
                            // Above ground - air
                            Voxel::new(VoxelType::Air, Color::BLACK)
                        };

                        world.set_voxel(&pos, voxel);
                    } else if y < self.config.water_level {
                        // Below water level - fill with water (but not above terrain)
                        let voxel =
                            Voxel::new(VoxelType::Destructible, Color::srgba(0.2, 0.4, 0.8, 0.7));
                        world.set_voxel(&pos, voxel);
                        voxels_generated += 1;
                    }
                }

                // Generate trees on surface
                if height > self.config.tree_level && self.should_generate_tree(x, z) {
                    self.generate_tree(world, x, height, z);
                    trees_generated += 1;
                }
            }
        }

        // Generate ores
        self.generate_ores(
            world,
            min_x,
            max_x,
            self.config.min_surface_height - 10,
            max_y,
            min_z,
            max_z,
        );

        // Log terrain generation metrics
        debug!(
            voxels_generated = voxels_generated,
            caves_generated = caves_generated,
            trees_generated = trees_generated,
            "Terrain generation metrics"
        );

        // Mark all chunks in this region as generated
        let chunk_size = CHUNK_SIZE as i32;
        let min_cx = min_x.div_euclid(chunk_size);
        let max_cx = (max_x - 1).div_euclid(chunk_size);
        let min_cz = min_z.div_euclid(chunk_size);
        let max_cz = (max_z - 1).div_euclid(chunk_size);
        let min_cy = (self.config.min_surface_height - 10).div_euclid(chunk_size);
        let max_cy = (max_y - 1).div_euclid(chunk_size);

        for cx in min_cx..=max_cx {
            for cz in min_cz..=max_cz {
                for cy in min_cy..=max_cy {
                    let chunk_pos = crate::voxel::ChunkPosition::new(cx, cy, cz);
                    let chunk = world.get_chunk_mut(&chunk_pos);
                    chunk.generated = true;
                }
            }
        }
    }

    /// Gets terrain height at the given XZ coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate
    /// * `z` - Z coordinate
    ///
    /// # Returns
    ///
    /// The height (Y coordinate) of terrain at this position
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
    ///
    /// Uses 3D noise to create cave systems.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    /// * `z` - Z coordinate
    ///
    /// # Returns
    ///
    /// True if this position should be a cave (air), false otherwise
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

    /// Determines if a tree should be generated at the given position.
    fn should_generate_tree(&self, x: i32, z: i32) -> bool {
        let nx = x as f64 * self.config.scale * 10.0;
        let nz = z as f64 * self.config.scale * 10.0;

        // Use noise to determine tree placement
        let tree_value = self.height_noise.get([nx, nz]);
        tree_value > 0.7
    }

    /// Generates a tree at the given position.
    ///
    /// Creates a simple tree with a trunk and leaves.
    ///
    /// # Arguments
    ///
    /// * `world` - The voxel world
    /// * `x` - X coordinate of the tree base
    /// * `y` - Y coordinate of the tree base
    /// * `z` - Z coordinate of the tree base
    #[instrument(skip(world), fields(x, y, z))]
    fn generate_tree(&self, world: &mut VoxelWorld, x: i32, y: i32, z: i32) {
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

    /// Generates ore deposits in the world.
    ///
    /// Creates various types of ore deposits at different depths.
    ///
    /// # Arguments
    ///
    /// * `world` - The voxel world
    /// * `min_x` - Minimum X coordinate
    /// * `max_x` - Maximum X coordinate
    /// * `min_y` - Minimum Y coordinate
    /// * `max_y` - Maximum Y coordinate
    /// * `min_z` - Minimum Z coordinate
    /// * `max_z` - Maximum Z coordinate
    #[instrument(skip(world), fields(
        min_x, max_x, min_y, max_y, min_z, max_z,
        volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    ))]
    fn generate_ores(
        &self,
        world: &mut VoxelWorld,
        min_x: i32,
        max_x: i32,
        min_y: i32,
        max_y: i32,
        min_z: i32,
        max_z: i32,
    ) {
        for x in min_x..max_x {
            for y in min_y..max_y {
                for z in min_z..max_z {
                    let pos = VoxelPosition::new(x, y, z);

                    // Skip air voxels
                    if let Some(voxel) = world.get_voxel(&pos) {
                        if voxel.voxel_type == VoxelType::Air {
                            continue;
                        }
                    }

                    // Use noise to determine ore placement
                    let nx = x as f64 * self.config.scale * 5.0;
                    let ny = y as f64 * self.config.scale * 5.0;
                    let nz = z as f64 * self.config.scale * 5.0;

                    let ore_value = self.ore_noise.get([nx, ny, nz]);

                    // Different ores at different depths
                    let ore_voxel = if y < 10 && ore_value > 0.85 {
                        // Deep ore - red
                        Voxel::new(VoxelType::Magic, Color::srgb(0.8, 0.2, 0.2))
                    } else if y < 20 && ore_value > 0.8 {
                        // Mid-depth ore - blue
                        Voxel::new(VoxelType::Magic, Color::srgb(0.2, 0.2, 0.8))
                    } else if y < 30 && ore_value > 0.75 {
                        // Shallow ore - green
                        Voxel::new(VoxelType::Magic, Color::srgb(0.2, 0.8, 0.2))
                    } else {
                        continue;
                    };

                    world.set_voxel(&pos, ore_voxel);
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

/// Bevy system to initialize terrain generation.
///
/// This system is responsible for:
/// 1. Creating a terrain generator with the current configuration
/// 2. Generating initial terrain around the origin
/// 3. Setting up the world for gameplay
///
/// # Arguments
///
/// * `world_resource` - Our voxel world resource
/// * `config` - Terrain configuration
#[instrument(skip(commands, world_resource, config))]
pub fn initialize_terrain(
    world_resource: Res<VoxelWorld>,
    config: Res<TerrainConfig>,
    mut commands: Commands,
) {
    let _profiler = crate::profiling::Profiler::new("initialize_terrain");

    // Generate terrain in a specified area around the origin
    let world_size = config.initial_world_size;
    // Generate terrain covering our full height range plus some buffer
    let world_height = config.max_surface_height + 10;

    // Create terrain region
    let region = crate::parallel_terrain::TerrainRegion::new(
        -world_size / 2,
        world_size / 2,
        -world_size / 2,
        world_size / 2,
        world_height,
    );

    // Use multithreading if enabled
    if config.enable_multithreading {
        debug!("Using multithreaded terrain generation");

        // Generate terrain in parallel
        let (new_world, _metrics) = crate::parallel_terrain::generate_terrain_parallel(
            &region,
            &config,
            config.num_threads,
        );

        // Merge the new chunks into our world
        let mut world = world_resource.clone();
        world.merge(new_world);
        commands.insert_resource(world);
    } else {
        debug!("Using single-threaded terrain generation");

        // Create a mutable copy of the world for single-threaded generation
        let mut world_copy = world_resource.clone();

        // Create terrain generator
        let generator = TerrainGenerator::new(config.clone());

        // Generate terrain normally
        generator.generate_terrain(
            &mut world_copy,
            region.min_x,
            region.max_x,
            region.min_z,
            region.max_z,
            region.max_y,
        );

        // Store the world back as a resource
        commands.insert_resource(world_copy);
    }
}

/// Bevy system to generate terrain on demand.
///
/// This system generates new terrain chunks as needed when the player
/// explores the world. It checks which chunks are nearby and generates
/// terrain for any that don't exist yet.
///
/// # Arguments
///
/// * `world_resource` - Our voxel world resource
/// * `config` - Terrain configuration
/// * `player_query` - Query to find player position
pub fn generate_terrain_on_demand(
    mut world_resource: ResMut<VoxelWorld>,
    config: Res<TerrainConfig>,
    player_query: Query<&Transform, With<PlayerController>>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        // Get player position in voxel coordinates
        let player_pos = player_transform.translation;
        let player_voxel_x = player_pos.x as i32;
        let player_voxel_z = player_pos.z as i32;

        // Generate terrain in chunks around the player
        let chunk_radius = config.generation_distance;
        let chunk_size = CHUNK_SIZE as i32;

        let player_chunk_x = player_voxel_x.div_euclid(chunk_size);
        let player_chunk_z = player_voxel_z.div_euclid(chunk_size);

        for dx in -chunk_radius..=chunk_radius {
            for dz in -chunk_radius..=chunk_radius {
                // Calculate chunk coordinates
                let cx = player_chunk_x + dx;
                let cz = player_chunk_z + dz;

                // Calculate chunk boundaries
                let min_x = cx * chunk_size;
                let max_x = min_x + chunk_size;
                let min_z = cz * chunk_size;
                let max_z = min_z + chunk_size;

                // Check if this chunk already has terrain
                // Use a more reliable check by looking for a chunk at a representative height
                use crate::voxel::ChunkPosition;
                let chunk_pos = ChunkPosition::new(
                    cx,
                    (config.min_surface_height - 10).div_euclid(chunk_size),
                    cz,
                );

                let is_generated = world_resource
                    .get_chunk(&chunk_pos)
                    .map(|c| c.generated)
                    .unwrap_or(false);

                if !is_generated {
                    // Create terrain region
                    let region = crate::parallel_terrain::TerrainRegion::new(
                        min_x,
                        max_x,
                        min_z,
                        max_z,
                        config.max_surface_height + 10,
                    );

                    // Use multithreaded generation if enabled
                    if config.enable_multithreading {
                        let (new_world, _metrics) =
                            crate::parallel_terrain::generate_terrain_parallel(
                                &region,
                                &config,
                                config.num_threads,
                            );
                        world_resource.merge(new_world);
                    } else {
                        let generator = TerrainGenerator::new(config.clone());
                        generator.generate_terrain(
                            &mut world_resource,
                            region.min_x,
                            region.max_x,
                            region.min_z,
                            region.max_z,
                            region.max_y,
                        );
                    }
                }
            }
        }
    }
}

/// Bevy system to manage chunk visibility based on distance to player.
///
/// This system hides chunks that are further than the configured render distance.
///
/// # Arguments
///
/// * `config` - Terrain configuration
/// * `player_query` - Query to find player position
/// * `chunk_query` - Query to find all chunk entities
pub fn manage_chunk_visibility(
    config: Res<TerrainConfig>,
    player_query: Query<&Transform, With<PlayerController>>,
    mut chunk_query: Query<(&mut Visibility, &crate::voxel_mesh::ChunkComponent)>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        let player_pos = player_transform.translation;
        let chunk_size = CHUNK_SIZE as f32;

        // Use a slightly larger radius for the actual check to avoid flickering at the edge
        let render_distance = config.render_distance as f32 * chunk_size;
        let render_distance_sq = render_distance * render_distance;

        for (mut visibility, chunk_comp) in chunk_query.iter_mut() {
            let chunk_pos = chunk_comp.0;
            // Calculate center of chunk in world coordinates (horizontal only)
            let chunk_center_x = (chunk_pos.x * CHUNK_SIZE as i32) as f32 + chunk_size / 2.0;
            let chunk_center_z = (chunk_pos.z * CHUNK_SIZE as i32) as f32 + chunk_size / 2.0;

            let dx = player_pos.x - chunk_center_x;
            let dz = player_pos.z - chunk_center_z;
            let distance_sq = dx * dx + dz * dz;

            if distance_sq > render_distance_sq {
                if *visibility != Visibility::Hidden {
                    *visibility = Visibility::Hidden;
                }
            } else {
                if *visibility != Visibility::Inherited {
                    *visibility = Visibility::Inherited;
                }
            }
        }
    }
}

/// Component to identify player entities.
///
/// This component is added to player entities so we can find them
/// for terrain generation and other systems.
#[derive(Component)]
pub struct PlayerController;
