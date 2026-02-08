/// A modular voxel system designed for team deathmatch games.
///
/// This module provides the core components and systems for:
/// - Voxel data representation
/// - Chunk-based world storage
/// - Efficient mesh generation with face culling
/// - Terrain generation using noise functions
/// - Integration with Bevy's ECS system
///
/// The design focuses on performance for real-time editing during gameplay,
/// which is crucial for a voxel-based TDM where the environment can change
/// dynamically during matches.
use bevy::prelude::*;
use tracing::{debug, instrument};

/// A single voxel in the world.
///
/// Voxels are the basic building blocks of our voxel-based game.
/// Each voxel has a type (air, solid, different materials) and a color.
/// For the TDM game, we can extend this with more properties like
/// durability, special effects, etc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voxel {
    /// Type of the voxel (air, solid, special blocks)
    pub voxel_type: VoxelType,
    /// Color of the voxel
    pub color: Color,
}

/// Types of voxels in our game.
///
/// This defines the different kinds of blocks that can exist in the world.
/// We'll expand this as we add more game mechanics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoxelType {
    /// Air - no block, can be passed through
    Air,
    /// Basic solid block
    Solid,
    /// Destructible block (lower health)
    Destructible,
    /// Magic block (for magic-based gameplay elements)
    Magic,
}

impl Voxel {
    /// Creates a new voxel with the given type and color.
    ///
    /// # Arguments
    ///
    /// * `voxel_type` - The type of voxel to create
    /// * `color` - The color of the voxel
    ///
    /// # Returns
    ///
    /// A new Voxel instance
    pub fn new(voxel_type: VoxelType, color: Color) -> Self {
        Self { voxel_type, color }
    }

    /// Returns true if this voxel is solid (blocks movement).
    ///
    /// Air voxels are non-solid, all others are considered solid.
    pub fn is_solid(&self) -> bool {
        !matches!(self.voxel_type, VoxelType::Air)
    }
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            voxel_type: VoxelType::Air,
            color: Color::BLACK,
        }
    }
}

/// Represents a position in voxel coordinates.
///
/// Voxel coordinates use integers to represent discrete positions
/// in the voxel grid. This is different from world coordinates
/// which use floating point numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelPosition {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelPosition {
    /// Creates a new voxel position.
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Converts voxel position to world coordinates.
    ///
    /// This is needed for positioning meshes in the 3D world.
    /// Voxel size is assumed to be 1.0 world units.
    pub fn to_world_position(&self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }
}

/// A chunk of voxels.
///
/// The world is divided into chunks for performance reasons.
/// Chunks allow for efficient culling and mesh generation.
/// Each chunk has a fixed size defined by CHUNK_SIZE.
#[derive(Debug, Clone)]
pub struct VoxelChunk {
    /// Position of this chunk in chunk coordinates
    pub chunk_position: ChunkPosition,
    /// The voxels in this chunk
    voxels: Vec<Voxel>,
    /// Handle to the mesh if this chunk has been rendered
    mesh_handle: Option<Handle<Mesh>>,
    /// Entity for this chunk's mesh in the ECS world
    entity: Option<Entity>,
    /// Whether this chunk needs to be remeshed
    dirty: bool,
}

/// Represents a chunk position in the world.
///
/// Chunk coordinates are used to organize chunks in the world.
/// They are different from voxel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Component)]
pub struct ChunkPosition {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPosition {
    /// Creates a new chunk position.
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// The size of each chunk in voxels.
///
/// This is a constant that defines how many voxels are in each dimension of a chunk.
/// 16x16x16 is a good balance between performance and memory usage.
pub const CHUNK_SIZE: usize = 16;

/// The number of voxels in a chunk (cubed).
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

impl VoxelChunk {
    /// Creates a new empty chunk at the given position.
    ///
    /// # Arguments
    ///
    /// * `chunk_position` - The position of this chunk in chunk coordinates
    pub fn new(chunk_position: ChunkPosition) -> Self {
        Self {
            chunk_position,
            voxels: vec![Voxel::default(); CHUNK_VOLUME],
            mesh_handle: None,
            entity: None,
            dirty: true, // New chunks start dirty to trigger mesh generation
        }
    }

    /// Gets a voxel at local chunk coordinates.
    ///
    /// # Arguments
    ///
    /// * `local_pos` - Local position within the chunk (0..CHUNK_SIZE)
    ///
    /// # Returns
    ///
    /// A reference to the voxel at the given position
    pub fn get_voxel(&self, local_pos: VoxelPosition) -> &Voxel {
        let index = Self::local_to_index(local_pos);
        &self.voxels[index]
    }

    /// Sets a voxel at local chunk coordinates.
    ///
    /// # Arguments
    ///
    /// * `local_pos` - Local position within the chunk (0..CHUNK_SIZE)
    /// * `voxel` - The voxel to set
    pub fn set_voxel(&mut self, local_pos: VoxelPosition, voxel: Voxel) {
        let index = Self::local_to_index(local_pos);
        self.voxels[index] = voxel;
        self.dirty = true; // Mark chunk as dirty when voxel changes
    }

    /// Converts local chunk coordinates to array index.
    ///
    /// This is used to store voxels in a flat array.
    /// Uses Z-order (Morton order) for better cache locality.
    fn local_to_index(local_pos: VoxelPosition) -> usize {
        (local_pos.z as usize * CHUNK_SIZE * CHUNK_SIZE)
            + (local_pos.y as usize * CHUNK_SIZE)
            + local_pos.x as usize
    }

    /// Converts array index to local chunk coordinates.
    ///
    /// The inverse of local_to_index.
    pub fn index_to_local(index: usize) -> VoxelPosition {
        VoxelPosition {
            x: (index % CHUNK_SIZE) as i32,
            y: ((index / CHUNK_SIZE) % CHUNK_SIZE) as i32,
            z: (index / (CHUNK_SIZE * CHUNK_SIZE)) as i32,
        }
    }

    /// Gets the world position of this chunk.
    ///
    /// Converts chunk coordinates to world coordinates.
    pub fn get_world_position(&self) -> Vec3 {
        Vec3::new(
            (self.chunk_position.x * CHUNK_SIZE as i32) as f32,
            (self.chunk_position.y * CHUNK_SIZE as i32) as f32,
            (self.chunk_position.z * CHUNK_SIZE as i32) as f32,
        )
    }

    /// Returns whether this chunk needs to be remeshed.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Marks this chunk as clean (after meshing).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Sets the mesh handle for this chunk.
    pub fn set_mesh_handle(&mut self, handle: Handle<Mesh>) {
        self.mesh_handle = Some(handle);
    }

    /// Gets the mesh handle for this chunk.
    pub fn get_mesh_handle(&self) -> Option<Handle<Mesh>> {
        self.mesh_handle.clone()
    }

    /// Sets the entity for this chunk's mesh.
    pub fn set_entity(&mut self, entity: Entity) {
        self.entity = Some(entity);
    }

    /// Gets the entity for this chunk's mesh.
    pub fn get_entity(&self) -> Option<Entity> {
        self.entity
    }

    /// Merges another chunk into this one.
    ///
    /// Non-air voxels from the other chunk will overwrite voxels in this chunk.
    pub fn merge(&mut self, other: VoxelChunk) {
        let mut changed = false;
        for i in 0..CHUNK_VOLUME {
            let other_voxel = other.voxels[i];
            if other_voxel.voxel_type != VoxelType::Air {
                if self.voxels[i] != other_voxel {
                    self.voxels[i] = other_voxel;
                    changed = true;
                }
            }
        }
        if changed {
            self.dirty = true;
        }
    }
}

/// The voxel world containing all chunks.
///
/// This is the main data structure for storing a voxel world.
/// It manages all chunks and provides methods for accessing them.
#[derive(Resource, Debug, Clone)]
pub struct VoxelWorld {
    /// All chunks in the world, indexed by chunk position
    chunks: std::collections::HashMap<ChunkPosition, VoxelChunk>,
}

impl VoxelWorld {
    /// Creates a new empty voxel world.
    pub fn new() -> Self {
        Self {
            chunks: std::collections::HashMap::new(),
        }
    }

    /// Gets a chunk at the given chunk position.
    ///
    /// # Arguments
    ///
    /// * `chunk_pos` - The chunk position to get
    ///
    /// # Returns
    ///
    /// A reference to the chunk if it exists, None otherwise
    pub fn get_chunk(&self, chunk_pos: &ChunkPosition) -> Option<&VoxelChunk> {
        self.chunks.get(chunk_pos)
    }

    /// Gets a mutable chunk at the given chunk position.
    ///
    /// If the chunk doesn't exist, it will be created.
    ///
    /// # Arguments
    ///
    /// * `chunk_pos` - The chunk position to get
    ///
    /// # Returns
    ///
    /// A mutable reference to the chunk
    pub fn get_chunk_mut(&mut self, chunk_pos: &ChunkPosition) -> &mut VoxelChunk {
        self.chunks
            .entry(*chunk_pos)
            .or_insert_with(|| VoxelChunk::new(*chunk_pos))
    }

    /// Gets a voxel at the given voxel position.
    ///
    /// # Arguments
    ///
    /// * `voxel_pos` - The voxel position to get
    ///
    /// # Returns
    ///
    /// A reference to the voxel if the chunk exists, None otherwise
    pub fn get_voxel(&self, voxel_pos: &VoxelPosition) -> Option<&Voxel> {
        let chunk_pos = Self::voxel_to_chunk(voxel_pos);
        let local_pos = Self::voxel_to_local(voxel_pos);

        self.get_chunk(&chunk_pos)
            .map(|chunk| chunk.get_voxel(local_pos))
    }

    /// Sets a voxel at the given voxel position.
    ///
    /// If the chunk doesn't exist, it will be created.
    ///
    /// # Arguments
    ///
    /// * `voxel_pos` - The voxel position to set
    /// * `voxel` - The voxel to set
    pub fn set_voxel(&mut self, voxel_pos: &VoxelPosition, voxel: Voxel) {
        let chunk_pos = Self::voxel_to_chunk(voxel_pos);
        let local_pos = Self::voxel_to_local(voxel_pos);

        let chunk = self.get_chunk_mut(&chunk_pos);
        chunk.set_voxel(local_pos, voxel);
    }

    /// Converts voxel coordinates to chunk coordinates.
    ///
    /// # Arguments
    ///
    /// * `voxel_pos` - The voxel position to convert
    ///
    /// # Returns
    ///
    /// The corresponding chunk position
    pub fn voxel_to_chunk(voxel_pos: &VoxelPosition) -> ChunkPosition {
        ChunkPosition {
            x: voxel_pos.x.div_euclid(CHUNK_SIZE as i32),
            y: voxel_pos.y.div_euclid(CHUNK_SIZE as i32),
            z: voxel_pos.z.div_euclid(CHUNK_SIZE as i32),
        }
    }

    /// Converts voxel coordinates to local chunk coordinates.
    ///
    /// # Arguments
    ///
    /// * `voxel_pos` - The voxel position to convert
    ///
    /// # Returns
    ///
    /// The corresponding local position within the chunk
    pub fn voxel_to_local(voxel_pos: &VoxelPosition) -> VoxelPosition {
        VoxelPosition {
            x: voxel_pos.x.rem_euclid(CHUNK_SIZE as i32),
            y: voxel_pos.y.rem_euclid(CHUNK_SIZE as i32),
            z: voxel_pos.z.rem_euclid(CHUNK_SIZE as i32),
        }
    }

    /// Gets all chunks that need to be remeshed.
    ///
    /// # Returns
    ///
    /// A vector of chunk positions that are marked as dirty
    #[instrument(skip(self))]
    pub fn get_dirty_chunks(&self) -> Vec<&ChunkPosition> {
        let dirty_chunks: Vec<_> = self
            .chunks
            .iter()
            .filter(|(_, chunk)| chunk.is_dirty())
            .map(|(pos, _)| pos)
            .collect();

        debug!(
            total_chunks = self.chunks.len(),
            dirty_chunks = dirty_chunks.len(),
            "Retrieved dirty chunks"
        );

        dirty_chunks
    }

    /// Iterates over all chunks in the world.
    pub fn iter_chunks(&self) -> impl Iterator<Item = (&ChunkPosition, &VoxelChunk)> {
        self.chunks.iter()
    }

    /// Iterates over all chunks in the world (mutable).
    pub fn iter_chunks_mut(&mut self) -> impl Iterator<Item = (&ChunkPosition, &mut VoxelChunk)> {
        self.chunks.iter_mut()
    }

    /// Merges another voxel world into this one.
    ///
    /// This will combine the contents of chunks that exist in both worlds.
    pub fn merge(&mut self, other: VoxelWorld) {
        for (pos, other_chunk) in other.chunks {
            if let Some(self_chunk) = self.chunks.get_mut(&pos) {
                self_chunk.merge(other_chunk);
            } else {
                self.chunks.insert(pos, other_chunk);
            }
        }
    }
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self::new()
    }
}
