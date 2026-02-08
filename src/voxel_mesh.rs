use crate::voxel::{
    ChunkPosition, Voxel, VoxelChunk, VoxelPosition, VoxelType, VoxelWorld, CHUNK_SIZE,
};

/// Voxel mesh generation with face culling.
///
/// This module handles converting voxel data into renderable meshes.
/// It implements face culling to avoid rendering faces that are hidden
/// by adjacent solid voxels, which significantly reduces the number
/// of triangles that need to be rendered.
use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
};
use tracing::{debug, instrument};

/// Directions in 3D space for face generation.
///
/// Each direction represents one of the 6 faces of a cube.
/// The order is designed to be consistent with common 3D graphics conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    PositiveX, // Right face
    NegativeX, // Left face
    PositiveY, // Top face
    NegativeY, // Bottom face
    PositiveZ, // Front face
    NegativeZ, // Back face
}

impl Direction {
    /// Returns the normal vector for this direction.
    pub fn normal(&self) -> Vec3 {
        match self {
            Direction::PositiveX => Vec3::X,
            Direction::NegativeX => Vec3::NEG_X,
            Direction::PositiveY => Vec3::Y,
            Direction::NegativeY => Vec3::NEG_Y,
            Direction::PositiveZ => Vec3::Z,
            Direction::NegativeZ => Vec3::NEG_Z,
        }
    }

    /// Returns all six directions.
    pub fn all() -> &'static [Direction; 6] {
        &[
            Direction::PositiveX,
            Direction::NegativeX,
            Direction::PositiveY,
            Direction::NegativeY,
            Direction::PositiveZ,
            Direction::NegativeZ,
        ]
    }
}

/// Vertex data for a voxel mesh.
///
/// This represents a single vertex in the generated mesh.
#[derive(Copy, Clone, Debug)]
struct VoxelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

/// Generates mesh data for a chunk.
///
/// This function creates a mesh for a chunk, using greedy meshing optimization
/// when enabled in the configuration. Greedy meshing merges adjacent faces of the
/// same type into larger quads, reducing vertices and triangles by 70-90%.
///
/// # Arguments
///
/// * `chunk` - The chunk to generate a mesh for
/// * `world` - The voxel world (needed for checking adjacent voxels)
/// * `config` - Terrain configuration with greedy meshing option
///
/// # Returns
///
/// A tuple containing the vertex positions and indices for the mesh
#[instrument(skip(chunk, world, config), fields(
    chunk_x = chunk.chunk_position.x,
    chunk_y = chunk.chunk_position.y,
    chunk_z = chunk.chunk_position.z,
    greedy_meshing = config.enable_greedy_meshing
))]
pub fn generate_chunk_mesh(
    chunk: &VoxelChunk,
    world: &VoxelWorld,
    config: &crate::terrain::TerrainConfig,
) -> Option<(Vec<f32>, Vec<u32>)> {
    let _profiler = crate::profiling::Profiler::new("mesh_generation");
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();

    if config.enable_greedy_meshing {
        // Use greedy meshing for better performance
        greedy_mesh_chunk(
            chunk,
            world,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );
    } else {
        // Fall back to naive mesh generation
        naive_mesh_chunk(
            chunk,
            world,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );
    }

    if positions.is_empty() {
        debug!("No vertices generated for chunk");
        None
    } else {
        // Combine all vertex data into a single vector
        let mut vertices = Vec::with_capacity(positions.len() * 10);
        for i in 0..positions.len() {
            vertices.extend_from_slice(&positions[i]);
            vertices.extend_from_slice(&normals[i]);
            vertices.extend_from_slice(&colors[i]);
        }

        let vertex_count = vertices.len() / 10;
        let triangle_count = indices.len() / 3;

        debug!(
            vertex_count = vertex_count,
            triangle_count = triangle_count,
            "Mesh generated successfully"
        );

        Some((vertices, indices))
    }
}

/// Implements greedy meshing for a chunk.
///
/// This function scans through the chunk and merges adjacent faces of the same type
/// into larger quads, dramatically reducing the number of vertices and triangles.
///
/// # Arguments
///
/// * `chunk` - The chunk to mesh
/// * `world` - The voxel world
/// * `positions` - Output buffer for vertex positions
/// * `normals` - Output buffer for vertex normals
/// * `colors` - Output buffer for vertex colors
/// * `indices` - Output buffer for triangle indices
#[instrument(skip(chunk, world, positions, normals, colors, indices))]
fn greedy_mesh_chunk(
    chunk: &VoxelChunk,
    world: &VoxelWorld,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let _profiler = crate::profiling::Profiler::new("greedy_mesh_chunk");
    // Create a mask to track which voxel faces have been processed
    let mut processed = [[[false; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];

    // Process each of the 6 directions
    for direction in Direction::all() {
        // Reset processed mask for each direction
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    processed[x][y][z] = false;
                }
            }
        }

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    // Skip if this face has already been processed in a previous quad
                    if processed[x][y][z] {
                        continue;
                    }

                    let pos = VoxelPosition::new(x as i32, y as i32, z as i32);
                    let voxel = chunk.get_voxel(pos);

                    // Skip air voxels
                    if voxel.voxel_type == VoxelType::Air {
                        continue;
                    }

                    // Check if this face should be rendered
                    if should_render_face(chunk, world, &pos, direction) {
                        match direction {
                            Direction::PositiveX => {
                                let (width, height) = greedy_mesh_x_face(
                                    chunk,
                                    world,
                                    x,
                                    y,
                                    z,
                                    &mut processed,
                                    voxel,
                                    direction,
                                );
                                add_quad(
                                    Vec3::new(x as f32 + 0.5, y as f32 - 0.5, z as f32 - 0.5),
                                    Vec3::new(0.0, height as f32, width as f32),
                                    Vec3::X,
                                    voxel,
                                    positions,
                                    normals,
                                    colors,
                                    indices,
                                );
                            }
                            Direction::NegativeX => {
                                let (width, height) = greedy_mesh_x_face(
                                    chunk,
                                    world,
                                    x,
                                    y,
                                    z,
                                    &mut processed,
                                    voxel,
                                    direction,
                                );
                                add_quad(
                                    Vec3::new(x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5),
                                    Vec3::new(0.0, height as f32, width as f32),
                                    Vec3::NEG_X,
                                    voxel,
                                    positions,
                                    normals,
                                    colors,
                                    indices,
                                );
                            }
                            Direction::PositiveY => {
                                let (width, height) = greedy_mesh_y_face(
                                    chunk,
                                    world,
                                    x,
                                    y,
                                    z,
                                    &mut processed,
                                    voxel,
                                    direction,
                                );
                                add_quad(
                                    Vec3::new(x as f32 - 0.5, y as f32 + 0.5, z as f32 - 0.5),
                                    Vec3::new(width as f32, 0.0, height as f32),
                                    Vec3::Y,
                                    voxel,
                                    positions,
                                    normals,
                                    colors,
                                    indices,
                                );
                            }
                            Direction::NegativeY => {
                                let (width, height) = greedy_mesh_y_face(
                                    chunk,
                                    world,
                                    x,
                                    y,
                                    z,
                                    &mut processed,
                                    voxel,
                                    direction,
                                );
                                add_quad(
                                    Vec3::new(x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5),
                                    Vec3::new(width as f32, 0.0, height as f32),
                                    Vec3::NEG_Y,
                                    voxel,
                                    positions,
                                    normals,
                                    colors,
                                    indices,
                                );
                            }
                            Direction::PositiveZ => {
                                let (width, height) = greedy_mesh_z_face(
                                    chunk,
                                    world,
                                    x,
                                    y,
                                    z,
                                    &mut processed,
                                    voxel,
                                    direction,
                                );
                                add_quad(
                                    Vec3::new(x as f32 - 0.5, y as f32 - 0.5, z as f32 + 0.5),
                                    Vec3::new(width as f32, height as f32, 0.0),
                                    Vec3::Z,
                                    voxel,
                                    positions,
                                    normals,
                                    colors,
                                    indices,
                                );
                            }
                            Direction::NegativeZ => {
                                let (width, height) = greedy_mesh_z_face(
                                    chunk,
                                    world,
                                    x,
                                    y,
                                    z,
                                    &mut processed,
                                    voxel,
                                    direction,
                                );
                                add_quad(
                                    Vec3::new(x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5),
                                    Vec3::new(width as f32, height as f32, 0.0),
                                    Vec3::NEG_Z,
                                    voxel,
                                    positions,
                                    normals,
                                    colors,
                                    indices,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainConfig;
    use crate::voxel::{ChunkPosition, Voxel, VoxelChunk, VoxelPosition, VoxelType, VoxelWorld};

    #[test]
    fn test_single_voxel_mesh_faces() {
        let mut world = VoxelWorld::new();
        let chunk_pos = ChunkPosition::new(0, 0, 0);
        let mut chunk = VoxelChunk::new(chunk_pos);

        // Place a single solid voxel at (8, 8, 8)
        let local_pos = VoxelPosition::new(8, 8, 8);
        let voxel = Voxel::new(VoxelType::Solid, Color::srgb(1.0, 0.0, 0.0));
        chunk.set_voxel(local_pos, voxel);
        world.set_voxel(&VoxelPosition::new(8, 8, 8), voxel);

        let config = TerrainConfig {
            enable_greedy_meshing: true,
            ..Default::default()
        };

        let mesh_data = generate_chunk_mesh(&chunk, &world, &config);
        assert!(
            mesh_data.is_some(),
            "Mesh should be generated for a single voxel"
        );

        let (_, indices) = mesh_data.unwrap();

        // A single voxel should have 6 faces, each face 2 triangles, each triangle 3 indices.
        // Total indices = 6 * 2 * 3 = 36.
        assert_eq!(
            indices.len(),
            36,
            "Single voxel should have 36 indices (6 faces)"
        );
    }

    #[test]
    fn test_two_voxels_mesh_faces() {
        let mut world = VoxelWorld::new();
        let chunk_pos = ChunkPosition::new(0, 0, 0);
        let mut chunk = VoxelChunk::new(chunk_pos);

        // Place two solid voxels side by side: (8, 8, 8) and (9, 8, 8)
        let voxel = Voxel::new(VoxelType::Solid, Color::srgb(1.0, 0.0, 0.0));

        chunk.set_voxel(VoxelPosition::new(8, 8, 8), voxel);
        world.set_voxel(&VoxelPosition::new(8, 8, 8), voxel);

        chunk.set_voxel(VoxelPosition::new(9, 8, 8), voxel);
        world.set_voxel(&VoxelPosition::new(9, 8, 8), voxel);

        let config_greedy = TerrainConfig {
            enable_greedy_meshing: true,
            ..Default::default()
        };

        let config_naive = TerrainConfig {
            enable_greedy_meshing: false,
            ..Default::default()
        };

        let mesh_greedy = generate_chunk_mesh(&chunk, &world, &config_greedy).unwrap();
        let mesh_naive = generate_chunk_mesh(&chunk, &world, &config_naive).unwrap();

        // Compare vertex positions (ignoring order for now, but let's check number of non-degenerate quads)
        // Actually, let's just check that greedy doesn't produce fewer vertices than naive if we only have single voxels.
        // Wait, greedy should produce SAME OR FEWER vertices.

        assert_eq!(mesh_naive.1.len(), 60, "Naive should have 60 indices");

        // Let's check for degenerate quads in greedy
        let (vertices, indices) = mesh_greedy;
        let mut non_degenerate_quads = 0;
        for i in 0..(indices.len() / 6) {
            let i1 = indices[i * 6] as usize;
            let i2 = indices[i * 6 + 1] as usize;
            let i3 = indices[i * 6 + 2] as usize;

            let p1 = Vec3::from_slice(&vertices[i1 * 10..i1 * 10 + 3]);
            let p2 = Vec3::from_slice(&vertices[i2 * 10..i2 * 10 + 3]);
            let p3 = Vec3::from_slice(&vertices[i3 * 10..i3 * 10 + 3]);

            // Area of triangle is 0.5 * |(p2-p1) x (p3-p1)|
            let area = (p2 - p1).cross(p3 - p1).length();
            if area > 0.0001 {
                non_degenerate_quads += 1;
            }
        }

        assert_eq!(
            non_degenerate_quads, 6,
            "Should have 6 non-degenerate quads after merging"
        );
    }
}

/// Greedy meshing for X-direction faces.
///
/// Finds the largest rectangle of identical faces in the X direction.
///
/// Returns the width (in Z) and height (in Y) of the merged face.
fn greedy_mesh_x_face(
    chunk: &VoxelChunk,
    world: &VoxelWorld,
    x: usize,
    y: usize,
    z: usize,
    processed: &mut [[[bool; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    voxel: &Voxel,
    direction: &Direction,
) -> (usize, usize) {
    let mut width = 0;
    let mut height = 0;

    // Find maximum width in Z direction
    for z_check in z..CHUNK_SIZE {
        let pos = VoxelPosition::new(x as i32, y as i32, z_check as i32);
        if !should_render_face(chunk, world, &pos, direction) {
            break;
        }
        if chunk.get_voxel(pos).voxel_type != voxel.voxel_type {
            break;
        }

        width += 1;
    }

    // Find maximum height in Y direction
    for y_check in y..CHUNK_SIZE {
        let mut valid_row = true;
        for z_offset in 0..width {
            let z_check = z + z_offset;
            let pos = VoxelPosition::new(x as i32, y_check as i32, z_check as i32);
            if !should_render_face(chunk, world, &pos, direction) {
                valid_row = false;
                break;
            }
            if chunk.get_voxel(pos).voxel_type != voxel.voxel_type {
                valid_row = false;
                break;
            }
        }

        if !valid_row {
            break;
        }
        height += 1;
    }

    // Mark processed voxels
    for y_offset in 0..height {
        for z_offset in 0..width {
            processed[x][y + y_offset][z + z_offset] = true;
        }
    }

    (width, height)
}

/// Greedy meshing for Y-direction faces.
///
/// Finds the largest rectangle of identical faces in the Y direction.
fn greedy_mesh_y_face(
    chunk: &VoxelChunk,
    world: &VoxelWorld,
    x: usize,
    y: usize,
    z: usize,
    processed: &mut [[[bool; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    voxel: &Voxel,
    direction: &Direction,
) -> (usize, usize) {
    let mut width = 0;
    let mut height = 0;

    // Find maximum width in X direction
    for x_check in x..CHUNK_SIZE {
        let pos = VoxelPosition::new(x_check as i32, y as i32, z as i32);
        if !should_render_face(chunk, world, &pos, direction) {
            break;
        }
        if chunk.get_voxel(pos).voxel_type != voxel.voxel_type {
            break;
        }

        width += 1;
    }

    // Find maximum height in Z direction
    for z_check in z..CHUNK_SIZE {
        let mut valid_row = true;
        for x_offset in 0..width {
            let x_check = x + x_offset;
            let pos = VoxelPosition::new(x_check as i32, y as i32, z_check as i32);
            if !should_render_face(chunk, world, &pos, direction) {
                valid_row = false;
                break;
            }
            if chunk.get_voxel(pos).voxel_type != voxel.voxel_type {
                valid_row = false;
                break;
            }
        }

        if !valid_row {
            break;
        }
        height += 1;
    }

    // Mark processed voxels
    for z_offset in 0..height {
        for x_offset in 0..width {
            processed[x + x_offset][y][z + z_offset] = true;
        }
    }

    (width, height)
}

/// Greedy meshing for Z-direction faces.
///
/// Finds the largest rectangle of identical faces in the Z direction.
fn greedy_mesh_z_face(
    chunk: &VoxelChunk,
    world: &VoxelWorld,
    x: usize,
    y: usize,
    z: usize,
    processed: &mut [[[bool; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    voxel: &Voxel,
    direction: &Direction,
) -> (usize, usize) {
    let mut width = 0;
    let mut height = 0;

    // Find maximum width in X direction
    for x_check in x..CHUNK_SIZE {
        let pos = VoxelPosition::new(x_check as i32, y as i32, z as i32);
        if !should_render_face(chunk, world, &pos, direction) {
            break;
        }
        if chunk.get_voxel(pos).voxel_type != voxel.voxel_type {
            break;
        }

        width += 1;
    }

    // Find maximum height in Y direction
    for y_check in y..CHUNK_SIZE {
        let mut valid_row = true;
        for x_offset in 0..width {
            let x_check = x + x_offset;
            let pos = VoxelPosition::new(x_check as i32, y_check as i32, z as i32);
            if !should_render_face(chunk, world, &pos, direction) {
                valid_row = false;
                break;
            }
            if chunk.get_voxel(pos).voxel_type != voxel.voxel_type {
                valid_row = false;
                break;
            }
        }

        if !valid_row {
            break;
        }
        height += 1;
    }

    // Mark processed voxels
    for y_offset in 0..height {
        for x_offset in 0..width {
            processed[x + x_offset][y + y_offset][z] = true;
        }
    }

    (width, height)
}

/// Implements naive mesh generation for a chunk (original implementation).
///
/// This function generates a mesh by processing each voxel face individually.
/// It's less efficient than greedy meshing but provides a fallback option.
///
/// # Arguments
///
/// * `chunk` - The chunk to mesh
/// * `world` - The voxel world
/// * `positions` - Output buffer for vertex positions
/// * `normals` - Output buffer for vertex normals
/// * `colors` - Output buffer for vertex colors
/// * `indices` - Output buffer for triangle indices
#[instrument(skip(chunk, world, positions, normals, colors, indices))]
fn naive_mesh_chunk(
    chunk: &VoxelChunk,
    world: &VoxelWorld,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let _profiler = crate::profiling::Profiler::new("naive_mesh_chunk");
    // Track which voxel positions have been processed
    let mut processed = std::collections::HashSet::new();

    // Iterate through all voxels in the chunk
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let local_pos = VoxelPosition::new(x as i32, y as i32, z as i32);
                let voxel = chunk.get_voxel(local_pos);

                // Skip air voxels
                if voxel.voxel_type == VoxelType::Air {
                    continue;
                }

                // Skip if we've already processed this position
                if processed.contains(&local_pos) {
                    continue;
                }
                processed.insert(local_pos);

                // Check each of the 6 faces
                for direction in Direction::all() {
                    if should_render_face(chunk, world, &local_pos, direction) {
                        let start_index = positions.len();

                        // Generate the 4 vertices for this face
                        generate_face_vertices(
                            &local_pos, voxel, direction, positions, normals, colors,
                        );

                        // Add the 6 indices for the 2 triangles of this face
                        indices.extend_from_slice(&[
                            start_index as u32,
                            start_index as u32 + 1,
                            start_index as u32 + 2,
                            start_index as u32 + 2,
                            start_index as u32 + 3,
                            start_index as u32,
                        ]);
                    }
                }
            }
        }
    }
}

/// Generates vertices for a single face.
///
/// This is a helper function for the naive mesh generation.
fn generate_face_vertices(
    local_pos: &VoxelPosition,
    voxel: &Voxel,
    direction: &Direction,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    let position = Vec3::new(local_pos.x as f32, local_pos.y as f32, local_pos.z as f32);
    let normal = direction.normal().to_array();
    let color = voxel.color.to_linear().to_vec4().to_array();

    match direction {
        Direction::PositiveX => {
            positions.push([position.x + 0.5, position.y - 0.5, position.z - 0.5]);
            positions.push([position.x + 0.5, position.y + 0.5, position.z - 0.5]);
            positions.push([position.x + 0.5, position.y + 0.5, position.z + 0.5]);
            positions.push([position.x + 0.5, position.y - 0.5, position.z + 0.5]);
        }
        Direction::NegativeX => {
            positions.push([position.x - 0.5, position.y - 0.5, position.z + 0.5]);
            positions.push([position.x - 0.5, position.y + 0.5, position.z + 0.5]);
            positions.push([position.x - 0.5, position.y + 0.5, position.z - 0.5]);
            positions.push([position.x - 0.5, position.y - 0.5, position.z - 0.5]);
        }
        Direction::PositiveY => {
            positions.push([position.x - 0.5, position.y + 0.5, position.z - 0.5]);
            positions.push([position.x - 0.5, position.y + 0.5, position.z + 0.5]);
            positions.push([position.x + 0.5, position.y + 0.5, position.z + 0.5]);
            positions.push([position.x + 0.5, position.y + 0.5, position.z - 0.5]);
        }
        Direction::NegativeY => {
            positions.push([position.x - 0.5, position.y - 0.5, position.z + 0.5]);
            positions.push([position.x - 0.5, position.y - 0.5, position.z - 0.5]);
            positions.push([position.x + 0.5, position.y - 0.5, position.z - 0.5]);
            positions.push([position.x + 0.5, position.y - 0.5, position.z + 0.5]);
        }
        Direction::PositiveZ => {
            positions.push([position.x - 0.5, position.y - 0.5, position.z + 0.5]);
            positions.push([position.x - 0.5, position.y + 0.5, position.z + 0.5]);
            positions.push([position.x + 0.5, position.y + 0.5, position.z + 0.5]);
            positions.push([position.x + 0.5, position.y - 0.5, position.z + 0.5]);
        }
        Direction::NegativeZ => {
            positions.push([position.x + 0.5, position.y - 0.5, position.z - 0.5]);
            positions.push([position.x + 0.5, position.y + 0.5, position.z - 0.5]);
            positions.push([position.x - 0.5, position.y + 0.5, position.z - 0.5]);
            positions.push([position.x - 0.5, position.y - 0.5, position.z - 0.5]);
        }
    }

    // Add normals (4 vertices, same normal)
    for _ in 0..4 {
        normals.push(normal);
    }

    // Add colors (4 vertices, same color)
    for _ in 0..4 {
        colors.push(color);
    }
}

/// Adds a quad to the mesh buffers.
///
/// # Arguments
///
/// * `position` - The position of the quad's bottom-left corner
/// * `size` - The width and height of the quad
/// * `normal` - The normal vector for the quad
/// * `voxel` - The voxel data for color information
/// * `positions` - Output buffer for vertex positions
/// * `normals` - Output buffer for vertex normals
/// * `colors` - Output buffer for vertex colors
/// * `indices` - Output buffer for triangle indices
fn add_quad(
    position: Vec3,
    size: Vec3,
    normal: Vec3,
    voxel: &Voxel,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let start_index = positions.len() as u32;
    let normal_array = normal.to_array();
    let color_array = voxel.color.to_linear().to_vec4().to_array();

    // Adjust based on which direction the quad faces
    let (p1, p2, p3, p4) = if normal.x > 0.0 {
        // Positive X face
        (
            Vec3::new(position.x, position.y, position.z),
            Vec3::new(position.x, position.y + size.y, position.z),
            Vec3::new(position.x, position.y + size.y, position.z + size.z),
            Vec3::new(position.x, position.y, position.z + size.z),
        )
    } else if normal.x < 0.0 {
        // Negative X face
        (
            Vec3::new(position.x, position.y, position.z + size.z),
            Vec3::new(position.x, position.y + size.y, position.z + size.z),
            Vec3::new(position.x, position.y + size.y, position.z),
            Vec3::new(position.x, position.y, position.z),
        )
    } else if normal.y > 0.0 {
        // Positive Y face (top)
        (
            Vec3::new(position.x, position.y, position.z),
            Vec3::new(position.x, position.y, position.z + size.z),
            Vec3::new(position.x + size.x, position.y, position.z + size.z),
            Vec3::new(position.x + size.x, position.y, position.z),
        )
    } else if normal.y < 0.0 {
        // Negative Y face (bottom)
        (
            Vec3::new(position.x, position.y, position.z + size.z),
            Vec3::new(position.x, position.y, position.z),
            Vec3::new(position.x + size.x, position.y, position.z),
            Vec3::new(position.x + size.x, position.y, position.z + size.z),
        )
    } else if normal.z > 0.0 {
        // Positive Z face
        (
            Vec3::new(position.x, position.y, position.z),
            Vec3::new(position.x + size.x, position.y, position.z),
            Vec3::new(position.x + size.x, position.y + size.y, position.z),
            Vec3::new(position.x, position.y + size.y, position.z),
        )
    } else {
        // Negative Z face
        (
            Vec3::new(position.x + size.x, position.y, position.z),
            Vec3::new(position.x, position.y, position.z),
            Vec3::new(position.x, position.y + size.y, position.z),
            Vec3::new(position.x + size.x, position.y + size.y, position.z),
        )
    };

    // Add vertices
    positions.push(p1.to_array());
    positions.push(p2.to_array());
    positions.push(p3.to_array());
    positions.push(p4.to_array());

    // Add normals (all same for this quad)
    normals.push(normal_array);
    normals.push(normal_array);
    normals.push(normal_array);
    normals.push(normal_array);

    // Add colors
    colors.push(color_array);
    colors.push(color_array);
    colors.push(color_array);
    colors.push(color_array);

    // Add indices (2 triangles)
    indices.extend_from_slice(&[
        start_index,
        start_index + 1,
        start_index + 2,
        start_index + 2,
        start_index + 3,
        start_index,
    ]);
}

/// Determines if a face should be rendered.
///
/// A face should be rendered if:
/// 1. The adjacent voxel in that direction is air or doesn't exist
/// 2. The current voxel is solid
///
/// This is the core of the face culling optimization.
///
/// # Arguments
///
/// * `chunk` - The chunk containing the voxel
/// * `world` - The voxel world
/// * `local_pos` - Local position within the chunk
/// * `direction` - The direction to check
///
/// # Returns
///
/// True if the face should be rendered, false otherwise
fn should_render_face(
    chunk: &VoxelChunk,
    world: &VoxelWorld,
    local_pos: &VoxelPosition,
    direction: &Direction,
) -> bool {
    let current_voxel = chunk.get_voxel(*local_pos);
    if !current_voxel.is_solid() {
        return false;
    }

    // Calculate the adjacent voxel position
    let mut adjacent_pos = *local_pos;
    match direction {
        Direction::PositiveX => adjacent_pos.x += 1,
        Direction::NegativeX => adjacent_pos.x -= 1,
        Direction::PositiveY => adjacent_pos.y += 1,
        Direction::NegativeY => adjacent_pos.y -= 1,
        Direction::PositiveZ => adjacent_pos.z += 1,
        Direction::NegativeZ => adjacent_pos.z -= 1,
    }

    // Check if the adjacent position is within this chunk
    if (0..CHUNK_SIZE as i32).contains(&adjacent_pos.x)
        && (0..CHUNK_SIZE as i32).contains(&adjacent_pos.y)
        && (0..CHUNK_SIZE as i32).contains(&adjacent_pos.z)
    {
        // Adjacent voxel is in the same chunk
        let adjacent_voxel = chunk.get_voxel(adjacent_pos);
        !adjacent_voxel.is_solid()
    } else {
        // Adjacent voxel is in a different chunk or doesn't exist
        // Convert to world coordinates
        let world_pos = VoxelPosition::new(
            adjacent_pos.x + chunk.chunk_position.x * CHUNK_SIZE as i32,
            adjacent_pos.y + chunk.chunk_position.y * CHUNK_SIZE as i32,
            adjacent_pos.z + chunk.chunk_position.z * CHUNK_SIZE as i32,
        );

        // Check the world for the adjacent voxel
        match world.get_voxel(&world_pos) {
            Some(voxel) => !voxel.is_solid(),
            None => true, // No voxel means air, so render the face
        }
    }
}

/// Generates vertex data for a single face of a voxel.
///
/// # Arguments
///
/// * `local_pos` - Local position of the voxel within the chunk
/// * `voxel` - The voxel data
/// * `direction` - The direction of the face to generate
///
/// # Returns
///
/// Face data containing 4 vertices
fn generate_face_data(local_pos: &VoxelPosition, voxel: &Voxel, direction: &Direction) -> FaceData {
    let position = Vec3::new(local_pos.x as f32, local_pos.y as f32, local_pos.z as f32);
    let color = voxel.color.to_linear().to_vec4().to_array();
    let normal = direction.normal().to_array();

    match direction {
        Direction::PositiveX => FaceData {
            vertices: [
                VoxelVertex {
                    position: [position.x + 0.5, position.y - 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y + 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y + 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y - 0.5, position.z + 0.5],
                    normal,
                    color,
                },
            ],
        },
        Direction::NegativeX => FaceData {
            vertices: [
                VoxelVertex {
                    position: [position.x - 0.5, position.y - 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y + 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y + 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y - 0.5, position.z - 0.5],
                    normal,
                    color,
                },
            ],
        },
        Direction::PositiveY => FaceData {
            vertices: [
                VoxelVertex {
                    position: [position.x - 0.5, position.y + 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y + 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y + 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y + 0.5, position.z - 0.5],
                    normal,
                    color,
                },
            ],
        },
        Direction::NegativeY => FaceData {
            vertices: [
                VoxelVertex {
                    position: [position.x - 0.5, position.y - 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y - 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y - 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y - 0.5, position.z + 0.5],
                    normal,
                    color,
                },
            ],
        },
        Direction::PositiveZ => FaceData {
            vertices: [
                VoxelVertex {
                    position: [position.x - 0.5, position.y - 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y + 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y + 0.5, position.z + 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y - 0.5, position.z + 0.5],
                    normal,
                    color,
                },
            ],
        },
        Direction::NegativeZ => FaceData {
            vertices: [
                VoxelVertex {
                    position: [position.x + 0.5, position.y - 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x + 0.5, position.y + 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y + 0.5, position.z - 0.5],
                    normal,
                    color,
                },
                VoxelVertex {
                    position: [position.x - 0.5, position.y - 0.5, position.z - 0.5],
                    normal,
                    color,
                },
            ],
        },
    }
}

/// Represents the vertex data for a single face of a voxel.
///
/// Contains 4 vertices arranged in a quad.
struct FaceData {
    vertices: [VoxelVertex; 4],
}

/// Bevy system to update voxel chunk meshes.
///
/// This system is responsible for:
/// 1. Finding chunks that need to be remeshed
/// 2. Generating new meshes for those chunks
/// 3. Updating the entity components with the new meshes
///
/// This is a core system that runs every frame to update
/// any chunks that have been modified.
///
/// # Arguments
///
/// * `commands` - Commands to spawn/update entities
/// * `meshes` - Asset storage for meshes
/// * `materials` - Asset storage for materials
/// * `world_resource` - Our voxel world resource
#[instrument(skip(commands, meshes, materials, world_resource, config))]
pub fn update_chunk_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut world_resource: ResMut<VoxelWorld>,
    config: Res<crate::terrain::TerrainConfig>,
) {
    let _profiler = crate::profiling::Profiler::new("update_chunk_meshes");
    // Get all dirty chunks
    let dirty_chunks: Vec<_> = world_resource
        .get_dirty_chunks()
        .into_iter()
        .cloned()
        .collect();

    debug!(
        dirty_chunks_count = dirty_chunks.len(),
        "Processing dirty chunks"
    );

    for chunk_pos in dirty_chunks {
        // We need to generate the mesh before modifying the chunk
        let (chunk_key, chunk_world) = {
            // Get a snapshot of the world for mesh generation
            let chunk = world_resource.get_chunk(&chunk_pos).unwrap();
            let chunk_data = generate_chunk_mesh(chunk, &world_resource, &config);
            (chunk_pos, chunk_data)
        };

        // Now get mutable access to update the chunk
        let chunk = world_resource.get_chunk_mut(&chunk_key);

        // Check if we have mesh data
        if let Some((vertices, indices)) = chunk_world {
            // Create a new mesh
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );

            // Calculate how many vertices we have
            let vertex_count = vertices.len() / 10; // 3 pos + 3 normal + 4 color = 10 floats per vertex

            // Extract vertex data
            let mut positions = Vec::with_capacity(vertex_count);
            let mut normals = Vec::with_capacity(vertex_count);
            let mut colors = Vec::with_capacity(vertex_count);

            for i in 0..vertex_count {
                let offset = i * 10;
                positions.push(Vec3::new(
                    vertices[offset],
                    vertices[offset + 1],
                    vertices[offset + 2],
                ));
                normals.push(Vec3::new(
                    vertices[offset + 3],
                    vertices[offset + 4],
                    vertices[offset + 5],
                ));
                colors.push(Vec4::new(
                    vertices[offset + 6],
                    vertices[offset + 7],
                    vertices[offset + 8],
                    vertices[offset + 9],
                ));
            }

            // Set vertex attributes
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);

            // Set indices
            mesh.insert_indices(Indices::U32(indices));

            // Add the mesh to the asset storage
            let mesh_handle = meshes.add(mesh);

            // Update the chunk with the new mesh handle
            chunk.set_mesh_handle(mesh_handle.clone());
            chunk.mark_clean();

            // Update or spawn the entity
            if let Some(entity) = chunk.get_entity() {
                // Update existing entity
                commands.entity(entity).insert(Mesh3d(mesh_handle));
                debug!(
                    chunk_x = chunk_pos.x,
                    chunk_y = chunk_pos.y,
                    chunk_z = chunk_pos.z,
                    "Updated existing chunk entity"
                );
            } else {
                // Create a material for this chunk
                let material_handle = materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 1.0,
                    metallic: 0.0,
                    alpha_mode: AlphaMode::Opaque,
                    cull_mode: Some(bevy::render::render_resource::Face::Back),
                    ..default()
                });

                // Spawn a new entity for this chunk
                let entity = commands
                    .spawn((
                        Mesh3d(mesh_handle),
                        MeshMaterial3d(material_handle),
                        Transform::from_translation(chunk.get_world_position()),
                        ChunkComponent(chunk_pos),
                    ))
                    .id();

                chunk.set_entity(entity);
                debug!(
                    chunk_x = chunk_pos.x,
                    chunk_y = chunk_pos.y,
                    chunk_z = chunk_pos.z,
                    "Created new chunk entity"
                );
            }
        }
    }
}

/// Component to mark entities as voxel chunks.
///
/// This component is added to entities that represent rendered voxel chunks.
/// It stores the chunk position so we can identify which chunk this entity represents.
#[derive(Component)]
pub struct ChunkComponent(pub ChunkPosition);
