/// Terrain generation module.
///
/// This module provides terrain generation functionality for the voxel game.
/// It includes both single-threaded and multithreaded implementations.
pub mod terrain;
#[cfg(test)]
mod terrain_tests;

// Re-export the types needed by other modules
pub use self::terrain::*;
