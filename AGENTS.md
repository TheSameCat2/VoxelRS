# AGENTS.md

This file contains guidelines for agentic coding agents working with this voxel-based game project built with Bevy.

## Project Overview

This is a Rust-based voxel game using the Bevy game engine with the following key components:
- Voxel world system with chunk-based terrain generation
- Multithreaded terrain generation using Rayon
- Player controller with voxel interaction
- Remote debugging capabilities via Bevy Remote Protocol (BRP)
- Profiling and tracing integration

## Build/Test Commands

### Core Commands
- `cargo build` - Build the project in debug mode
- `cargo build --release` - Build optimized release version
- `cargo run` - Run the application in debug mode
- `cargo run --release` - Run the optimized release version

### Testing
- `cargo test` - Run all tests
- `cargo test test_multithreaded_terrain_generation` - Run a specific test
- `cargo test terrain::tests` - Run all tests in terrain module
- `cargo test -- --nocapture` - Run tests with stdout output
- `cargo test -- --exact` - Run tests with exact name matching

### Linting and Formatting
- `cargo fmt` - Format code according to Rust standards
- `cargo clippy` - Run Rust linter with all checks
- `cargo clippy -- -D warnings` - Treat warnings as errors
- `cargo check` - Quick compile check without building
- `cargo fix --edition-idioms` - Automatically fix edition-related issues

### Profiling
- `cargo run --features profiling` - Run with profiling enabled (if profiling feature is added)
- `cargo flamegraph --bin my-test` - Generate flamegraph for performance analysis

## Code Style Guidelines

### General Rust Style
1. Use `rustfmt` for consistent formatting
2. Follow Rust naming conventions:
   - Functions and variables: `snake_case`
   - Types and structs: `PascalCase`
   - Constants: `SCREAMING_SNAKE_CASE`
3. Use `#[derive(Debug, Clone, PartialEq)]` for public structs where appropriate
4. Use `#[instrument(skip(...))]` from tracing for performance-critical functions

### Imports Organization
1. Group imports in this order:
   - Standard library imports
   - External crate imports (bevy, noise, rayon, tracing)
   - Local module imports (crate::*)
2. Use `use bevy::prelude::*;` for common Bevy components
3. Avoid wildcard imports except for `prelude` modules

### Error Handling
1. Use `Result<T, E>` for fallible operations
2. Use `?` operator for error propagation
3. Implement `thiserror` or `anyhow` for custom error types (if added)
4. Use `expect()` with descriptive messages for unavoidable panics
5. Prefer `Option<T>` over nullable values

### Documentation
1. Add module-level documentation using `///` comments
2. Document public APIs with examples where helpful
3. Use `#[derive(Clone, Copy, Debug, PartialEq)]` for value types
4. Include field-level documentation for complex structures

### Performance Considerations
1. Use `#[inline]` for small, performance-critical functions
2. Prefer `Vec<T>` over `Box<[T]>` for growable collections
3. Use `&str` instead of `String` for function parameters when possible
4. Leverage Rayon for parallel processing of independent data
5. Use Bevy's query system efficiently - batch queries when possible

### Bevy-Specific Guidelines
1. Organize systems into logical groups in `App::new()`
2. Use appropriate system ordering with `.before()` and `.after()`
3. Use `Commands` for entity spawning and despawning
4. Implement `Component` for game data
5. Implement `Resource` for global state
6. Use `Query` for accessing component data
7. Use `Event` for communication between systems
8. Use `Plugin` pattern for modular functionality

### Voxel System Specific
1. Chunk size is defined by `CHUNK_SIZE` constant
2. Use `VoxelPosition` for 3D integer coordinates
3. Prefer batch operations on chunks over individual voxel operations
4. Use `VoxelWorld` resource for world state management
5. Implement greedy meshing for performance optimization

### Testing Guidelines
1. Unit tests should be in the same module or in a `*_tests.rs` file
2. Integration tests go in the `tests/` directory
3. Use `#[cfg(test)]` for test-specific code
4. Mock Bevy resources when needed for testing
5. Test voxel operations with edge cases (chunk boundaries, etc.)

## Key Files and Modules

### Core Structure
- `src/main.rs` - Application entry point and main systems
- `src/voxel.rs` - Core voxel data structures and world management
- `src/voxel_mesh.rs` - Mesh generation and optimization
- `src/terrain/` - Terrain generation modules
  - `terrain.rs` - Terrain generation implementation
  - `terrain_tests.rs` - Terrain-related tests
- `src/parallel_terrain.rs` - Multithreaded terrain generation
- `src/profiling.rs` - Performance profiling utilities

### Dependencies
- `bevy` - Game engine (version 0.15 with bevy_remote feature)
- `noise` - Perlin noise for terrain generation
- `rayon` - Parallel processing
- `tracing` - Structured logging and profiling
- `tracing-subscriber` - Log formatting and filtering

## Development Workflow

1. Make changes to source code
2. Run `cargo fmt` to format code
3. Run `cargo clippy` to check for issues
4. Run `cargo test` to verify tests pass
5. Run `cargo run` to test manually
6. Commit changes with descriptive messages

## Rust Version Compatibility

This project targets Rust 2021 edition and requires:
- Rust 1.70+ for async/await syntax improvements
- Rust 1.65+ for GATs (Generic Associated Types) support in Bevy

## Additional Development Tools

### Debugging
- Use `println!` or `dbg!` for quick debugging
- Use `tracing::info!`, `tracing::debug!`, etc. for structured logging
- Enable RUST_LOG environment variable for log filtering: `RUST_LOG=debug cargo run`

### IDE Support
- Configure rust-analyzer for VS Code or similar IDEs
- Use `cargo doc --open` to generate and view documentation
- Use `cargo expand` to view macro expansions (requires cargo-expand crate)

## Remote Debugging

The application includes Bevy Remote Protocol support:
- Default port: 15702
- Use the BRP tools for runtime inspection and debugging
- Entity inspection, component mutation, and event triggering are supported

## Common Patterns

### Module Organization
- Group related functionality in modules
- Use `mod.rs` for module organization with subdirectories
- Re-export public types with `pub use`
- Keep test modules in `*_tests.rs` files or inside `#[cfg(test)]`

### Threading and Performance
- Use Rayon's `par_iter()` for CPU-bound parallel work
- Bevy systems already run in parallel where possible
- Use `Commands` deferred operations for performance-critical code
- Consider using `bevy::ecs::system::Local` for system-local state