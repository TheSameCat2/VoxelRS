/// Performance profiling utilities for the voxel game.
///
/// This module provides structured profiling capabilities using the `tracing` crate.
/// It includes helper functions for timing operations, measuring performance metrics,
/// and logging detailed information about system performance.
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument};

/// A profiler that measures execution time and logs performance metrics.
pub struct Profiler {
    name: String,
    start_time: Instant,
}

impl Drop for Profiler {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed();
        debug!(
            name = %self.name,
            elapsed_ms = elapsed.as_millis(),
            "Operation completed (dropped)"
        );
    }
}

impl Profiler {
    /// Creates a new profiler for the given operation.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the operation being profiled
    ///
    /// # Returns
    ///
    /// A new Profiler instance
    pub fn new(name: &str) -> Self {
        debug!(name = %name, "Starting operation");
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
        }
    }

    /// Finishes profiling and logs the elapsed time.
    pub fn finish(&self) {}

    /// Logs a metric for the current operation.
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Name of the metric
    /// * `value` - Metric value
    pub fn metric(&self, metric_name: &str, value: impl std::fmt::Display) {
        debug!(
            name = %self.name,
            metric = metric_name,
            value = %value,
            "Performance metric"
        );
    }
}

/// Macro for profiling a function or block of code.
///
/// This macro creates a profiler and automatically logs the elapsed time
/// when the scope ends.
///
/// # Examples
///
/// ```rust
/// profile_scope!("terrain_generation");
/// // ... code to profile ...
/// ```
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _profiler = $crate::profiling::Profiler::new($name);
    };
}

/// Macro for profiling with additional metrics.
///
/// This macro creates a profiler and allows adding metrics during execution.
///
/// # Examples
///
/// ```rust
/// profile_scope_with_metrics!("mesh_generation", {
///     profiler.metric("vertices", vertex_count);
///     profiler.metric("triangles", triangle_count);
/// });
/// ```
#[macro_export]
macro_rules! profile_scope_with_metrics {
    ($name:expr, $block:block) => {
        let profiler = $crate::profiling::Profiler::new($name);
        let profiler = &profiler;
        $block
    };
}

/// Instruments terrain generation operations.
pub fn profile_terrain_generation(
    operation: &str,
    world_size: i32,
    height: i32,
) -> impl FnOnce() -> Duration + use<'_> {
    let start = Instant::now();
    info!(
        operation = operation,
        world_size = world_size,
        height = height,
        "Starting terrain generation"
    );

    move || {
        let elapsed = start.elapsed();
        info!(
            operation = operation,
            world_size = world_size,
            height = height,
            elapsed_ms = elapsed.as_millis(),
            "Terrain generation completed"
        );
        elapsed
    }
}

/// Instruments mesh generation operations.
#[instrument(skip(config))]
pub fn profile_mesh_generation(
    chunk_pos: &crate::voxel::ChunkPosition,
    _chunk: &crate::voxel::VoxelChunk,
    _world: &crate::voxel::VoxelWorld,
    config: &crate::terrain::TerrainConfig,
) {
    debug!(
        chunk_x = chunk_pos.x,
        chunk_y = chunk_pos.y,
        chunk_z = chunk_pos.z,
        greedy_meshing = config.enable_greedy_meshing,
        "Starting mesh generation"
    );
}

/// Logs mesh generation results.
pub fn log_mesh_results(
    chunk_pos: &crate::voxel::ChunkPosition,
    vertex_count: usize,
    triangle_count: usize,
    elapsed: Duration,
) {
    info!(
        chunk_x = chunk_pos.x,
        chunk_y = chunk_pos.y,
        chunk_z = chunk_pos.z,
        vertex_count = vertex_count,
        triangle_count = triangle_count,
        elapsed_ms = elapsed.as_millis(),
        "Mesh generation completed"
    );
}

/// Profiles chunk operations.
pub fn profile_chunk_operation(operation: &str, chunk_count: usize) -> Profiler {
    debug!(
        operation = operation,
        chunk_count = chunk_count,
        "Starting chunk operation"
    );
    Profiler::new(operation)
}

/// Initializes the tracing subscriber for profiling.
pub fn init_profiling() {
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    // Default to info level to reduce noise
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("my_test=info,warn"));

    // Set up file appender
    let file_appender = tracing_appender::rolling::never(".", "profiling.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // Keep the guard alive to ensure logs are flushed
    // Note: In a real app, you might want to return this guard to main
    // so it lives for the duration of the program.
    // For this implementation, we'll leak it to keep it simple as it's a global logger.
    Box::leak(Box::new(_guard));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stdout).compact())
        .with(
            fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .compact(),
        )
        .init();

    info!("Profiling system initialized and logging to profiling.log");
}

/// Performance metrics collection.
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub terrain_generation_time: Duration,
    pub mesh_generation_time: Duration,
    pub chunk_updates: usize,
    pub vertices_generated: usize,
    pub triangles_generated: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn log_summary(&self) {
        info!(
            terrain_gen_ms = self.terrain_generation_time.as_millis(),
            mesh_gen_ms = self.mesh_generation_time.as_millis(),
            chunk_updates = self.chunk_updates,
            vertices = self.vertices_generated,
            triangles = self.triangles_generated,
            "Performance metrics summary"
        );
    }
}
