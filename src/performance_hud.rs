/// Performance HUD module for the voxel game.
///
/// This module provides a performance monitoring heads-up display (HUD)
/// that shows real-time performance metrics for the voxel game engine.
/// It includes FPS counter, terrain generation metrics, mesh statistics,
/// and other critical performance indicators.
use bevy::{
    app::{App, Plugin, Startup, Update},
    ecs::component::Component,
    ecs::system::{Commands, Query, Res, ResMut},
    hierarchy::ChildBuild,
    input::ButtonInput,
    prelude::{
        default, AlignItems, BuildChildren, Color, Display, FlexDirection, JustifyContent, KeyCode,
        Node, Resource, Text, TextFont, TextLayout, Time, Timer, TimerMode, With,
    },
    text::TextColor,
    time::{Real, Virtual},
    ui::{prelude::*, BackgroundColor, PositionType, Val},
};
use std::{fmt::Write, time::Instant};
use tracing::{debug, info};

use crate::{profiling::PerformanceMetrics, voxel::VoxelWorld};

/// Plugin for the performance HUD system.
pub struct PerformanceHudPlugin;

impl Plugin for PerformanceHudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HudState>()
            .init_resource::<HudPerformanceData>()
            .insert_resource(HudUpdateTimer::new())
            .insert_resource(PerformanceMetrics::default())
            .add_systems(Startup, setup_hud)
            .add_systems(
                Update,
                (toggle_hud, collect_performance_metrics, update_hud_display),
            );
    }
}

/// Resource to control HUD visibility and state.
#[derive(Resource, Debug)]
pub struct HudState {
    /// Whether the HUD is currently visible
    pub visible: bool,
    /// Whether the HUD has been initialized
    pub initialized: bool,
}

impl Default for HudState {
    fn default() -> Self {
        Self {
            visible: true,
            initialized: false,
        }
    }
}

/// Resource to store collected performance metrics.
#[derive(Resource, Debug)]
pub struct HudPerformanceData {
    /// Frames per second
    pub fps: f32,
    /// Frame time in milliseconds
    pub frame_time_ms: f32,
    /// Terrain generation time in milliseconds
    pub terrain_gen_time_ms: f32,
    /// Mesh generation time in milliseconds
    pub mesh_gen_time_ms: f32,
    /// Number of visible chunks
    pub chunks_visible: usize,
    /// Number of dirty chunks
    pub chunks_dirty: usize,
    /// Number of vertices generated
    pub vertices: usize,
    /// Number of triangles generated
    pub triangles: usize,
    /// Last time metrics were updated
    pub last_update: Instant,
    /// Frame count for FPS calculation
    pub frame_count: u64,
    /// FPS calculation start time
    pub fps_start_time: Instant,
    /// Number of chunks in memory
    pub total_chunks: usize,
    /// Number of entities in the world
    pub entity_count: usize,
    /// Memory usage estimate in MB
    pub memory_usage_mb: f64,
}

impl Default for HudPerformanceData {
    fn default() -> Self {
        Self {
            fps: 0.0,
            frame_time_ms: 0.0,
            terrain_gen_time_ms: 0.0,
            mesh_gen_time_ms: 0.0,
            chunks_visible: 0,
            chunks_dirty: 0,
            vertices: 0,
            triangles: 0,
            last_update: Instant::now(),
            frame_count: 0,
            fps_start_time: Instant::now(),
            total_chunks: 0,
            entity_count: 0,
            memory_usage_mb: 0.0,
        }
    }
}

impl HudPerformanceData {
    /// Update FPS calculation.
    pub fn update_fps(&mut self) {
        self.frame_count += 1;
        let elapsed = self.fps_start_time.elapsed();
        if elapsed.as_secs() >= 1 {
            self.fps = self.frame_count as f32 / elapsed.as_secs_f32();
            self.frame_count = 0;
            self.fps_start_time = Instant::now();
        }
    }

    /// Get color for a metric based on its value.
    pub fn get_metric_color(&self, value: f32, good: f32, warning: f32) -> Color {
        if value <= good {
            Color::srgb(0.0, 1.0, 0.0) // Green
        } else if value <= warning {
            Color::srgb(1.0, 1.0, 0.0) // Yellow
        } else {
            Color::srgb(1.0, 0.0, 0.0) // Red
        }
    }
}

/// Component to mark HUD container entities.
#[derive(Component)]
pub struct PerformanceHudContainer;

/// Component to mark HUD text entities.
#[derive(Component)]
pub struct PerformanceHudText;

/// Resource to control HUD update frequency.
#[derive(Resource, Debug)]
pub struct HudUpdateTimer {
    pub timer: Timer,
}

impl HudUpdateTimer {
    /// Create a new HUD update timer.
    pub fn new() -> Self {
        Self {
            timer: Timer::from_seconds(0.5, TimerMode::Repeating),
        }
    }
}

/// Setup the performance HUD UI elements.
pub fn setup_hud(
    mut commands: Commands,
    asset_server: Res<bevy::asset::AssetServer>,
    hud_state: Res<HudState>,
) {
    debug!("Setting up performance HUD");

    // Load a font
    let font_handle = asset_server.load("built-in/fonts/FiraSans-Bold.ttf");

    // Create the main HUD container
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                min_width: Val::Px(200.0),
                min_height: Val::Px(150.0),
                display: if hud_state.visible {
                    Display::Flex
                } else {
                    Display::None
                },
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)), // Semi-transparent background
            PerformanceHudContainer,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Initializing HUD..."),
                TextFont {
                    font: font_handle,
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                TextLayout::new_with_justify(bevy::text::JustifyText::Left),
                PerformanceHudText,
            ));
        });

    info!("Performance HUD setup complete");
}

/// Toggle HUD visibility with F3 key.
pub fn toggle_hud(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut hud_state: ResMut<HudState>,
    mut query: Query<&mut Node, With<PerformanceHudContainer>>,
) {
    if keyboard.just_pressed(KeyCode::F3) {
        hud_state.visible = !hud_state.visible;

        info!("Performance HUD toggled: {}", hud_state.visible);

        if let Ok(mut node) = query.get_single_mut() {
            node.display = if hud_state.visible {
                Display::Flex
            } else {
                Display::None
            };
        }
    }
}

/// Collect performance metrics from various game systems.
pub fn collect_performance_metrics(
    mut performance_data: ResMut<HudPerformanceData>,
    world_resource: Res<VoxelWorld>,
    time: Res<Time<Real>>,
    mut metrics: ResMut<PerformanceMetrics>,
) {
    // Update FPS
    performance_data.update_fps();
    performance_data.frame_time_ms = time.delta_secs() * 1000.0;

    // Collect metrics from profiling system
    performance_data.terrain_gen_time_ms = metrics.terrain_gen_time_ms();
    performance_data.mesh_gen_time_ms = metrics.mesh_gen_time_ms();
    
    // Accumulate mesh stats from this frame
    performance_data.vertices += metrics.vertices_generated;
    performance_data.triangles += metrics.triangles_generated;
    
    // Reset frame-based metrics
    metrics.vertices_generated = 0;
    metrics.triangles_generated = 0;

    // Collect chunk metrics (using public methods)
    performance_data.total_chunks = world_resource.chunk_count();
    performance_data.chunks_visible = world_resource.visible_chunk_count();
    performance_data.chunks_dirty = world_resource.dirty_chunk_count();

    // Calculate memory usage (rough estimate)
    // Each voxel is ~4 bytes (type) + 4 bytes (color) = 8 bytes
    const BYTES_PER_VOXEL: usize = 8;
    const VOXELS_PER_CHUNK: usize = 32 * 32 * 32; // 32x32x32 chunks
    let total_voxels = performance_data.total_chunks * VOXELS_PER_CHUNK;
    performance_data.memory_usage_mb = (total_voxels * BYTES_PER_VOXEL) as f64 / 1024.0 / 1024.0;

    // Update timestamp
    performance_data.last_update = Instant::now();
}

/// Update the HUD display with current performance metrics.
pub fn update_hud_display(
    mut query: Query<&mut Text, With<PerformanceHudText>>,
    time: Res<Time<Virtual>>,
    mut update_timer: ResMut<HudUpdateTimer>,
    mut performance_data: ResMut<HudPerformanceData>,
    hud_state: Res<HudState>,
) {
    // Only update if HUD is visible and timer has elapsed
    if !hud_state.visible {
        return;
    }

    update_timer.timer.tick(time.delta());
    if !update_timer.timer.just_finished() {
        return;
    }

    if let Ok(mut text) = query.get_single_mut() {
        let mut display_text = String::new();

        // Format the performance metrics
        let _ = writeln!(display_text, "=== Performance Monitor ===");

        // FPS and frame time
        let _ = writeln!(
            display_text,
            "FPS: {:.1} (Frame: {:.1}ms)",
            performance_data.fps, performance_data.frame_time_ms
        );

        // Terrain generation
        if performance_data.terrain_gen_time_ms > 0.0 {
            let _ = writeln!(
                display_text,
                "Terrain Gen: {:.1}ms",
                performance_data.terrain_gen_time_ms
            );
        }

        // Mesh generation
        if performance_data.mesh_gen_time_ms > 0.0 {
            let _ = writeln!(
                display_text,
                "Mesh Gen: {:.1}ms",
                performance_data.mesh_gen_time_ms
            );
        }

        // Chunk statistics
        let _ = writeln!(
            display_text,
            "Chunks: {}/{} visible, {} dirty",
            performance_data.chunks_visible,
            performance_data.total_chunks,
            performance_data.chunks_dirty
        );

        // Mesh statistics (Throughput)
        let _ = writeln!(
            display_text,
            "Gen (0.5s): {} verts, {} tris",
            performance_data.vertices, performance_data.triangles
        );
        
        // Reset accumulators for next interval
        performance_data.vertices = 0;
        performance_data.triangles = 0;

        // Memory usage
        let _ = writeln!(
            display_text,
            "Memory: {:.1} MB",
            performance_data.memory_usage_mb
        );

        *text = Text::new(display_text);
    }
}
