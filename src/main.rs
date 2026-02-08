use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, PrimaryWindow};
use tracing::instrument;

// Import our voxel system modules
mod parallel_terrain;
mod performance_hud;
mod profiling;
mod terrain;
mod voxel;
mod voxel_mesh;

use performance_hud::PerformanceHudPlugin;
use terrain::{
    generate_terrain_on_demand, initialize_terrain, manage_chunk_visibility, PlayerController,
    TerrainConfig,
};
use voxel::{Voxel, VoxelPosition, VoxelType, VoxelWorld};
use voxel_mesh::update_chunk_meshes;

fn main() {
    // Initialize profiling system
    profiling::init_profiling();

    println!("Starting voxel game...");

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Voxel TDM - Custom Voxel System".to_string(),
                        ..default()
                    }),
                    ..default()
                })
                .disable::<bevy::log::LogPlugin>(),
        )
        .add_plugins(PerformanceHudPlugin)
        .insert_resource(VoxelWorld::new())
        .insert_resource(TerrainConfig::default())
        .add_systems(Startup, (setup, initialize_terrain))
        .add_systems(
            Update,
            (
                move_camera,
                cursor_grab,
                player_voxel_interaction,
                generate_terrain_on_demand,
                update_chunk_meshes,
                manage_chunk_visibility,
            ),
        )
        .run();
}

#[derive(Component)]
struct CameraController {
    pub yaw: f32,
    pub pitch: f32,
    pub sensitivity: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            sensitivity: 0.002,
        }
    }
}

#[instrument(skip(commands))]
fn setup(mut commands: Commands) {
    // Add a camera with controller
    commands.spawn((
        Camera3d::default(),
        CameraController::default(),
        PlayerController, // Mark this as player
        Transform::from_xyz(0.0, 10.0, 20.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));

    // Add a directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, -0.5, 0.0)),
    ));

    // Add an ambient light for better visibility
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.3,
    });
}

fn move_camera(
    mut query: Query<(&mut Transform, &mut CameraController)>,
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    window_query: Query<&Window, With<PrimaryWindow>>,
) {
    let (mut transform, mut controller) = query.single_mut();
    let window = window_query.single();

    // Only rotate if cursor is locked
    if window.cursor_options.grab_mode == CursorGrabMode::Locked {
        for event in mouse_motion.read() {
            controller.yaw -= event.delta.x * controller.sensitivity;
            controller.pitch -= event.delta.y * controller.sensitivity;
            controller.pitch = controller.pitch.clamp(-1.54, 1.54); // Limit pitch to avoid flipping
        }
    }

    transform.rotation = Quat::from_euler(EulerRot::YXZ, controller.yaw, controller.pitch, 0.0);

    // Camera movement speed
    let speed = 20.0;
    let forward = transform.forward();
    let right = transform.right();
    let up = transform.up();

    let mut velocity = Vec3::ZERO;

    if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
        velocity += *forward;
    }
    if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
        velocity -= *forward;
    }
    if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
        velocity -= *right;
    }
    if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
        velocity += *right;
    }
    if keyboard.pressed(KeyCode::Space) {
        velocity += *up;
    }
    if keyboard.pressed(KeyCode::ShiftLeft) {
        velocity -= *up;
    }

    transform.translation += velocity.normalize_or_zero() * speed * time.delta_secs();
}

fn cursor_grab(
    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut window_query: Query<&mut Window, With<PrimaryWindow>>,
) {
    if let Ok(mut window) = window_query.get_single_mut() {
        if mouse_button.just_pressed(MouseButton::Left) {
            window.cursor_options.grab_mode = CursorGrabMode::Locked;
            window.cursor_options.visible = false;
        }
        if keyboard.just_pressed(KeyCode::Escape) {
            window.cursor_options.grab_mode = CursorGrabMode::None;
            window.cursor_options.visible = true;
        }
    }
}

fn player_voxel_interaction(
    mut world_resource: ResMut<VoxelWorld>,
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<&Transform, With<PlayerController>>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        // Simple voxel editing for testing
        // Get player position in voxel coordinates
        let player_pos = player_transform.translation;
        let voxel_pos = VoxelPosition::new(
            player_pos.x.floor() as i32,
            (player_pos.y - 1.0).floor() as i32, // Place/remove block below player
            player_pos.z.floor() as i32,
        );

        // Press 'B' to place a block at player position
        if keyboard.just_pressed(KeyCode::KeyB) {
            let block_voxel = Voxel::new(VoxelType::Solid, Color::srgb(1.0, 0.0, 0.0));
            world_resource.set_voxel(&voxel_pos, block_voxel);
        }

        // Press 'R' to remove a block at player position
        if keyboard.just_pressed(KeyCode::KeyR) {
            let air_voxel = Voxel::new(VoxelType::Air, Color::BLACK);
            world_resource.set_voxel(&voxel_pos, air_voxel);
        }

        // Press 'T' to add a test column starting at sea level
        if keyboard.just_pressed(KeyCode::KeyT) {
            for y in 0..15 {
                let pos = VoxelPosition::new(voxel_pos.x, y, voxel_pos.z);
                let voxel = Voxel::new(VoxelType::Solid, Color::srgb(0.8, 0.4, 0.2));
                world_resource.set_voxel(&pos, voxel);
            }
        }
    }
}
