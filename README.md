# VoxelRS

A voxel-based game prototype built with Bevy.

## Local build prerequisites

Bevy's default audio stack pulls in `alsa-sys` on Linux, which requires ALSA development headers discoverable via `pkg-config`.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y pkg-config libasound2-dev
```

### Fedora

```bash
sudo dnf install -y pkgconf-pkg-config alsa-lib-devel
```

### Arch Linux

```bash
sudo pacman -S --needed pkgconf alsa-lib
```

## Build and test

```bash
cargo fmt
cargo check
cargo test
cargo run
```
