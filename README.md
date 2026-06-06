# thinfer

Under development.

## Build

Requires [Rust](https://rustup.rs/).

    cargo install --path projects/thinfer-cli

Installs the `thinfer` binary into `~/.cargo/bin` (already on PATH for Rust installs).

## Generate an image

    thinfer generate image --prompt "a dragon made of stained glass" --output dragon.png

Downloads model weights on first run (asks first; pass `--download-as-needed` to skip the prompt). See `--help` for size, steps, seed, and memory-budget options.
