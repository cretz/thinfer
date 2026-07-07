# thinfer

thinfer runs image, video, and face-swap generation models on consumer GPUs,
streaming weights to stay inside tight RAM and VRAM budgets instead of requiring
the whole model resident.

Requires [Rust](https://rustup.rs/). Model weights download from Hugging Face on
first use: you are asked to confirm, or pass `--download-as-needed` to skip the
prompt.

## CLI

Install the `thinfer` binary (lands in `~/.cargo/bin`, already on PATH for Rust
installs):

    cargo install --path projects/thinfer-cli

Each modality is a `thinfer generate` subcommand; run `--help` on any of them for
the full option set.

### Image

Defaults to `qwen-image-rapid`, a 4-step distilled model with strong prompt
adherence. Its authored resolution is 1024x1024; 512x512 renders faster and still
looks great:

    thinfer generate image --prompt "a dragon made of stained glass" --width 512 --height 512 --output dragon.png

_~2 min on a weak RTX 5070 laptop GPU (~5 min at the default 1024x1024)._

### Video

Defaults to `fastwan-ti2v-5b`, a fast 5B distill at 960x544. `--duration` sets the
clip length in seconds:

    thinfer generate video --prompt "a red fox trotting through falling snow at dawn" --duration 2 --output fox.mp4

_~2.5 min on a weak RTX 5070 laptop GPU._

### Face swap

Swaps the face from a source image into every frame of a video (HyperSwap):

    thinfer generate face-swap --input-video clip.mp4 --source-image face.png --output swapped.mp4

_~10 s for a short 1080p clip on a weak RTX 5070 laptop GPU._

### Memory budgets

RAM and VRAM budgets both default to 2 GiB. The residency manager pages weights to
stay within them, so a small budget means more streaming, not failure. Raise them on
a card with more memory for less weight traffic (`--ram-budget` / `--vram-budget`
accept `8G`, `512M`, or raw bytes):

    thinfer generate image --prompt "..." --output out.png --vram-budget 6G --ram-budget 16G

## Web UI

To drive image, video, and face-swap generation from a browser, run the server:

    cargo install --path projects/thinfer-serve
    thinfer-serve

Then open http://localhost:8080. To serve over HTTPS with a self-signed certificate
(generated at startup), pass a config file containing `tls_self_signed = true`:

    thinfer-serve --config serve.toml

## About

thinfer was vibe coded: built largely through AI-assisted iteration. It works, but
it is a personal project, not a polished product, and may not be stable or high
quality. Expect rough edges.
