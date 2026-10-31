# thinfer

Image, video and face-swap generation on consumer GPUs. Weights stream in, so
models far bigger than your VRAM still run.

## Setup

Install [Rust](https://rustup.rs/), then:

    cargo install --path projects/thinfer-cli

NVIDIA GPU: also install the [CUDA 13 toolkit](https://developer.nvidia.com/cuda-downloads)
for the fast path (H3 video needs it to be practical).

Weights download from Hugging Face on first use. Add `--download-as-needed` to skip
the confirmation.

## Pick a model

    thinfer models

lists every model. Pass one with `--model <id>`. Defaults: image `zimage-turbo-q4`,
video `fastwan-ti2v-5b`.

`qwen-image-rapid`, `qwen-image-edit-rapid` and `ltx2-rapid` ship no default
weights: name yours in `thinfer-sources.toml` in your Hugging Face cache dir (or
point `THINFER_SOURCES` at it):

    [slots."qwen-image-rapid/dit-q4_k_m"]
    repo = "owner/repo"
    path = "file.gguf"

## Make an image

    thinfer generate image --prompt "a dragon made of stained glass" --output dragon.png

## Make a video

    thinfer generate video --prompt "a red fox trotting through falling snow" --output fox.mp4

## Make a video with sound

    thinfer generate video --model fasth3-trim --prompt "a woman by a window says: good morning" --output talk.mp4

5s at 832x480. ~33 GB download. Fastest on RTX 50-series, which decode with a
fast NVFP4 decoder by default (`--vae full`: original decoder; `--vae
nvfp4-fast`: faster, softer).

## Make a video from reference images

    thinfer generate video --model fasth3-omni --ref-image me.png --prompt "the man from <Picture 1> walks on a beach" --output beach.mp4

Name each `--ref-image` in the prompt as `<Picture 1>`, `<Picture 2>`, ... (up to 9).

## Use a LoRA

    thinfer vault add --model fasth3-trim --url <safetensors url>
    thinfer generate video --model fasth3-trim --lora <name>:0.8 --prompt "..." --output out.mp4

Adapters are stored encrypted, per model. `thinfer vault list --model <model>`
shows them.

## Swap a face into a video

    thinfer generate face-swap --input-video clip.mp4 --source-image face.png --output swapped.mp4

## Web UI

    cargo install --path projects/thinfer-serve
    thinfer-serve --config serve.toml

`serve.toml` containing `tls_self_signed = true` gives HTTPS, required when opening
it from another device. Open https://localhost:8080.

## Options

`--help` on any command lists everything. Common ones: `--model`, `--width`,
`--height`, `--duration`, `--seed`, and `--vram-budget` / `--ram-budget` (default
2G each; raise them on bigger machines, e.g. `--vram-budget 6G`).

## About

Vibe coded: built largely through AI-assisted iteration. A personal project, not a
polished product. Expect rough edges.
