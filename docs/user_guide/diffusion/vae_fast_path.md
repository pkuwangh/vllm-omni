# Wan VAE Encoder and Decoder Fast Paths

Video pipelines built on the Wan 2.1/2.2 causal VAE (Cosmos3, Wan2.2, LingBot,
Helios, SANA-Video, LongCat-Video, DreamZero) spend a large share of their
end-to-end time decoding latents to pixels. The decoder is a chunked 3D
convolutional network whose reference implementation in diffusers moves every
activation through memory several times per layer (normalization, activation,
causal padding, feature-cache bookkeeping, shortcut upsampling). vLLM-Omni
installs a fast path on every loaded Wan VAE that fuses this data movement into
a handful of Triton kernels. The opt-in `channels_last` level also specializes
first-frame convolutions that have no history. Strict FP32 runs spatial convolutions, including first-frame shortcuts, as
one-plane native Conv3D with channels-first operands and channels-last output.
This avoids the oversized Conv2D FFT workspaces without width-specific rules.

## Decoder levels

The fast path is controlled by `--vae-fast-path` (engine argument
`vae_fast_path`, deploy-config key `vae_fast_path`):

| Level | Default | Output vs. diffusers | What it does |
| --- | --- | --- | --- |
| `lossless` | yes | bit-identical | Fused RMSNorm epilogue, fused causal-conv input assembly and cache refresh, fused shortcut upsampling and residual adds (with the neighbouring convolution biases folded in), single-pass nearest 2x upsampling, preallocated output assembly. |
| `channels_last` | no | within tolerance (PSNR typically > 60 dB) | Everything in `lossless`, plus channels-last convolution weights, a single-pass channels-last RMSNorm+SiLU kernel that also absorbs the preceding `conv1` bias, and Conv2d for supported single-frame causal convolutions without history. |
| `cudnn` | no | within model tolerance | Blackwell cuDNN Conv3D fused with normalization, SiLU, causal padding and history refresh at supported residual blocks. Uses channels-last for the remaining operations. |
| `off` | no | bit-identical | Reference diffusers decoder. |

```bash
# Default: bit-exact fast path
vllm serve nvidia/Cosmos3-Nano --omni

# Channels-last decoder
vllm serve nvidia/Cosmos3-Nano --omni --vae-fast-path channels_last

# Reference implementation, e.g. to bisect a quality issue
vllm serve nvidia/Cosmos3-Nano --omni --vae-fast-path off
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    vae_fast_path: channels_last
```

## Behavior and limitations

- Only CUDA and XPU are supported. On other platforms the reference decoder runs.
- The fast path is installed once per VAE instance when the pipeline is
  initialized. It rebinds the forwards of the loaded decoder modules; parameter
  names, `state_dict` keys and weight loading are unchanged.
- Installation is skipped, with a logged reason, if a custom forward wrapper
  or forward hook would be replaced or bypassed by the fused paths. Wrappers
  and hooks on modules that the fast path still calls normally are preserved.
- The Triton fast paths fall back to PyTorch for unsupported inputs, autograd
  and `torch.compile` tracing. The `cudnn` level uses registered custom operators
  during compilation; its supported hardware and dtypes are listed below.
- The SiLU activation is folded into the normalization kernel only after an
  exhaustive self-test over all bf16/fp16 values proves the fused epilogue
  bit-identical to `F.silu` on the running toolkit; otherwise SiLU stays a
  separate operation. The startup log reports `fused_silu=...`.
- Spatially sharded VAE decode (`--vae-parallel-mode spatial_shard_height` or
  `spatial_shard_width`) is not combined with the fast path; the installer skips
  the VAE and logs why. Tiled and tile-parallel decode work unchanged.
- The `channels_last` level changes the order in which cuDNN accumulates
  convolutions, so outputs differ from the reference in the last bits. Use
  `lossless` when bitwise reproducibility against diffusers matters.
- `channels_last` also skips internal decoder bitwise padding checks to reduce
  first-decode peak memory. Lossless decoding retains all of its checks.
- For a supported causal convolution with one input frame and no history,
  the temporal input is `[0, 0, x]`. The `channels_last` level convolves
  only the last temporal weight slice, preserving the convolution formula for
  finite inputs and the existing cache update. Strict FP32 uses one-plane Conv3D;
  other arithmetic modes use Conv2D. Calls with history retain full Conv3D. BF16 convolutions with 512 or more output channels and Wan 2.1 latent input (C16→C384) also retain
  Conv3d because Conv2d rounding errors amplify through decoding. The weight slice is taken
  on each call, so weight updates and device
  moves do not leave a stale packed copy.
- For VAEs kept in fp32, cuDNN's channels-last convolution algorithms run in
  TF32 under PyTorch's default `torch.backends.cudnn.allow_tf32 = True`, which
  dominates the difference to the reference (about 1e-3). Set
  `torch.backends.cudnn.allow_tf32 = False` if full fp32 convolutions are
  required.

## Cosmos3 encoder

`--vae-encode-fast-path` selects the encoder level independently of the decoder.
The engine/deploy-config field is `vae_encode_fast_path`. The supported target
is Cosmos3's residual Wan2.2 VAE: 2x2 patchification, 4x temporal compression,
16x spatial compression, and four residual downsampling stages. The real
Cosmos3 encoder uses 160/320/640 channels and produces 48 latent channels.
The `cudnn` level also supports the non-residual Wan 2.1 encoder. Other architectures retain their reference encoding path.

| Encoder level | Default | Behavior |
| --- | --- | --- |
| `lossless` | yes | Reuses the exact normalization epilogue, causal input/cache assembly and bias/residual fusions. Fuses spatial pad/input assembly and temporal downsampling cache copies. Keeps the reference shortcut averaging reduction. |
| `channels_last` | no | Also converts encoder convolution weights to channels-last, uses single-pass normalization and fuses shortcut averaging with residual addition. Outputs can differ in the last bits. |
| `cudnn` | no | cuDNN residual-block fusions for Wan 2.1 and Wan 2.2 on Blackwell, with channels-last convolution weights. |
| `off` | no | Reference encoding. |

For eligible cached encoder convolutions, normalization/SiLU writes directly
into the temporally assembled convolution input and the next two-frame cache.
Already-normalized history is copied unchanged; missing history stays zero.
`lossless` retains ATen's reduction and the existing intermediate rounding,
and only folds SiLU for dtypes that pass the installation-time exactness probe.
`channels_last` retains its existing single-pass normalization math, except
that for BF16 the encoder keeps the reference's intermediate roundings (the
`channels_last` decoder keeps fast math; `cudnn` preserves BF16 roundings); this fusion does not change its quality gates. Unsupported layouts, non-identity or hooked
dropout, and rejected spatial-padding probes use the existing separate path.
This Triton fusion is encoder-only. The `cudnn` level also fuses decoder stages.

Untiled encoding patchifies one temporal chunk at a time and writes encoder
features into a preallocated buffer. The schedule remains one initial frame,
then four frames per chunk. `quant_conv` runs once after assembly. Tiled
encoding retains its existing per-chunk `quant_conv`, tile coordinates and
blending. Both image conditioning and video conditioning use this path through
the ordinary `vae.encode(...).latent_dist` interface.

```bash
# Fast encoder, reference decoder
vllm serve nvidia/Cosmos3-Nano --omni \
    --vae-encode-fast-path channels_last --vae-fast-path off

# Reference encoder and decoder
vllm serve nvidia/Cosmos3-Nano --omni \
    --vae-encode-fast-path off --vae-fast-path off
```

```yaml
stages:
  - stage_id: 0
    vae_encode_fast_path: lossless
    vae_fast_path: channels_last
```

Encoder installation has independent reports and rollback/uninstall state.
It preserves weights and parameter names, and rejects custom forwards or hooks
that a fusion would bypass. A substituted `RMSNormVAE` keeps its own forward
and numerical semantics. Autograd uses PyTorch fallbacks. During compilation,
`cudnn` preserves its library calls as custom operators; the other levels use
PyTorch fallbacks. The
encoder can coexist with spatial-sharded decoding, since the two installers
modify different modules; spatial tiling remains the encoder's parallel mode.

The Python entry points are `install_wan_vae_encoder_fastpath`,
`is_encoder_installed`, and `uninstall_wan_vae_encoder_fastpath`, exported from
`vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath`. Uninstall restores
original forwards and tensor layouts while retaining current weight values.

### Encoder benchmarking

`benchmarks/diffusion/bench_wan_vae_encode.py` loads only the VAE. It compares all
encoder levels with the same weights, inputs and backend settings, always
setting up and validating against `off` first. It reports startup/first-call
time separately from warmed median latency, throughput and peak allocated
memory. Reference tensors are
kept on CPU during candidate timing. Console output includes timing and
per-tensor quality tables with explicit PASS/FAIL/OOM/NO REF statuses; speedups
are still shown when numerical validation fails. `--json` saves the full
metrics and per-level `errors` lists.

```bash
# Image conditioning
python benchmarks/diffusion/bench_wan_vae_encode.py \
    --model nvidia/Cosmos3-Nano --frames 1 --profile --json encode-image.json

# Video encoding and quality; --revision can pin the checkpoint
python benchmarks/diffusion/bench_wan_vae_encode.py \
    --model nvidia/Cosmos3-Nano --frames 93 --check-reconstruction --json encode-video.json

# A preprocessed natural image/video: floating RGB [B, 3, T, H, W] in [-1, 1]
python benchmarks/diffusion/bench_wan_vae_encode.py \
    --input pixels.pt --check-reconstruction --json encode-natural.json

# Two GPUs, existing tile-parallel encoding
torchrun --nproc-per-node 2 benchmarks/diffusion/bench_wan_vae_encode.py \
    --vae-patch-parallel-size 2 --frames 93 --json encode-tiled.json
```

The default is BF16, 1280x720, three warmups and ten timed iterations.
`--tiny` uses a reduced four-stage VAE for development; production performance
measurements must use real weights. `--tf32 off` controls FP32 convolution math.
The profiler reports CUDA operator times, including padding, copying,
normalization, reduction, attention and convolutions.

The default `--schedule sequential` warms and times one level at a time.
Consecutive samples let each implementation reach its own sustained power/clock
state. `--schedule interleaved` keeps all VAEs resident and rotates levels;
rotation balances positions but can carry predecessor power states into samples.
Peak memory excludes other levels' resident weights in that mode.
During timed encodes, NVML samples clocks, power, temperature and clock-event
reasons every `--telemetry-interval-ms` (default 50; 0 disables). The console
reports clock differences and throttling. `--lock-sm-clock-mhz` pins the SM
clock and resets it on exit; this requires administrator rights.

```bash
# Comparable timings on a power-limited GPU (clock lock needs root)
python benchmarks/diffusion/bench_wan_vae_encode.py \
    --model nvidia/Cosmos3-Nano --frames 93 --lock-sm-clock-mhz 1500 --json encode-locked.json
```

The benchmark exits unsuccessfully for a lossless bitwise mismatch, nonfinite
outputs, or channels-last/cuDNN normalized RMSE above 2% for BF16 and 1%
for FP32 posterior parameters, mean or log-variance. `--check-reconstruction` also requires at least 50 dB PSNR
between reconstructions of the reference and candidate mean, using the same
reference decoder. These are validation gates, not measured encoder results.
Single-GPU OOM cases are recorded explicitly; distributed failures terminate
the run because a failed collective cannot be retried safely on one rank.
Failure messages identify the level, tensor, measured metric and required
threshold. OOM messages identify the failing phase (such as timed encoding or
reference-decoder reconstruction), input shape and dtype, and retain the
original allocation error. If the reference fails, remaining timings are
marked NO REF rather than presented as validated results.

Validate on both H100 and GB200, using 256x256, 640x384 and 1280x720 inputs with
1, 5, 33, 93 and 189 frames. Record repeatable gains separately for images and
videos, and investigate latency or peak-memory regressions above 5%. The new
encoder kernel launch sizes are conservative defaults; architecture-specific
tuning and the numerical gates require validation on those GPUs.

## Measured decoder performance

Cosmos3-Nano VAE, 1280x720 x 189 frames, bf16, one GB200 GPU, `bench_wan_vae_decode.py`
(best of 2 runs after warmup):

| `--vae-fast-path` | Decode time | Speedup | Output vs. `off` |
| --- | ------------- | --------- | --- |
| `off` | 6.00 s | 1.00x | reference |
| `lossless` | 3.26 s | 1.84x | bit-identical |
| `channels_last` | 2.51 s | 2.39x | PSNR 62.6 dB, max abs diff 4.6e-2 |

Peak decode memory was unchanged (about 13 GiB): it is set by the largest
activations and the cuDNN workspace, not by the output assembly.

## Decoder benchmarking

`benchmarks/diffusion/bench_wan_vae_decode.py` decodes seeded latents with the
real VAE at each level and reports decode time, peak memory, bitwise equality
and PSNR against the `off` level. After `--warmup` decodes it times `--iters`
decodes and prints their median, min, max, standard deviation and every raw
sample; the speedup compares medians. `--tiling` decodes with the VAE's spatial
tiling on one GPU. `--profile` prints a per-kernel table with the convolution
share and any layout-transpose kernels:

```bash
python benchmarks/diffusion/bench_wan_vae_decode.py --model nvidia/Cosmos3-Nano \
    --size 1280x720 --frames 189 --fast-path off,lossless,channels_last --profile
```

For a single-GPU comparison that isolates first-frame Conv2d within
`channels_last`, use `--first-frame-ablation`. It alternates the two paths,
disables cuDNN TF32 and autotuning, and prints JSON with every timing sample,
summary statistics, peak allocated memory and output comparisons. Per-call
weight packing is included in the measured decode time.

```bash
python benchmarks/diffusion/bench_wan_vae_decode.py \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --size 832x480 --frames 1 \
    --dtype bf16 --first-frame-ablation --warmup 60 --iters 50 --seed 0
```

The default input is a seeded latent tensor. `--latents /path/to/latents.pt`
can instead load a saved VAE input after latent mean/std scaling; the benchmark
selects the temporal prefix requested by `--frames` and checks its shape.

The same script benchmarks multi-GPU VAE decode when launched with `torchrun`;
the decode is timed across all ranks and rank 0 reports:

```bash
torchrun --nproc-per-node 2 benchmarks/diffusion/bench_wan_vae_decode.py --model nvidia/Cosmos3-Nano \
    --vae-patch-parallel-size 2 --vae-parallel-mode tile
```

## cuDNN kernels on Blackwell

Select `--vae-fast-path cudnn` and/or `--vae-encode-fast-path cudnn` independently.
The default remains `lossless`. The optional provider requires SM100/SM103,
CUTLASS DSL 4.9 (including 4.9.0a0) or newer, and a cuDNN frontend build containing the Wan 2.1/TF32
Conv3D post-ops extension (development revision `df4c5671`). The existing
`cudnn-bsa` extra alone does not supply this extension.

BF16 uses custom convolution/post-operation fusions where supported. FP32
with `torch.backends.cudnn.allow_tf32=True` uses explicitly permitted TF32
convolutions. With TF32 disabled, convolution remains in native cuDNN and
FP32 normalization/SiLU/padding uses the library post-operation kernel.
Unsupported channel pairs keep native convolution; FP16 and other GPU
architectures decline installation with a reason.

Wan 2.2 encoding uses the library causal C12 input convolution and fuses
across residual blocks within each downsampling
stage. Wan 2.1 encoding groups its flat residual/downsample sequence the same
way. A convolution can prepare the next block's normalized input, and the
last convolution writes the spatial downsampler's padded input. C640 stages
combine deferred bias/residual additions with the next normalization. C640
TF32 uses native convolution; BF16 retains the custom convolution. Decoder
upsampling stages in both Wan versions carry prepared inputs across residual
blocks. TF32
fuses C160 normalization while C160 residual preparation and C320 preparation
use separate post-operation kernels; C96/C192 TF32 residual preparation
also remains separate, and C256 TF32 remains native. The decoder
uses explicit native input padding and a library-fused output normalization,
SiLU, padding and history update, matching the e2e demo. Remaining BF16 norms
preserve the reference intermediate roundings.

Feature histories remain in the ordinary Wan cache slots. The adapter fills
caller-owned history regions before using fused padded outputs. Packed weights
are nonpersistent derived buffers: `load_state_dict` and module device/dtype
moves invalidate them; ordinary parameters also detect in-place changes.
Inference tensors have no version counter, so update those weights through
`load_state_dict`. Custom operators preserve library calls under
`torch.compile`; lazy weight preparation is outside compiled regions.

Benchmark both implementations with matching precision:

```bash
python benchmarks/diffusion/bench_wan_vae_encode.py --model MODEL \
    --size 832x480 --frames 17 --dtype bf16 \
    --fast-path off,channels_last,cudnn --warmup 1 --iters 3 --json encode.json
python benchmarks/diffusion/bench_wan_vae_decode.py --model MODEL \
    --size 832x480 --frames 17 --dtype fp32 --tf32 on \
    --fast-path off,channels_last,cudnn --warmup 1 --iters 3 --json decode.json
```

Wan 2.1 encoding currently compares `off,cudnn`; the existing `channels_last`
encoder installer targets the residual Wan 2.2 architecture. Decoder validation
for `cudnn` checks finite output and PSNR >= 50 dB against the matching reference;
strict FP32 additionally checks `atol=rtol=1e-4`. Startup and packed-weight
memory are recorded separately from steady decode time.

For image-conditioned generation, approximate encoding can change the denoising
trajectory. Keep `--vae-encode-fast-path lossless` and select the decoder level
independently when preserving the reference conditioning is important.
