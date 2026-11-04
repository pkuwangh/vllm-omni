# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""cuDNN Conv3D/post-operation adapter for Blackwell Wan inference."""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Callable
from functools import lru_cache
from inspect import signature
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanCausalConv3d,
    WanDecoder3d,
    WanEncoder3d,
    WanResample,
    WanResidualBlock,
    WanResidualDownBlock,
    WanResidualUpBlock,
    WanUpBlock,
)
from packaging.version import Version

from vllm_omni.platforms import current_omni_platform

from . import encoder_forwards as ef
from . import forwards as fp
from . import triton_downsample as down

_CHANNELS = (96, 160, 192, 256, 320, 384, 512, 640, 1024)
_PAIRS = {
    (96, 96),
    (96, 192),
    (192, 192),
    (192, 384),
    (384, 384),
    (160, 160),
    (160, 320),
    (320, 320),
    (320, 640),
    (640, 640),
    (512, 256),
    (256, 256),
}
_FUSED = (96, 160, 192, 256, 320)


@lru_cache
def provider() -> ModuleType:
    import cudnn

    pack = getattr(cudnn, "pack_conv3d_weight_sm100", None)
    if pack is None or "allow_tf32" not in signature(pack).parameters:
        raise ImportError("Wan cuDNN fast path requires the Wan 2.1/TF32 Conv3D frontend extension")
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if Version(version).release[:2] < (4, 9):
        raise ImportError(f"Wan cuDNN fast path requires CUTLASS DSL >= 4.9 (including 4.9.0a0); found {version}")
    return cudnn


def availability(component: nn.Module) -> str | None:
    weight = next(component.parameters())
    if weight.device.type != "cuda":
        return "cuDNN Wan kernels require CUDA"
    capability = current_omni_platform.get_device_capability(weight.device.index)
    if capability is None or tuple(capability) not in ((10, 0), (10, 3)):
        return "cuDNN Wan kernels require SM100 or SM103"
    if weight.dtype not in (torch.bfloat16, torch.float32):
        return "cuDNN Wan kernels support BF16 and FP32"
    provider()
    if not any(type(m) is WanResidualBlock and supports_block(m) for m in component.modules()):
        return "no supported cuDNN Wan residual blocks"
    return None


class PackedWeight(nn.Module):
    """Derived, nonpersistent weights; source parameters remain authoritative."""

    def __init__(self, *, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.register_buffer("packed", None, persistent=False)
        self.key: tuple[int, int, int | None, torch.dtype, torch.device, bool] | None = None

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> PackedWeight:
        # Do not carry rounded/padded weights across offload or dtype conversion.
        self.packed = None
        self.key = None
        return self

    def invalidate(self, module: nn.Module, incompatible_keys: object) -> None:
        self.packed = None
        self.key = None

    @torch.compiler.disable
    def forward(self, weight: torch.Tensor, allow_tf32: bool) -> torch.Tensor:
        # Inference tensors use the load-state hook for invalidation; ordinary
        # parameters additionally track in-place updates through their version.
        version = None if weight.is_inference() else weight._version
        key = (id(weight), weight.data_ptr(), version, weight.dtype, weight.device, allow_tf32)
        if self.key != key:
            pack = provider().pack_causal_conv3d_weight_sm100 if self.causal else provider().pack_conv3d_weight_sm100
            self.packed = pack(weight, allow_tf32=allow_tf32)
            self.key = key
        return self.packed


def _input_conv(
    x: torch.Tensor, weight: torch.Tensor, previous: torch.Tensor | None, allow_tf32: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    result = provider().causal_conv3d_with_cache_wrapper_sm100(x, weight, previous, allow_tf32=allow_tf32)
    return result["output"], result["cache_output"]


input_conv = torch.library.custom_op("vllm_omni::wan_cudnn_input_conv", mutates_args=())(_input_conv)


@input_conv.register_fake
def _input_conv_fake(
    x: torch.Tensor, weight: torch.Tensor, previous: torch.Tensor | None, allow_tf32: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    n, c, t, h, w = x.shape
    history = 0 if previous is None else previous.shape[1]
    return x.new_empty((n, t, h, w, weight.shape[0])), x.new_empty((n, min(2, t + history), h, w, c))


def _convolve_input(
    conv: nn.Conv3d, x: torch.Tensor, previous: torch.Tensor | None, tf32: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv._wan_cudnn_weight(conv.weight, tf32)
    convolve = input_conv if torch.compiler.is_compiling() else _input_conv
    return convolve(x, weight, previous, tf32)


def supports_input(encoder: WanEncoder3d) -> bool:
    conv = encoder.conv_in
    return (
        (conv.in_channels, conv.out_channels) == (12, 160)
        and conv.kernel_size == (3, 3, 3)
        and conv.stride == conv.dilation == (1, 1, 1)
        and conv.groups == 1
        and tuple(conv._padding) == (1, 1, 1, 1, 2, 0)
    )


def encoder_forward(
    self: WanEncoder3d,
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    tf32 = x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or feat_cache is None
        or not (x.dtype == torch.bfloat16 or tf32)
        or fp._has_forward_hooks(self.conv_in)
    ):
        return ef.encoder_forward(self, x, feat_cache, feat_idx)
    if feat_idx is None:
        feat_idx = [0]
    slot = feat_idx[0]
    previous = feat_cache[slot]
    previous = None if previous is None else previous.permute(0, 2, 3, 4, 1).contiguous()
    x, history = _convolve_input(self.conv_in, x, previous, tf32)
    feat_cache[slot] = history.permute(0, 4, 1, 2, 3)
    feat_idx[0] += 1
    first = self.down_blocks[0]
    if getattr(first.forward, "__func__", None) is residual_down_block_forward and not fp._has_forward_hooks(first):
        x = first(x.permute(0, 4, 1, 2, 3), feat_cache=feat_cache, feat_idx=feat_idx, input_bias=self.conv_in.bias)
    else:
        x = first((x + self.conv_in.bias).permute(0, 4, 1, 2, 3), feat_cache=feat_cache, feat_idx=feat_idx)
    for block in self.down_blocks[1:]:
        x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
    x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
    x = fp._run_norm_act_cached_conv(self.norm_out, self.nonlinearity, self.conv_out, x, feat_cache, feat_idx[0])
    feat_idx[0] += 1
    return x


def flat_encoder_forward(
    self: WanEncoder3d,
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    """Carry preparation across Wan 2.1's flat residual/downsample sequence."""
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or feat_cache is None
        or x.dtype not in (torch.bfloat16, torch.float32)
        or _stage_has_forward_hooks(self.down_blocks)
    ):
        return ef.encoder_forward(self, x, feat_cache, feat_idx)
    if feat_idx is None:
        feat_idx = [0]
    tf32 = x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32
    x = fp._run_cached_causal_conv(self.conv_in, x, feat_cache, feat_idx[0])
    feat_idx[0] += 1
    blocks = self.down_blocks
    index = 0
    while index < len(blocks):
        group = []
        while index < len(blocks) and getattr(blocks[index].forward, "__func__", None) is residual_block_forward:
            group.append(blocks[index])
            index += 1
        if not group:
            x = blocks[index](x, feat_cache=feat_cache, feat_idx=feat_idx)
            index += 1
            continue
        downsampler = None
        if (
            index < len(blocks)
            and type(blocks[index]) is WanResample
            and getattr(blocks[index].forward, "__func__", None) is ef.downsample_forward
            and ef.is_spatial_downsample(blocks[index])
        ):
            downsampler = blocks[index]
            index += 1
        x, _ = _run_residual_chain(group, x, feat_cache, feat_idx, tf32, downsampler)
    x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
    x = fp._run_norm_act_cached_conv(self.norm_out, self.nonlinearity, self.conv_out, x, feat_cache, feat_idx[0])
    feat_idx[0] += 1
    return x


def _norm_pad(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    save_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    result = provider().rmsnorm_silu_pad_wrapper_sm100(
        x,
        gamma,
        input_bias=bias,
        previous=previous,
        residual=residual,
        residual_bias=residual_bias,
        save_input=save_input,
    )
    return result["padded_output"], result["cache_output"], result["residual_output"]


norm_pad = torch.library.custom_op(
    "vllm_omni::wan_cudnn_norm_pad",
    mutates_args=(),
    schema="(Tensor x, Tensor gamma, Tensor? bias, Tensor? previous, Tensor? residual=None, "
    "Tensor? residual_bias=None, bool save_input=False) -> (Tensor, Tensor, Tensor?)",
)(_norm_pad)


@norm_pad.register_fake
def _norm_pad_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    save_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n, t, h, w, c = x.shape
    history = 0 if previous is None else previous.shape[1]
    return (
        x.new_empty((n, t + 2, h + 2, w + 2, c)),
        x.new_empty((n, min(2, t + history), h, w, c)),
        torch.empty_like(x) if residual is not None or save_input else None,
    )


def _conv_norm_pad(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    allow_tf32: bool,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    history = 0 if previous is None else previous.shape[1]
    result = provider().conv3d_rmsnorm_silu_pad_wrapper_sm100(
        x,
        weight,
        bias,
        gamma,
        history_frames=history,
        residual=residual,
        residual_bias=residual_bias,
        allow_tf32=allow_tf32,
    )
    padded, cache = result["padded_output"], result["cache_output"]
    if previous is not None:
        padded[:, 2 - history : 2, 1:-1, 1:-1].copy_(previous)
        old = max(cache.shape[1] - (padded.shape[1] - 2), 0)
        if old:
            cache[:, :old].copy_(previous[:, -old:])
    return padded, cache, result["residual_output"]


conv_norm_pad = torch.library.custom_op(
    "vllm_omni::wan_cudnn_conv_norm_pad",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor bias, Tensor gamma, Tensor? previous, bool allow_tf32, "
    "Tensor? residual=None, Tensor? residual_bias=None) -> (Tensor, Tensor, Tensor?)",
)(_conv_norm_pad)


@conv_norm_pad.register_fake
def _conv_norm_pad_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    allow_tf32: bool,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n, t, h, w, _ = x.shape
    c = weight.shape[0]
    history = 0 if previous is None else previous.shape[1]
    return (
        x.new_empty((n, t, h, w, c)),
        x.new_empty((n, min(2, t - 2 + history), h - 2, w - 2, c)),
        x.new_empty((n, t - 2, h - 2, w - 2, c)) if residual is not None else None,
    )


def _conv_residual_pad(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor | None,
    allow_tf32: bool,
) -> torch.Tensor:
    return provider().conv3d_bias_residual_pad_wrapper_sm100(
        x, weight, bias, residual, residual_bias=residual_bias, allow_tf32=allow_tf32
    )["padded_output"]


conv_residual_pad = torch.library.custom_op("vllm_omni::wan_cudnn_conv_residual_pad", mutates_args=())(
    _conv_residual_pad
)


@conv_residual_pad.register_fake
def _conv_residual_pad_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor | None,
    allow_tf32: bool,
) -> torch.Tensor:
    n, t, h, w, _ = x.shape
    return x.new_empty((n, t - 2, h - 1, w - 1, weight.shape[0]))


def _raw_conv(x: torch.Tensor, weight: torch.Tensor, allow_tf32: bool) -> torch.Tensor:
    return provider().conv3d_raw_wrapper_sm100(x, weight, allow_tf32=allow_tf32)


raw_conv = torch.library.custom_op("vllm_omni::wan_cudnn_conv", mutates_args=())(_raw_conv)


@raw_conv.register_fake
def _raw_conv_fake(x: torch.Tensor, weight: torch.Tensor, allow_tf32: bool) -> torch.Tensor:
    n, t, h, w, _ = x.shape
    return x.new_empty((n, t - 2, h - 2, w - 2, weight.shape[0]))


def supports_block(block: WanResidualBlock) -> bool:
    return (
        fp._can_bypass_dropout(block.dropout)
        and type(block.nonlinearity) is nn.SiLU
        and all(
            fp.is_diffusers_rms_norm(norm)
            and isinstance(norm.bias, float)
            and norm.bias == 0.0
            and norm.scale == math.sqrt(norm.gamma.numel())
            and norm.gamma.numel() in _CHANNELS
            for norm in (block.norm1, block.norm2)
        )
        and all(
            conv.kernel_size == (3, 3, 3)
            and conv.stride == (1, 1, 1)
            and conv.dilation == (1, 1, 1)
            and conv.groups == 1
            and tuple(conv._padding) == (1, 1, 1, 1, 2, 0)
            for conv in (block.conv1, block.conv2)
        )
    )


def _custom_conv(conv: nn.Conv3d, x: torch.Tensor, tf32: bool, first_frame: bool = False) -> bool:
    # C384 BF16 accumulation differences amplify through the Wan 2.1 encoder.
    return (
        (conv.in_channels, conv.out_channels) in _PAIRS
        and (x.dtype == torch.bfloat16 or tf32)
        and not (x.dtype == torch.bfloat16 and conv.out_channels == 384)
        # Full-model controls on 1100 W B300 favor native C256/C640 TF32.
        and not (tf32 and conv.out_channels in (256, 640))
        # Wan 2.1 narrow/C256 BF16 and all TF32 residual chunks use native Conv2D.
        and not (first_frame and (tf32 or conv.out_channels in (96, 192, 256)))
    )


def _fused_conv(conv: nn.Conv3d, x: torch.Tensor, tf32: bool, first_frame: bool, *, residual: bool = False) -> bool:
    # TF32 C160 normalization uses the larger epilogue register budget.
    # Residual C96/C160/C192 and all C320 preparation favor separate post-ops.
    return (
        _custom_conv(conv, x, tf32, first_frame)
        and conv.out_channels in _FUSED
        and not (tf32 and (conv.out_channels == 320 or (conv.out_channels in (96, 160, 192) and residual)))
    )


def _convolve(conv: nn.Conv3d, x: torch.Tensor, tf32: bool, first_frame: bool = False) -> torch.Tensor:
    if first_frame and fp._use_first_frame_shortcut(conv, x.dtype) and not _custom_conv(conv, x, tf32, first_frame):
        # The first causal input is [0, 0, current]. Keep the native
        # first-frame shortcut instead of convolving two zero temporal planes.
        image = x[:, 2].permute(0, 3, 1, 2)
        weight = conv.weight[:, :, 2].contiguous(memory_format=torch.channels_last)
        return fp._spatial_convolution(image, weight).permute(0, 2, 3, 1).unsqueeze(1)
    if _custom_conv(conv, x, tf32, first_frame):
        weight = conv._wan_cudnn_weight(conv.weight, tf32)
        convolve = raw_conv if torch.compiler.is_compiling() else _raw_conv
        return convolve(x, weight, tf32)
    return F.conv3d(x.permute(0, 4, 1, 2, 3), conv.weight).permute(0, 2, 3, 4, 1)


def _prepare(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    save_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    prepare = norm_pad if torch.compiler.is_compiling() else _norm_pad
    return prepare(x, gamma, bias, previous, residual, residual_bias, save_input)


def _prepare_convolution(
    conv: nn.Conv3d,
    x: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    tf32: bool,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    weight = conv._wan_cudnn_weight(conv.weight, tf32)
    # Resolve dispatch after packing's graph break, in the frame that calls it.
    prepare = conv_norm_pad if torch.compiler.is_compiling() else _conv_norm_pad
    return prepare(x, weight, conv.bias, gamma, previous, tf32, residual, residual_bias)


def _prepare_spatial(
    conv: nn.Conv3d,
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor | None,
    tf32: bool,
) -> torch.Tensor:
    weight = conv._wan_cudnn_weight(conv.weight, tf32)
    prepare = conv_residual_pad if torch.compiler.is_compiling() else _conv_residual_pad
    return prepare(x, weight, conv.bias, residual, residual_bias, tf32)


def residual_block_forward(
    self: WanResidualBlock,
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or feat_cache is None
        or x.dtype not in (torch.bfloat16, torch.float32)
        or fp._has_forward_hooks(self.norm1, self.norm2, self.nonlinearity, self.dropout, self.conv1, self.conv2)
    ):
        return fp.residual_block_forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx)
    if feat_idx is None:
        feat_idx = [0]
    index = feat_idx[0]
    tf32 = x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32
    residual = self.conv_shortcut(x)
    previous = feat_cache[index]
    first1 = x.shape[2] == 1 and previous is None
    previous = None if previous is None else previous.permute(0, 2, 3, 4, 1).contiguous()
    # Eager inference avoids dispatcher overhead; compilation keeps opaque
    # library operators with explicit fake implementations and alias contracts.
    padded, cache, _ = _prepare(x.permute(0, 2, 3, 4, 1).contiguous(), self.norm1.gamma.reshape(-1), None, previous)
    feat_cache[index] = cache.permute(0, 4, 1, 2, 3)
    previous = feat_cache[index + 1]
    first2 = x.shape[2] == 1 and previous is None
    previous = None if previous is None else previous.permute(0, 2, 3, 4, 1).contiguous()
    if _fused_conv(self.conv1, x, tf32, first1):
        padded, cache, _ = _prepare_convolution(self.conv1, padded, self.norm2.gamma.reshape(-1), previous, tf32)
    else:
        raw = _convolve(self.conv1, padded, tf32, first1)
        padded, cache, _ = _prepare(raw.contiguous(), self.norm2.gamma.reshape(-1), self.conv1.bias, previous)
    feat_cache[index + 1] = cache.permute(0, 4, 1, 2, 3)
    feat_idx[0] += 2
    raw = _convolve(self.conv2, padded, tf32, first2)
    return fp._residual_add(raw.permute(0, 4, 1, 2, 3), self.conv2.bias, residual, None)


def supports_stage(stage: WanResidualDownBlock) -> bool:
    """Stages whose residual chain can stay in NTHWC between consumers."""
    return (
        all(type(block) is WanResidualBlock and supports_block(block) for block in stage.resnets)
        and all(type(block.conv_shortcut) is nn.Identity for block in stage.resnets[1:])
        and (stage.downsampler is None or ef.is_spatial_downsample(stage.downsampler))
    )


def _stage_has_forward_hooks(stage: nn.Module) -> bool:
    # Inspect the live tree so hooks installed after the fast path are honored.
    # These stages are trees; named_modules' path construction and deduplication
    # are unnecessary in this per-chunk check.
    if fp._has_forward_hooks():
        return True
    pending = list(stage._modules.values())
    while pending:
        module = pending.pop()
        if module is None:
            continue
        if module._forward_pre_hooks or module._forward_hooks:
            return True
        pending.extend(module._modules.values())
    return False


def residual_down_block_forward(
    self: WanResidualDownBlock,
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
    input_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Carry prepared activations and deferred residuals across encoder blocks.

    Match the standalone encoder: narrow conv2 epilogues prepare the next
    block directly; wide conv2 bias/residual is consumed by its next norm.
    The last convolution also writes the spatial downsampler's padded input.
    Histories remain owned tensors in the caller's NCTHW cache slots.
    """
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or feat_cache is None
        or x.dtype not in (torch.bfloat16, torch.float32)
        or _stage_has_forward_hooks(self)
    ):
        if input_bias is not None:
            x = (x.permute(0, 2, 3, 4, 1) + input_bias).permute(0, 4, 1, 2, 3)
        return ef.residual_down_block_forward(self, x, feat_cache, feat_idx)
    if feat_idx is None:
        feat_idx = [0]
    tf32 = x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32

    x, source = _run_residual_chain(self.resnets, x, feat_cache, feat_idx, tf32, self.downsampler, input_bias)
    shortcut = self.avg_shortcut
    out = down.avg_down3d_add(x, source, shortcut.factor_t, shortcut.factor_s, shortcut.group_size)
    return x + shortcut(source) if out is None else out


def _run_residual_chain(
    resnets: nn.ModuleList | list[WanResidualBlock],
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None],
    feat_idx: list[int],
    tf32: bool,
    downsampler: nn.Module | None = None,
    input_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carry prepared conv2 output into the next block for either VAE direction."""

    def previous(index: int) -> torch.Tensor | None:
        history = feat_cache[index]
        return None if history is None else history.permute(0, 2, 3, 4, 1).contiguous()

    def save(index: int, cache: torch.Tensor) -> None:
        feat_cache[index] = cache.permute(0, 4, 1, 2, 3)

    source = x
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    padded = None
    prepared_first = False
    bias = input_bias
    residual = residual_bias = None
    for index, block in enumerate(resnets):
        slot = feat_idx[0]
        first1 = prepared_first if padded is not None else x.shape[1] == 1 and feat_cache[slot] is None
        if padded is None:
            padded, cache, summed = _prepare(
                x.contiguous(),
                block.norm1.gamma.reshape(-1),
                bias,
                previous(slot),
                residual,
                residual_bias,
                save_input=index == 0 and input_bias is not None,
            )
            save(slot, cache)
            if summed is not None:
                x = summed
                if index == 0:
                    source = x.permute(0, 4, 1, 2, 3)
        residual, residual_bias = x, None
        if type(block.conv_shortcut) is not nn.Identity:
            conv = block.conv_shortcut
            residual = F.conv3d(x.permute(0, 4, 1, 2, 3), conv.weight).permute(0, 2, 3, 4, 1).contiguous()
            residual_bias = conv.bias
        first2 = x.shape[1] == 1 and feat_cache[slot + 1] is None
        if _fused_conv(block.conv1, x, tf32, first1):
            padded, cache, _ = _prepare_convolution(
                block.conv1, padded, block.norm2.gamma.reshape(-1), previous(slot + 1), tf32
            )
        else:
            raw = _convolve(block.conv1, padded, tf32, first1)
            padded, cache, _ = _prepare(
                raw.contiguous(), block.norm2.gamma.reshape(-1), block.conv1.bias, previous(slot + 1)
            )
        save(slot + 1, cache)
        feat_idx[0] += 2
        following = resnets[index + 1] if index + 1 < len(resnets) else None
        custom = _custom_conv(block.conv2, x, tf32, first2)
        if following is not None and _fused_conv(block.conv2, x, tf32, first2, residual=True):
            prepared_first = x.shape[1] == 1 and feat_cache[slot + 2] is None
            padded, cache, x = _prepare_convolution(
                block.conv2,
                padded,
                following.norm1.gamma.reshape(-1),
                previous(slot + 2),
                tf32,
                residual,
                residual_bias,
            )
            save(slot + 2, cache)
        elif following is None and downsampler is not None and custom:
            spatial_input = _prepare_spatial(block.conv2, padded, residual, residual_bias, tf32)
            n, t, h, w, c = spatial_input.shape
            spatial = downsampler.resample[1](spatial_input.view(n * t, h, w, c).permute(0, 3, 1, 2))
            x = ef.finish_downsample(downsampler, spatial, n, t, feat_cache, feat_idx, channels_last=True)
        else:
            x = _convolve(block.conv2, padded, tf32, first2)
            bias = block.conv2.bias
            padded = None
            if following is None:
                x = fp._residual_add(x.permute(0, 4, 1, 2, 3), bias, residual.permute(0, 4, 1, 2, 3), residual_bias)
                if downsampler is not None:
                    x = downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
    return x, source


def supports_up_stage(stage: WanResidualUpBlock | WanUpBlock) -> bool:
    return all(supports_block(block) for block in stage.resnets) and all(
        type(block.conv_shortcut) is nn.Identity for block in stage.resnets[1:]
    )


def up_block_forward(
    self: WanResidualUpBlock | WanUpBlock,
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
    first_chunk: bool = False,
) -> torch.Tensor:
    """Fuse the decoder residual chain while retaining native upsampling semantics."""
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or feat_cache is None
        or x.dtype not in (torch.bfloat16, torch.float32)
        or _stage_has_forward_hooks(self)
    ):
        forward = fp.residual_up_block_forward if type(self) is WanResidualUpBlock else WanUpBlock.forward
        return forward(self, x, feat_cache, feat_idx, first_chunk)
    if feat_idx is None:
        feat_idx = [0]
    tf32 = x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32
    x, source = _run_residual_chain(self.resnets, x, feat_cache, feat_idx, tf32)
    if type(self) is WanResidualUpBlock:
        return fp._finish_residual_up_block(self, x, source, feat_cache, feat_idx, first_chunk)
    if self.upsamplers is not None:
        x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
    return x


def supports_decoder(decoder: WanDecoder3d) -> bool:
    norm = decoder.norm_out
    return (
        fp.is_diffusers_rms_norm(norm)
        and isinstance(norm.bias, float)
        and norm.bias == 0.0
        and norm.scale == math.sqrt(norm.gamma.numel())
        and norm.gamma.numel() in _CHANNELS
        and type(decoder.nonlinearity) is nn.SiLU
        and all(
            type(conv) is WanCausalConv3d
            and conv.kernel_size == (3, 3, 3)
            and conv.stride == conv.dilation == (1, 1, 1)
            and conv.groups == 1
            and tuple(conv._padding) == (1, 1, 1, 1, 2, 0)
            for conv in (decoder.conv_in, decoder.conv_out)
        )
    )


def decoder_forward(
    self: WanDecoder3d,
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None] | None = None,
    feat_idx: list[int] | None = None,
    first_chunk: bool = False,
) -> torch.Tensor:
    """Keep native input accumulation and fuse output norm, SiLU, padding and cache."""
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or x.dtype not in (torch.bfloat16, torch.float32)
        or feat_cache is None
        or fp._has_forward_hooks(self.conv_in, self.norm_out, self.nonlinearity, self.conv_out)
    ):
        return fp.decoder_forward(self, x, feat_cache, feat_idx, first_chunk)
    if feat_idx is None:
        feat_idx = [0]
    index = feat_idx[0]
    previous = feat_cache[index]
    first_input = fp._first_frame_convolution(self.conv_in, x, self.conv_in.bias) if previous is None else None
    if first_input is not None:
        x, feat_cache[index] = first_input
    else:
        cache = x[:, :, -2:].clone(memory_format=torch.channels_last_3d)
        if cache.shape[2] < 2 and previous is not None:
            cache = torch.cat((previous[:, :, -1:], cache), dim=2)
        # Explicit spatial padding preserves the demo's native convolution choice.
        x = WanCausalConv3d.forward(self.conv_in, x, previous)
        feat_cache[index] = cache
    feat_idx[0] += 1
    x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
    for stage in self.up_blocks:
        x = stage(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
    tf32 = x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32
    index = feat_idx[0]
    previous = feat_cache[index]
    first = x.shape[2] == 1 and previous is None
    previous = None if previous is None else previous.permute(0, 2, 3, 4, 1).contiguous()
    padded, cache, _ = _prepare(x.permute(0, 2, 3, 4, 1).contiguous(), self.norm_out.gamma.reshape(-1), None, previous)
    feat_cache[index] = cache.permute(0, 4, 1, 2, 3)
    feat_idx[0] += 1
    out = _convolve(self.conv_out, padded, tf32, first)
    if self.conv_out.bias is not None:
        out = out + self.conv_out.bias
    return out.permute(0, 4, 1, 2, 3)
