# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming, compilation and weight-lifecycle coverage for cuDNN Wan blocks."""

from copy import deepcopy
from types import MethodType

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanResidualBlock,
    WanResidualDownBlock,
    WanResidualUpBlock,
    WanUpBlock,
)

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import cudnn, forwards
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["B200"]}, num_cards=1),
]


@pytest.fixture(autouse=True)
def blackwell():
    capability = current_omni_platform.get_device_capability()
    if capability is None or tuple(capability) not in ((10, 0), (10, 3)):
        pytest.skip("SM100/SM103 required")
    pytest.importorskip("cudnn")
    try:
        cudnn.provider()
    except ImportError as error:
        pytest.skip(str(error))


def make_pair(channels, dtype):
    torch.manual_seed(17)
    original = WanResidualBlock(channels, channels, 0).eval().to(device="cuda", dtype=dtype)
    candidate = deepcopy(original)
    for conv in (candidate.conv1, candidate.conv2):
        conv.to(memory_format=torch.channels_last_3d)
        conv.add_module("_wan_cudnn_weight", cudnn.PackedWeight())
    setattr(candidate, forwards.CFG_ATTR, forwards.FastPathConfig(channels_last=True))
    candidate.forward = MethodType(cudnn.residual_block_forward, candidate)
    return original, candidate


@torch.no_grad()
@pytest.mark.parametrize("channels", [96, 160, 256, 320, 384])
@pytest.mark.parametrize("mode", ["bf16", "fp32", "tf32"])
def test_streaming_chunks(channels, mode):
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    with torch.backends.cudnn.flags(allow_tf32=mode == "tf32"):
        original, candidate = make_pair(channels, dtype)
        expected_cache: list[torch.Tensor | None] = [None, None]
        actual_cache: list[torch.Tensor | None] = [None, None]
        for frames in (1, 1, 4):
            x = torch.randn(2, channels, frames, 5, 7, device="cuda", dtype=dtype)
            expected = original(x, expected_cache, [0])
            actual = candidate(x, actual_cache, [0])
            assert actual.isfinite().all()
            if mode == "fp32":
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            else:
                error = (actual.float() - expected.float()).norm() / expected.float().norm()
                assert error < (0.02 if mode == "bf16" else 0.001)
            for actual_history, expected_history in zip(actual_cache, expected_cache):
                assert actual_history is not None and expected_history is not None
                assert actual_history.shape == expected_history.shape


@torch.no_grad()
def test_packed_weight_update_and_offload():
    _, candidate = make_pair(96, torch.bfloat16)
    conv = candidate.conv1
    packed = conv._wan_cudnn_weight
    first = packed(conv.weight, False)
    assert packed(conv.weight, False) is first
    conv.weight.add_(0.125)
    second = packed(conv.weight, False)
    assert not torch.equal(first, second)
    state = candidate.state_dict()
    conv.weight.zero_()
    # load_state_dict updates the authoritative parameters and their versions.
    state["conv1.weight"] = torch.ones_like(conv.weight)
    candidate.load_state_dict(state)
    assert torch.equal(packed(conv.weight, False), cudnn.provider().pack_conv3d_weight_sm100(conv.weight))
    candidate.cpu()
    assert packed.packed is None
    candidate.to(device="cuda", dtype=torch.float32)
    assert packed(conv.weight, True).dtype == torch.float32


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiled_custom_ops_and_changed_graph_input(dtype):
    # Packing is model preparation; the operator sequence itself is fullgraph.
    _, block = make_pair(96, dtype)
    allow_tf32 = dtype == torch.float32
    weight = block.conv1._wan_cudnn_weight(block.conv1.weight, allow_tf32)
    gamma = block.norm1.gamma.reshape(-1)
    x = torch.randn(1, 1, 5, 7, 96, device="cuda", dtype=dtype)

    def run(x, previous):
        padded, cache1, summed1 = cudnn.norm_pad(x, gamma, block.conv1.bias, previous, x)
        output, cache2, summed2 = cudnn.conv_norm_pad(
            padded, weight, block.conv1.bias, gamma, previous, allow_tf32, summed1
        )
        spatial = cudnn.conv_residual_pad(padded, weight, block.conv1.bias, summed1, None, allow_tf32)
        return output, cache1, cache2, summed1, summed2, spatial

    previous = torch.randn_like(x)
    compiled = torch.compile(run, fullgraph=True)
    graph = None
    try:
        expected = run(x, previous)
        actual = compiled(x, previous)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        changed = x * 0.5
        for a, b in zip(compiled(changed, previous), run(changed, previous)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        captured = []
        graph = current_omni_platform.capture_graph(lambda: captured.append(run(x, previous)))
        assert graph is not None
        x.copy_(changed)
        current_omni_platform.replay_graph(graph, lambda: None)
        for a, b in zip(captured[0], run(x, previous)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    finally:
        if graph is not None:
            graph.reset()
        torch.compiler.reset()


@torch.inference_mode()
def test_inference_weight_reload():
    _, candidate = make_pair(96, torch.bfloat16)
    conv = candidate.conv1
    packed = conv._wan_cudnn_weight
    hook = conv.register_load_state_dict_post_hook(packed.invalidate)
    try:
        first = packed(conv.weight, False)
        assert packed(conv.weight, False) is first
        conv.load_state_dict({"weight": torch.zeros_like(conv.weight), "bias": conv.bias})
        actual = packed(conv.weight, False)
        assert torch.count_nonzero(actual) == 0
    finally:
        hook.remove()


@torch.no_grad()
@pytest.mark.parametrize(
    "in_channels,out_channels,spatial,temporal",
    [
        (160, 160, True, False),
        (160, 320, True, True),
        (320, 640, True, True),
        (640, 640, False, False),
    ],
)
@pytest.mark.parametrize("mode", ["bf16", "fp32", "tf32"])
def test_encoder_stage_streaming(in_channels, out_channels, spatial, temporal, mode):
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    torch.manual_seed(31)
    with torch.backends.cudnn.flags(allow_tf32=mode == "tf32"):
        original = WanResidualDownBlock(in_channels, out_channels, 0, 2, temporal, spatial).eval().to("cuda", dtype)
        candidate = deepcopy(original)
        for module in candidate.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
            elif isinstance(module, torch.nn.Conv2d):
                module.to(memory_format=torch.channels_last)
        candidate.forward = MethodType(cudnn.residual_down_block_forward, candidate)
        for block in candidate.resnets:
            for conv in (block.conv1, block.conv2):
                conv.add_module("_wan_cudnn_weight", cudnn.PackedWeight())
        # The first encoder stage consumes the causal input convolution's bias
        # in norm preparation; both the block skip and outer shortcut need it.
        bias = torch.randn(in_channels, device="cuda", dtype=dtype) if in_channels == out_channels == 160 else None

        def biased(value):
            return value if bias is None else (value.permute(0, 2, 3, 4, 1) + bias).permute(0, 4, 1, 2, 3)

        slots = 4 + int(temporal)
        expected_cache: list[torch.Tensor | None] = [None] * slots
        actual_cache: list[torch.Tensor | None] = [None] * slots
        for frames in (1, 4, 4) if temporal else (1, 1, 4):
            x = torch.randn(2, in_channels, frames, 8, 10, device="cuda", dtype=dtype)
            before = x.clone()
            retained = [(value, value.clone()) for value in actual_cache if value is not None]
            expected_idx, actual_idx = [0], [0]
            expected = original(biased(x), expected_cache, expected_idx)
            actual = candidate(x, actual_cache, actual_idx, input_bias=bias)
            assert actual_idx == expected_idx == [slots]
            torch.testing.assert_close(x, before, rtol=0, atol=0)
            for value, saved in retained:
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            for a, b in [(actual, expected), *zip(actual_cache, expected_cache)]:
                assert a is not None and b is not None
                assert a.shape == b.shape
                if mode == "fp32":
                    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
                else:
                    error = (a.float() - b.float()).norm() / b.float().norm()
                    assert error < (0.02 if mode == "bf16" else 0.003)

        if mode == "bf16":
            # A hook installed after warmup must still force the ordinary stage
            # path, including nested normalization calls that fusion bypasses.
            seen = []

            def change_norm(module, args, output):
                seen.append(module)
                return output * 0.5

            hooks = [stage.resnets[0].norm1.register_forward_hook(change_norm) for stage in (original, candidate)]
            try:
                first = x[:, :, :1].contiguous()
                expected = original(biased(first), [None] * slots, [0])
                actual = candidate(first, [None] * slots, [0], input_bias=bias)
                assert seen == [original.resnets[0].norm1, candidate.resnets[0].norm1]
                assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.02
            finally:
                for hook in hooks:
                    hook.remove()


@torch.no_grad()
@pytest.mark.parametrize("mode", ["bf16", "tf32"])
def test_causal_input_streaming_and_compilation(mode):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    tf32 = mode == "tf32"
    torch.manual_seed(23)
    conv = WanCausalConv3d(12, 160, 3, padding=1).eval().to(device="cuda", dtype=dtype)
    packed = cudnn.PackedWeight(causal=True)(conv.weight, tf32)
    compiled = torch.compile(cudnn.input_conv, fullgraph=True)
    previous: torch.Tensor | None = None
    with torch.backends.cudnn.flags(allow_tf32=tf32):
        for frames in (1, 1, 4):
            x = torch.randn(2, 12, frames, 8, 10, device="cuda", dtype=dtype)
            old = previous.clone() if previous is not None else None
            expected = conv(x, None if previous is None else previous.permute(0, 4, 1, 2, 3))
            raw, history = compiled(x, packed, previous, tf32)
            actual = (raw + conv.bias).permute(0, 4, 1, 2, 3)
            assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.003
            current = x.permute(0, 2, 3, 4, 1)
            all_inputs = current if previous is None else torch.cat((previous, current), dim=1)
            torch.testing.assert_close(history, all_inputs[:, -2:], atol=0, rtol=0)
            if old is not None:
                torch.testing.assert_close(previous, old, atol=0, rtol=0)
            previous = history


@torch.no_grad()
@pytest.mark.parametrize(
    "in_channels,out_channels,spatial,temporal,residual_stage",
    [
        (512, 256, False, False, True),
        (256, 256, True, False, True),
        (512, 512, True, True, True),
        (96, 96, True, True, False),
    ],
)
@pytest.mark.parametrize("mode", ["bf16", "fp32", "tf32"])
def test_decoder_stage_streaming(in_channels, out_channels, spatial, temporal, residual_stage, mode):
    """Check cross-block preparation, upsampling, retained histories, and late hooks."""
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    torch.manual_seed(41)
    with torch.backends.cudnn.flags(allow_tf32=mode == "tf32"):
        original = (
            (
                WanResidualUpBlock(in_channels, out_channels, 1, 0, temporal, spatial)
                if residual_stage
                else WanUpBlock(in_channels, out_channels, 1, upsample_mode="upsample3d")
            )
            .eval()
            .to("cuda", dtype)
        )
        candidate = deepcopy(original)
        for module in candidate.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
            elif isinstance(module, torch.nn.Conv2d):
                module.to(memory_format=torch.channels_last)
        candidate.forward = MethodType(cudnn.up_block_forward, candidate)
        for block in candidate.resnets:
            for conv in (block.conv1, block.conv2):
                conv.add_module("_wan_cudnn_weight", cudnn.PackedWeight())
        slots = 4 + int(temporal)
        expected_cache, actual_cache = [None] * slots, [None] * slots
        for chunk, frames in enumerate((1, 2, 2)):
            x = torch.randn(2, in_channels, frames, 4, 6, device="cuda", dtype=dtype)
            retained = [(v, v.clone()) for v in actual_cache if isinstance(v, torch.Tensor)]
            expected_idx, actual_idx = [0], [0]
            expected = original(x, expected_cache, expected_idx, first_chunk=chunk == 0)
            actual = candidate(x, actual_cache, actual_idx, first_chunk=chunk == 0)
            assert actual_idx == expected_idx == [slots]
            for value, before in retained:
                torch.testing.assert_close(value, before, atol=0, rtol=0)
            for value, reference in [(actual, expected), *zip(actual_cache, expected_cache)]:
                if not isinstance(reference, torch.Tensor):
                    assert value == reference
                    continue
                assert value.shape == reference.shape
                if mode == "fp32":
                    torch.testing.assert_close(value, reference, atol=1e-4, rtol=1e-4)
                else:
                    assert (value.float() - reference.float()).norm() / reference.float().norm() < (
                        0.02 if mode == "bf16" else 0.003
                    )
        if mode == "bf16":
            seen = []

            def hook(module, args, output):
                seen.append(module)
                return output * 0.5

            hooks = [stage.resnets[1].norm1.register_forward_hook(hook) for stage in (original, candidate)]
            try:
                first = x[:, :, :1].contiguous()
                expected = original(first, [None] * slots, [0], first_chunk=True)
                actual = candidate(first, [None] * slots, [0], first_chunk=True)
                assert seen == [original.resnets[1].norm1, candidate.resnets[1].norm1]
                assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.02
            finally:
                for handle in hooks:
                    handle.remove()


@torch.no_grad()
@pytest.mark.parametrize("mode", ["bf16", "fp32", "tf32"])
def test_decoder_input_and_prepared_head(mode):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanDecoder3d

    torch.manual_seed(29)
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    original = WanDecoder3d(dim=96, z_dim=4, dim_mult=[1], num_res_blocks=1, temperal_upsample=[])
    original = original.eval().to(device="cuda", dtype=dtype)
    torch.nn.utils.convert_conv3d_weight_memory_format(original, torch.channels_last_3d)
    candidate = deepcopy(original)
    assert cudnn.supports_decoder(candidate)
    for stage in candidate.up_blocks:
        assert cudnn.supports_up_stage(stage)
        stage.forward = MethodType(cudnn.up_block_forward, stage)
        for block in stage.resnets:
            for conv in (block.conv1, block.conv2):
                conv.add_module("_wan_cudnn_weight", cudnn.PackedWeight())
    candidate.forward = MethodType(cudnn.decoder_forward, candidate)
    slots = sum(type(module) is WanCausalConv3d for module in original.modules())
    expected_cache, actual_cache = [None] * slots, [None] * slots
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=mode == "tf32"):
        for frames in (1, 2, 2):
            x = torch.randn(2, 4, frames, 4, 6, device="cuda", dtype=dtype)
            retained = [(value, value.clone()) for value in actual_cache if isinstance(value, torch.Tensor)]
            expected_idx, actual_idx = [0], [0]
            expected = original(x, expected_cache, expected_idx)
            actual = candidate(x, actual_cache, actual_idx)
            assert actual_idx == expected_idx == [slots]
            for value, before in retained:
                torch.testing.assert_close(value, before, atol=0, rtol=0)
            for value, reference in [(actual, expected), *zip(actual_cache, expected_cache)]:
                assert isinstance(value, torch.Tensor) and isinstance(reference, torch.Tensor)
                assert value.shape == reference.shape
                if mode == "fp32":
                    torch.testing.assert_close(value, reference, atol=1e-4, rtol=1e-4)
                else:
                    assert (value.float() - reference.float()).norm() / reference.float().norm() < (
                        0.02 if mode == "bf16" else 0.003
                    )
        if mode == "fp32":
            compiled = torch.compile(candidate, fullgraph=True)
            expected = candidate(x, [None] * slots, [0])
            actual = compiled(x, [None] * slots, [0])
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        if mode == "bf16":
            seen = []

            def hook(module, args, output):
                seen.append(module)
                return output * 0.5

            handles = [model.norm_out.register_forward_hook(hook) for model in (original, candidate)]
            try:
                expected = original(x, [None] * slots, [0])
                actual = candidate(x, [None] * slots, [0])
                assert seen == [original.norm_out, candidate.norm_out]
                assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.02
            finally:
                for handle in handles:
                    handle.remove()


@torch.no_grad()
def test_wide_bf16_temporal_upsample_preserves_accumulation():
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    torch.manual_seed(31)
    conv = WanCausalConv3d(1024, 2048, kernel_size=(3, 1, 1), padding=(1, 0, 0))
    conv = conv.eval().to(device="cuda", dtype=torch.bfloat16, memory_format=torch.channels_last_3d)
    setattr(conv, forwards.CFG_ATTR, forwards.FastPathConfig(channels_last=True))
    cache = ["Rep"]
    previous: torch.Tensor | None = None
    for _ in range(2):
        x = torch.randn(1, 1024, 1, 4, 6, device="cuda", dtype=torch.bfloat16)
        x = x.contiguous(memory_format=torch.channels_last_3d)
        expected = WanCausalConv3d.forward(conv, x, previous)
        actual = forwards._run_cached_causal_conv(conv, x, cache, 0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        previous = torch.cat((torch.zeros_like(x) if previous is None else previous[:, :, -1:], x), dim=2)
        torch.testing.assert_close(cache[0], previous, atol=0, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize("mode", ["bf16", "fp32", "tf32"])
def test_flat_encoder_streaming_and_late_hook(mode):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanEncoder3d, WanResample

    from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import encoder_forwards

    torch.manual_seed(37)
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    original = WanEncoder3d(dim=96, z_dim=4, dim_mult=[1, 2], num_res_blocks=2, temperal_downsample=[False])
    original = original.eval().to(device="cuda", dtype=dtype)
    torch.nn.utils.convert_conv3d_weight_memory_format(original, torch.channels_last_3d)
    candidate = deepcopy(original)
    cfg = forwards.FastPathConfig(channels_last=True, exact_bf16_norm=True)
    for module in candidate.modules():
        setattr(module, forwards.CFG_ATTR, cfg)
        if type(module) is WanResidualBlock:
            module.forward = MethodType(cudnn.residual_block_forward, module)
            for conv in (module.conv1, module.conv2):
                conv.add_module("_wan_cudnn_weight", cudnn.PackedWeight())
        elif type(module) is WanResample:
            module.forward = MethodType(encoder_forwards.downsample_forward, module)
    candidate.forward = MethodType(cudnn.flat_encoder_forward, candidate)
    slots = sum(type(module) is WanCausalConv3d and module.kernel_size[0] > 1 for module in original.modules())
    expected_cache, actual_cache = [None] * slots, [None] * slots

    def check(actual, expected):
        if mode == "fp32":
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        else:
            assert (actual.float() - expected.float()).norm() / expected.float().norm() < (
                0.02 if mode == "bf16" else 0.003
            )

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=mode == "tf32"):
        for frames in (1, 4, 4):
            x = torch.randn(2, 3, frames, 8, 10, device="cuda", dtype=dtype)
            retained = [(value, value.clone()) for value in actual_cache if isinstance(value, torch.Tensor)]
            expected_idx, actual_idx = [0], [0]
            check(candidate(x, actual_cache, actual_idx), original(x, expected_cache, expected_idx))
            assert actual_idx == expected_idx == [slots]
            for value, before in retained:
                torch.testing.assert_close(value, before, atol=0, rtol=0)
            for value, reference in zip(actual_cache, expected_cache):
                assert isinstance(value, torch.Tensor) and isinstance(reference, torch.Tensor)
                check(value[:, :, -reference.shape[2] :], reference)
        seen = []

        def hook(module, args, output):
            seen.append(module)
            return output * 0.5

        norms = [model.down_blocks[0].norm1 for model in (original, candidate)]
        handles = [norm.register_forward_hook(hook) for norm in norms]
        try:
            expected = original(x, [None] * slots, [0])
            actual = candidate(x, [None] * slots, [0])
            assert seen == norms
            check(actual, expected)
        finally:
            for handle in handles:
                handle.remove()
