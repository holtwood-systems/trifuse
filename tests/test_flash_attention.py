import pytest
import torch

from trifuse import reference_flash_attention
from trifuse.exceptions import ShapeMismatchError
from trifuse.validation import validate_flash_attention_inputs


@pytest.mark.parametrize("causal", [False, True])
def test_reference_matches_pytorch_sdpa(causal: bool):
    torch.manual_seed(7)
    q = torch.randn(2, 3, 11, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = reference_flash_attention(q, k, v, causal=causal)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(actual, expected)


def test_validation_rejects_shape_mismatch_before_device_check():
    q = torch.empty(1, 2, 8, 16)
    k = torch.empty(1, 2, 7, 16)
    v = torch.empty_like(q)

    with pytest.raises(ShapeMismatchError, match="shapes must match"):
        validate_flash_attention_inputs(q, k, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq_len", [11, 65])
def test_kernel_matches_reference_for_tail_blocks(causal: bool, seq_len: int):
    from trifuse.kernels.flash_attention import flash_attention

    torch.manual_seed(11)
    q = torch.randn(2, 3, seq_len, 16, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = flash_attention(q, k, v, causal=causal)
    expected = reference_flash_attention(q, k, v, causal=causal)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_launch_grid_keeps_batch_times_heads_out_of_grid_y():
    """B*H > 65535 时不能把 batch*heads 放进 CUDA 的 gridDim.y。"""
    from trifuse.kernels.flash_attention import _launch_grid

    # 512 * 128 = 65536，已超过 gridDim.y 的 65535 上限。
    assert _launch_grid(seq_len=4, batch=512, heads=128, block_m=64) == (512 * 128,)
    # 多个 query block 时一维 grid 仍覆盖全部 (query_block, batch_head) 组合。
    assert _launch_grid(seq_len=129, batch=2, heads=3, block_m=64) == (3 * 2 * 3,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_launches_when_batch_times_heads_exceeds_grid_y_limit():
    """回归：B*H = 65536 的请求不能因 gridDim.y 上限而启动失败。"""
    from trifuse.kernels.flash_attention import flash_attention

    torch.manual_seed(13)
    q = torch.randn(512, 128, 4, 16, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = flash_attention(q, k, v, causal=False)
    expected = reference_flash_attention(q, k, v, causal=False)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _compiled_kernel_count(kernel):
    total = 0
    for caches in kernel.device_caches.values():
        total += len(caches[0])
    return total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_not_recompiled_per_seq_len():
    """不同 seq_len 首次调用不应触发 kernel 重新编译（维度/stride 不能全部 constexpr）。"""
    from trifuse.kernels.flash_attention import _flash_attention_kernel, flash_attention

    torch.manual_seed(3)
    base = _compiled_kernel_count(_flash_attention_kernel)
    for seq_len in (256, 128, 64):
        q = torch.randn(1, 2, seq_len, 64, device="cuda", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        flash_attention(q, k, v, causal=False)
    new_kernels = _compiled_kernel_count(_flash_attention_kernel) - base
    assert new_kernels <= 1, f"每个 seq_len 触发重新编译：新增 {new_kernels} 个 kernel"
