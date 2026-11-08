# Agent Note: FlashAttention 的 batch*heads 不能占用 gridDim.y

Status: implemented

## Problem

`trifuse/kernels/flash_attention.py` 以二维 grid 启动：

```python
grid = (triton.cdiv(seq_len, block_m), batch * heads)
```

Triton 的 grid 维度依次映射到 CUDA 的 `blockIdx.x/y/z`。`gridDim.x` 上限是
2**31-1，但 `gridDim.y` / `gridDim.z` 上限只有 65535。推理服务里
`batch * heads` 完全可能超过该值（例如 512 个序列 × 128 个 head = 65536），
此时 launch 直接失败，而 kernel 本身与输入 shape 都没有问题。

只在 GPU 上按正常小规模 shape 做差分测试不会暴露该问题：测试用的
`batch * heads` 远小于 65535，二维 grid 一直合法。

## Decision

把 launch grid 压成一维：

```python
def _launch_grid(seq_len, batch, heads, block_m):
    return (triton.cdiv(seq_len, block_m) * batch * heads,)
```

kernel 内恢复 program 坐标：

```python
num_query_blocks = tl.cdiv(n_ctx, block_m)
pid = tl.program_id(0)
query_block = pid % num_query_blocks
batch_head = pid // num_query_blocks
```

`query_block` 仍然是变化最快的维度，因此相邻 program 处理同一
`(batch, head)` 的连续 query block，保持 K/V 的 L2 复用模式与原来的
`blockIdx.x` 语义一致。所有 program 都落在 `gridDim.x`（2**31-1）内。

## Alternatives considered

- **交换二维 grid 的两个轴**（`(batch*heads, cdiv(seq_len, block_m))`）。
  改动更小，但 `batch*heads` 变成变化最快的维度，相邻 CTA 访问完全不同的
  K/V，FlashAttention 的 K/V 缓存局部性变差；而且仍把 `seq_len` 的 block
  数放在有 65535 上限的 `gridDim.y`，只是把上限推到了更少见的超长序列。
- **保留二维 grid 并在 Python 侧拒绝 `batch*heads > 65535`**。这会拒绝本可
  正确计算的合法输入，属于把 launch 限制泄漏成算子契约；不做。
- **按 `(batch, head)` 循环、每个 CTA 处理多个 head**。需要重写 kernel 的
  指针与累加器布局，收益只是减少 program 数量；与本 bug 无关。

## Consequences

Benefits:

- `batch * heads > 65535` 的请求可以正常启动，不再受 CUDA `gridDim.y`
  限制。
- 一维 grid 的计算被提取为 `_launch_grid`，可在无 GPU 的 CPU 测试里直接
  断言维度与 program 总数。

Costs and limits:

- kernel 内多了一次整数除法/取模（`pid // num_query_blocks` 与
  `%`）；相对每个 CTA 的 QK^T/softmax 计算可忽略。
- 只覆盖 `batch * heads` 这一条超限路径；`seq_len` 的 block 数同样被压进
  一维，因此也顺带不再受 `gridDim.y` 限制。
- sgemm 仍是二维 grid；其第二维是 `cdiv(N, block_n)`，需要 `N` 超过约
  419 万列才会触顶，不属于本次 FlashAttention 修复范围。

## Verification

- `pytest tests/test_flash_attention.py`（CPU）：4 passed，6 skipped
  （GPU 用例因 `CUDA not available` 跳过）。
- 新增 `test_launch_grid_keeps_batch_times_heads_out_of_grid_y`：断言
  `B*H = 65536` 时 grid 是一维 `(65536,)`，且多 query block 时总数仍为
  `query_blocks * batch * heads`。
- 新增 GPU 回归用例 `test_kernel_launches_when_batch_times_heads_exceeds_grid_y_limit`
  （`512 × 128 × 4 × 16`，与参考实现对拍）；本机无 GPU，未能执行。
- `ruff format --check`、`ruff check`、`mypy trifuse` 均通过；全量 CPU
  测试 66 passed / 67 skipped。
- 未运行：任何真实 GPU 上的 Triton 编译与执行，包括新增的 65536 program
  回归用例。
