# 2026-11-06 RTX 4090 benchmark 与测试快照

本目录是本仓第一份**带原始报告**的硬件快照（此前只有无 raw 的
RTX 3060 描述性数字，按仓库纪律不可引用）。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090（sm_89，驱动 550.107.02） |
| CUDA | 12.8（torch 自带 runtime） |
| PyTorch | 2.7.1+cu128 |
| Triton | 3.3.1 |
| Python | 3.11（conda base） |
| 仓库 | 本目录所在提交（`git log` 可查），benchmark 前工作区 clean |

## 原始报告（本目录）

- `rmsnorm_rope_benchmark.txt` / `.json`：`python -m tests.benchmarks.bench_rmsnorm_rope`
- `gated_mlp_benchmark.txt` / `.json`：`python -m tests.benchmarks.bench_gated_mlp`
- `pytest-full.log`：`python -m pytest tests/ -q` 全量输出

JSON 自带 `metadata`（device/cuda_version/pytorch_version），可直接复核。

## 结果摘要

### RMSNorm+RoPE（fused_rmsnorm_rope，18 个 shape）

- 正确性 18/18；latency 0.043–0.746 ms；
- 最大 shape `(8, 2048, 4096)`：0.746 ms、**359.9 GB/s**（带宽型 kernel，
  TFLOPS 不适用）；全局最高带宽 438.0 GB/s（小 shape）。

### Gated MLP（fused_gated_mlp_silu，8 个 shape）

- 正确性 8/8；latency 0.747–18.673 ms；最高 **31.8 TFLOPS**；
- 最大 shape `(4, 512, 4096, 11264)`：18.673 ms、20.2 TFLOPS、13.3 GB/s
  （大 M 下转为算力受限）。

### 测试

**130 passed / 1 failed**（131 项，6.24 s）。

失败项：`tests/test_sgemm.py::test_sgemm_bf16`——bf16 SGEMM 与
`torch.mm` 的 `rtol=atol=1e-2` 比较越界（例：-3.4062 vs -3.4375、
19.625 vs 19.5，约 0.3% 相对差）。同机复现（1024³、seed 0、bf16）
最大绝对/相对误差为 0，说明失败与测试内特定 shape/seed 的 bf16
累加顺序差异相关，**不是**普遍性精度缺陷；但也说明该用例的
容差在 sm_89/torch 2.7.1 组合下偏紧。按纪律保留失败，不掩盖。

## 边界

- 单 GPU、单次运行，无重复测量与方差统计（BenchmarkSuite 内部
  warmup=10/iter=100，已含基本稳态）；
- 与 RTX 3060 旧数字不构成前后对比（旧数字无 raw 报告）；
- autotuner 未在本次快照中启用；
- `test_sgemm_bf16` 的失败根因（容差 vs shape 相关累加差异）待后续
  单独分析，本报告只记录现象。
