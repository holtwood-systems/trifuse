# trifuse GPU 回归 — RTX 5070 Ti (sm_120)

- 日期：2026-11-10（UTC）；commit：见 `git rev-parse HEAD`（执行时工作区干净）
- 硬件/驱动：NVIDIA GeForce RTX 5070 Ti 16GB，driver 616.56，compute 12.0
- 工具链：CUDA 13.3.73；torch 2.11.0+cu128；triton 3.6.0；pytest 9.2.0
  （venv `/home/shane/ai-lab/envs/torch`）
- 命令：`python -m pytest tests/ -x -q`（完整输出 `pytest-full.log`）
- 结果：**133 passed, 0 failed**（18.63s）

## 与 2026-11-06 RTX 4090 基线的差异

- 4090 基线：130 passed / **1 failed**（`test_sgemm_bf16`）。
- 本次：bf16 用例通过，maxdiff 0.0（`trifuse.kernels.sgemm` 真实 Triton
  kernel，无 torch 回退）。已知失败未在本环境复现——属环境相关失败
  （旧 4090 运行时的 triton/torch 组合 vs cu128/sm_120），不能据此宣称
  缺陷已修复；若后续在 4090 类环境复现需恢复登记。
