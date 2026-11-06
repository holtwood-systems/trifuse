# Agent Note: RTX 4090 benchmark 原始报告归档惯例

Status: implemented

## Problem

本仓此前没有任何带原始报告的硬件快照：README 与旧文档中的 RTX 3060
数字没有 JSON/日志可复核，按仓库纪律不可引用。审计文档明确指出
"尚缺逐次 raw timing/provenance"。

## Decision

在 `docs/performance/results/<date>-<gpu>/` 归档每次硬件快照的
benchmark 原始报告（`*.txt` + `*.json`，JSON 自带 device/cuda/torch
metadata）、全量测试日志与一份 `report.md`（环境、摘要、边界、
失败项）。报告只做摘要与边界声明，不复写 raw 数据。

首批：`2026-11-06-rtx4090/`（RTX 4090，torch 2.7.1+cu128，triton 3.3.1）：
rmsnorm_rope 18/18 正确、gated_mlp 8/8 正确、pytest 130/131（bf16
SGEMM 容差用例失败，现象与复现条件已记录）。

## Alternatives considered

- 只更新 README 数字：仍是不可复核描述，重复旧问题。
- 把报告提交到仓库根：污染源码目录，且无日期/硬件维度。

## Consequences

- 新硬件快照有了可复核入口；旧 RTX 3060 数字仍不可引用，直到
  重跑产出同格式包。
- `test_sgemm_bf16` 在 sm_89 上的失败成为公开记录的待查项，
  后续修复需附新的快照包。

## Verification

`python -m tests.benchmarks.bench_rmsnorm_rope` 与 `bench_gated_mlp`
在 RTX 4090 上产出本目录 4 个报告文件；`python -m pytest tests/ -q`
日志 130 passed / 1 failed 与 report.md 一致。
