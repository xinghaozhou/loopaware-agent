# Ouro adaptive-exit / last-available baseline

当前基线：`baseline/adaptive-exit-last-available`。

只确认 adaptive exit 与 last-available KV 的核心推理语义，核心实现位于：

- `vendor/ouro_hf/modeling_ouro.py`：full-depth prefill、退出判断、物理停止、选中深度 logits。
- `vendor/ouro_hf/cache_ouro.py`：当前 token 暂存/提交、历史 KV 按 min(u, U_j) 查询。
- `vendor/ouro_hf/configuration_ouro.py`：上述功能的配置。
- `src/loopaware_agent/ut_trace.py`：验证脚本使用的诊断工具，不参与模型计算。

UT-aware batching、SUF、动态请求重组、prefill cache 合并与 profiling 优化均未纳入。
保留 tensor 的固定 batch 维度及 early exit 必需的行索引；不包含请求调度或组批策略。

项目环境运行：

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python experiments/adaptive_kv_smoke.py
```

无 GPU 时的轻量 CPU 验证：

```bash
uv venv /tmp/loopaware-baseline-venv --python 3.12
uv pip install --python /tmp/loopaware-baseline-venv/bin/python \
  torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/loopaware-baseline-venv/bin/python transformers==4.57.6
PYTHONPATH=src uv run --no-project --python /tmp/loopaware-baseline-venv/bin/python \
  python -m unittest discover -s tests -v
uv run --no-project --python /tmp/loopaware-baseline-venv/bin/python \
  python experiments/adaptive_kv_smoke.py
```

运行模式：

```python
config.kv_cache_policy = "last_available"
config.physical_early_exit = True
config.early_exit_threshold = 0.7
# 模型 .eval()，在 torch.inference_mode() 中以 use_cache=True 运行。
```

prefill 执行全部 UT；decode 每次输入一个 token。阈值 1.0 明确定义为全深度，
显式 exit_at_step 优先。原始参考模式为 full_per_depth + physical_early_exit=False。
last_available + physical_early_exit=False 是同 policy 的 run-all 对照。

真实 checkpoint 复核入口（需要模型权重和可用 GPU）：

```bash
uv run --frozen python experiments/validate_adaptive_kv_checkpoint.py \
  --output results/adaptive_kv_checkpoint_validation.json
uv run --frozen python experiments/sanity_threshold_one_equivalence.py \
  --output results/threshold_one_equivalence.json
```

本轮通过 29 项 CPU 回归和 smoke；GPU/BF16 未复跑。
[实现说明](docs/adaptive_exit_kv_implementation.md) · [审查报告](docs/baseline_audit.md)
