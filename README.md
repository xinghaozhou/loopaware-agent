# Loopaware Agent baseline

分支 `baseline/last-available-batching` 收敛 Ouro HF 推理运行时、last-available
KV policy 和 batching 正确性验证。模块准入、分支清单、修复记录与验证边界见
[baseline 审查报告](docs/baseline_audit.md)。

核心文件：

- `vendor/ouro_hf/configuration_ouro.py`：模型配置。
- `vendor/ouro_hf/modeling_ouro.py`：全深度 prefill、单 token decode、物理 early exit。
- `vendor/ouro_hf/cache_ouro.py`：full-per-depth 参考和 last-available KV。
- `src/loopaware_agent/ut_trace.py`：逐 token UT 追踪。
- `src/loopaware_agent/ut_scheduling.py`：离线组批和代价分析。

使用项目锁定环境：

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python experiments/adaptive_kv_smoke.py
```

无 GPU 时，可单独建立轻量 CPU 验证环境：

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

启用 last-available：设置 `config.kv_cache_policy = "last_available"`、
`config.physical_early_exit = True` 和 `early_exit_threshold`。
prefill 执行全部 UT，之后每次 forward 输入一个新 token。
阈值 1.0 用于与 full-per-depth 参考作全深度回归。

动态重组要求相同 padded prompt 长度，每个 token frontier 中所有请求各执行
一次；全部提交后才能推进。暂不支持异步推进、在线增删或 EOS 后直接移除行。

真实 checkpoint 验证需要可用 GPU 和权重，默认固定模型 revision：

```bash
uv run --frozen python experiments/validate_adaptive_kv_checkpoint.py \
  --output results/adaptive_kv_checkpoint_validation.json
uv run --frozen python experiments/sanity_threshold_one_equivalence.py \
  --output results/threshold_one_equivalence.json
```

本轮完成 CPU 回归和 smoke；当前机器无法使用 NVIDIA 驱动，尚未重跑 GPU/BF16。
历史 GPU 报告与本轮结果在审查报告中分别标明。
