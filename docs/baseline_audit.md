# 核心 baseline 审查：2026-10-29

## 当前结论

当前分支 `baseline/adaptive-exit-last-available` 只准入 adaptive exit 与
last-available 推理核心。上一版 `baseline/last-available-batching@2a1f499`
保留作为历史，但不再是本轮要求的核心基线。

modeling_ouro.py、cache_ouro.py 从动态调度引入前的 41870c5 提取，再应用边界修复；
配置同步缩窄为 full_per_depth / last_available。保留 tokenizer、验证脚本与诊断 tracer。

## 删除范围

- 删除 ut_scheduling.py、对应测试与普通 batching 报告。
- 移除 set_decode_batch_indices、global request ID 映射、pending frontier、
  merge_adaptive_prefill_caches、dense-history 预解析与 profiler ranges。
- 删除动态 regrouping 测试，移除 final_exit_shared 代码与配置选项。
- 不包含预测器、SUF、CBD、request-indexed 优化或 BFCL/PLAS/Poisson 调度。
- 保留普通固定 batch 张量、early exit 的 active-row 身份映射及相应正确性测试；
  这些不是 batching 调度策略。

## 两个核心文件

| 文件 | 确认范围 |
|---|---|
| modeling_ouro.py | full-depth prefill、CDF/forced 退出、物理停止、退出状态 logits、显式执行元数据 |
| cache_ouro.py | staged/commit/abort、历史 min(u, U_j)、按 token 保留真实生成的 KV、full-per-depth 参考 |

当前 last-available 存储是逐 token 的稀疏 KV 语义参考；查询时才按深度解析历史，
不再沿用上一版 dense-history/部分 frontier 实现。详见 adaptive_exit_kv_implementation.md。

## 本轮发现和修复

收窄后先运行独立核心测试，复现 5 个失败：

1. 不完整 prefill 未在 begin_decode 拒绝。
2. 请求数变化未在 transaction 启动前拒绝。
3. last_available + use_cache=False 静默退化为另一条执行路径。
4. 非法退出参数在 begin_decode 后报错，污染 transaction。
5. gate 饱和令累计概率过早达到 1，使阈值 1 错误地在 UT=1 退出。

上述问题均已修复。阈值 1 明确定义为全深度模式；这也是本 baseline 对 full-compute
等价回归的约定。该约定同时用于 prefill 输出选择、adaptive decode 和 tracer。

另补充：commit 拥有独立退出深度元数据副本，避免调用方修改 tensor 改写历史；
staged 行索引范围/唯一性检查；重复 depth/layer 写入拒绝；
transaction 在受异常保护的区域内启动。原始缺失 KV 提交验证被恢复，
保证不会把未生成的 KV 静默填零后当作合法历史。

## 验证证据

- uv 启动：Python 3.12.16、PyTorch 2.9.0+cpu、Transformers 4.57.6。
- 29 项 CPU 测试通过：24 项核心/cache/model 测试，5 项诊断 tracer 测试。
- last-available smoke 通过；两个 checkpoint 验证脚本 --help、uv lock --check 通过。
- 实际输出保存于 baseline_cpu_validation.txt。
- 独立 oracle 覆盖 2 层、4 UT、2 行、5 个历史 token，并分别检查 K/V。
- 物理退出的工作量通过 decoder forward hook 计数验证，而不只检查 metadata。
- 阈值 1 对照 full-per-depth，也对照无 cache 的完整前缀重新计算。
- greedy generate 闭环对照 threshold=0/0.7/1；异常注入后可精确重试。

测试使用独立 uv CPU 环境，不代表已同步全部项目 CUDA/数据集依赖。
当前机器 NVIDIA 驱动不可用，没有本轮真实模型 GPU/BF16、任务准确率、吞吐量或显存收益结论。
基线正确性结论仅限上述已执行的推理范围，不覆盖训练、beam 或服务后端。

## 原有分支清单

已 git fetch origin --prune，原有分支保留：

| 分支 | HEAD | 内容 |
|---|---|---|
| main / origin/main | 974cc97 | 旧入口及 HF/vLLM gitlink |
| origin/xinghao-codex | 9009146 | HF adaptive runtime、分析、KV 优化 |
| origin/cbd-ablation | d9176c9 | 外部 CBD runtime patch 与消融 |
| origin/xinghao-las-levels | 2f9d041 | BFCL program scheduling 扩展 |
| origin/autellix | c9e9cfc | PLAS 模拟与批处理实验 |
| origin/xinghao-las-level | 90bc3fe | shared-GPU Poisson 实验 |

origin/HEAD 是 origin/main 的符号引用。调度分支在 9009146 后未修改 HF 核心运行时。
