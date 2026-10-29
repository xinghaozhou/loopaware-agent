# Baseline 审查：2026-10-29

## 结论与来源

新分支：`baseline/last-available-batching`。以 `main@974cc97` 为 Git 历史起点，
从 `origin/xinghao-codex@9009146` 按文件选入 HF 运行时、UT 基础模块、测试、
必要验证脚本，再修复本次确认的问题。没有整体合并实验分支。
原有分支保留；此 baseline 是本地分支。原工作区干净，无用户未提交代码。

本轮 `git fetch origin --prune` 后的所有原有分支：

| 分支 | HEAD | 关系和处理 |
|---|---|---|
| `main` / `origin/main` | `974cc97` | 旧入口与未初始化 HF/vLLM gitlink，不作为可运行推理基线 |
| `origin/xinghao-codex` | `9009146` | 共享 HF 运行时来源；挑选核心，排除优化和大量实验 |
| `origin/cbd-ablation` | `d9176c9` | 在上述分支上增加 CBD 外部 runtime patch 与消融 |
| `origin/xinghao-las-levels` | `2f9d041` | 在 CBD 分支上增加 BFCL program scheduling 与扩展实验 |
| `origin/autellix` | `c9e9cfc` | 从 2f9d041 派生，增加 Autellix PLAS 模拟与批处理实验 |
| `origin/xinghao-las-level` | `90bc3fe` | 从 2f9d041 派生，增加 shared-GPU Poisson 实验 |

`origin/HEAD` 是指向 `origin/main` 的符号引用，不额外算一个分支。

## 模块准入

“准入”表示在以下范围内有源码审查及本次可执行测试支持，不代表任意输入、
硬件或生产服务场景均已验证。

| 模块 | 结论 | 证据及边界 |
|---|---|---|
| configuration、HF decoder 基础实现 | 准入：推理基础 | tiny model 实例化、prefill、decode、greedy generate；未验证训练和分类头 |
| FullPerDepthCache | 准入：语义参考 | greedy generate；阈值 1 下与 last-available 逐步 logits 完全一致 |
| LastAvailableCache | 准入 | 手工 KV oracle：历史 token 在深度 u 读取 min(u, U_j)；prompt 保留对应深度；当前 token 提交前使用当前深度 KV |
| 物理 early exit / active-row compaction | 准入 | forced/threshold 与同 policy run-all 对照；带 padding 三行 UT=1/2/3，连续三步改变深度 |
| 固定 batching、prefill cache 合并 | 准入 | batched parity 和 merge 测试；不要求 BF16 B=1 与 B>1 bitwise 相同 |
| 动态 microbatch regrouping | 限定准入 | 模型级两个连续 frontier 改变顺序、分组，与固定 batch 对照；每位置全部请求须恰好提交一次 |
| ut_trace.py | 准入：固定 batch 身份 | gate 分布、物理与 run-all forced exit 元数据；动态组批须调用方维护 global request ID |
| ut_scheduling.py | 准入：离线分析 | FCFS、预测 UT 稳定排序、SUF、true-UT 工作量守恒；只支持完整固定宽度批次，不是在线服务调度器 |
| FinalExitSharedCache | 兼容保留，非选定 policy | 与核心共用文件且原测试覆盖；用户确认的主 policy 仍是 last-available |
| cache reorder | 有限验证 | CPU KV 扩展、缩减、历史重排；不能据此确认真实 checkpoint beam generate |

## 未纳入的模块

| 模块 | 原因 |
|---|---|
| optimizations/request_indexed_kv/ | 独立 monkeypatch/cache factory，原测试仅覆盖少量存储行为；本轮未作模型级等价与 GPU 性能复核 |
| experiments/cbd_ablation/ | 面向另一仓库的 patch，单槽 paged KV/FlashAttention；速度结果不能证明此 HF last-available policy 正确 |
| experiments/program_scheduling/：BFCL、PLAS、Poisson | trace 驱动模拟、不同执行约束及大体积结果；本轮未完成独立端到端验证 |
| predictor headroom/runtime profiler/threshold sweep | 实验框架与受控预测准确率，不是已确认的通用预测器，不属最小运行时 |
| trajectory/SWE replay 与数据集 submodule | 部分数据规范化测试不足以确认完整流水线 |
| vLLM、旧 try.py/test_vllm.py、旧 trace 输出 | main gitlink 未初始化；vLLM 路径没有本轮 early-exit/KV 验证；旧入口与输出仅从新分支移除 |

其他分支在 9009146 之后没有修改 HF 核心运行时。本轮比较了分支文件差异，
并抽查 CBD 使用说明、PLAS 模拟器和 shared-GPU invariant。
“未纳入”表示证据不足或执行语义不同，不是断言模块实现错误。
所有未纳入代码仍可在原有分支恢复。

## 发现与修复

1. **重复 frontier 请求**：旧代码只累计行数，未排除重复请求，可能提前推进历史、
   遗漏其他请求。现拒绝已提交的 request ID，保留此前合法提交。
2. **beam 扩展/缩减**：reorder_cache 未更新 _prompt_batch_size，2 行扩展到
   3 行后 commit 越界。现更新计数，阻止未完成 frontier 时重排。
3. **非法退出参数污染 cache**：先 begin_decode 后检查 exit_at_step，非法值留下
   活动 transaction。现先检查参数再开启 transaction。
4. **run-all 追踪错误**：tracer 忽略非 physical 的权威元数据，forced exit 时报告
   阈值反推的另一深度。现优先采用模型 selected/executed UT。
5. 补充 commit depth 范围、未显式指定请求时的 batch/frontier 检查；
   移除指向不存在 main 函数的 console script。
6. 真实 checkpoint 验证脚本默认 revision 固定为
   f1edd81e7ac41355db670500ceaf204e0f73af68。

前 3 项由新增测试在修复前复现：2 failures、1 error，修复后通过。
run-all tracer 另用独立 fake gate/metadata 验证权威深度与阈值反推的区别。

## 本次验证

- uv 管理并启动：Python 3.12.16、PyTorch 2.9.0+cpu、Transformers 4.57.6。
- 原有 19 项全部通过；新增/扩展后共 **27 项测试通过**。
- adaptive_kv_smoke.py：last-available 与 final-exit-shared 均 PASS。
- uv lock --check 通过。实际 CPU 测试使用 README 的独立 uv 环境，
  未同步完整项目锁定环境中的 CUDA、数据集等依赖。
- 实际输出见 baseline_cpu_validation.txt；复跑方法见根 README。

当前 nvidia-smi 无法连接 NVIDIA 驱动，没有本轮 GPU/BF16、2.6B checkpoint、
吞吐量、任务准确率或生产 serving 复跑结论。随分支选入的
adaptive_exit_kv_implementation.md 和 ordinary_batching_numerical_stability.md
是来源分支历史报告，其引用的 results/*.json 未随仓库提交，不能当成本轮结果。

历史 batching 报告观察到普通 BF16 B=4/B=8 在阈值边界附近改变 selected UT。
此 baseline 以同 cache policy 的参考等价与数值容差为标准，
不保证不同 batch shape 下的 UT 决策在所有真实模型输入上完全相同。

## 不支持或尚未确认

- Adaptive prefill、多 token decode continuation、异步请求推进、在线增删请求。
- EOS 后缩小 resident request 集合；组批要求全部 resident 请求同步推进。
- 部分 frontier 提交后的局部恢复；abort_decode 丢弃整个 pending frontier，
  调用方须从该 frontier 第一个 microbatch 重放。
- 动态组批时 tracer 自动维护全局 request 身份、真实 checkpoint beam generation。
- 训练、compile/CUDA graphs、分页 KV、FlashAttention 服务后端及性能优化结论。

后续模块以此分支为起点，分别提交正确性证据后再进入 baseline。
