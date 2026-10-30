# Adaptive exit 与 last-available 核心语义

本文件描述当前 baseline；其他 policy 与 batching 调度不在此分支。

## 模型执行

prompt prefill 始终执行 config.total_ut_steps 个 UT；每个 UT 执行
config.num_hidden_layers 个 decoder layer。prefill 的输出仍按指定退出规则选择，
“全部计算”不自动表示“使用最后深度输出”。阈值 1 或显式指定最后深度可确保后者。

decode 每次输入一个 token。gate 的 sigmoid 给出条件退出概率 p_u，
剩余概率 r_u 与 p_u 相乘得到当前退出质量，并累计形成 CDF。首次达到阈值即选择退出；
最后深度强制退出。阈值 1 是显式 full-depth 模式，避免 sigmoid/CDF 数值饱和提前退出。
exit_at_step 是零基的强制深度参数，优先于阈值；返回的 UT 深度为一基。

physical_early_exit=True：已选中退出的行不再执行后续 decoder。
physical_early_exit=False：全部 UT 仍执行，但输出和提交的 KV 只采用所选退出语义。
两条路径采用相同 last-available 历史，作为物理停止是否正确的对照。

## 当前 token 与历史 KV

同一 decoder layer/UT 内先从层输入产生 Q、K、V，再执行 attention：

```text
begin_decode
    → 当前 K/V 暂存至 staged
    → attention(Q_current, concat(committed_history, current_KV))
    → 确定退出深度
    → commit
```

current_KV 会参与自身 attention；staged 不作为历史查询源。
跨 UT 不会把当前 token 的多个深度当成多个序列位置。

历史 token j 在当前深度 u 读取 KV_j[min(u, U_j)]。
prompt 的 U_j 为最大深度，因此按当前深度读取 prompt KV。
commit 验证每行从第一深度到选中深度的所有层 KV 齐全，再保留这些真实生成的 KV。
未执行的深度不会伪造或复制存储。后续查询时逐 token 解析 source depth 并组装 attention 张量。

这是调度引入前的稀疏语义参考结构，替代了上一版按目标深度预展开的 dense-history
和 partial-frontier 机制。没有 global request remapping、cache 合并或持久 dense-history 优化。
为清晰验证语义，查询与提交使用 Python 循环；不据此作 serving 性能或显存收益承诺。

## 生命周期与检查

- begin_decode 前要求每个深度/层的 prefill KV 齐全且长度、请求数匹配。
- commit 前检查退出深度范围与每行必需的 staged KV；历史只在成功 commit 后推进。
- abort_decode 清除当前 token 的 staged KV，不修改已经提交的历史。
- 模型在开启 transaction 前检查退出参数；decoder 发生异常会 abort，之后可重试。
- last_available 必须 use_cache=True；多 token continuation 与 policy 中途切换明确拒绝。
- 固定行身份及行内不同退出深度是张量正确性范围，没有动态组批接口。

## 已验证的边界

tests/test_adaptive_kv.py 与 tests/test_core_contract.py 验证：

- 多层、多个历史 token 的独立 KV oracle，同时检查 K 与 V。
- 当前 KV 自注意力、staged 不提前进入历史、abort、缺失深度拒绝提交、clear/reuse。
- forced/threshold 的 physical 与 run-all 逐步 logits、selected/executed UT 等价。
- padding、固定行不同退出深度；不涉及请求组批或调度。
- 阈值 0、0.5、0.75、1 的边界与实际 decoder 调用次数。
- gate 饱和时阈值 1 仍执行最后深度。
- 阈值 1 与 full-per-depth 完全等价，并与无缓存的完整前缀重新计算对照。
- 闭环 greedy token 一致性、异常后精确重试。

当前仅确认 CPU tiny-model 推理。未确认真实 checkpoint GPU/BF16、beam generation、
训练/weighted loss、compile/CUDA graphs、在线请求增删、分页 attention 或性能。
