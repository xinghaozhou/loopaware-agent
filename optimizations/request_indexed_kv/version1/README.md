# Version 1: Request-Indexed In-Place KV Optimization

## Goal

Keep physical adaptive exit and token-level dynamic membership while decoupling
persistent KV state from transient batch layout. Changing a scheduled batch
becomes an update to request-ID metadata rather than a reorganization of stored
KV tensors.

This directory is intentionally self-contained. The reference files under
`vendor/ouro_hf` remain unchanged. `profile_optimized.py` installs the copied
cache factory only for this experiment process.

## Implemented changes

| Optimization | Original Ouro | Pred90/SUF adaptive | Rationale |
|---|:---:|:---:|---|
| Geometric sequence capacity | Yes | Yes | Avoid growing the complete history with `torch.cat` every token. |
| In-place KV append | Yes | Yes | Write only the new token at its sequence position. |
| Stable request-indexed storage | No | Yes | Persistent rows retain global request identity across regrouping. |
| Membership represented by request IDs | No | Yes | Regrouping changes an index tensor, not persistent KV layout. |
| Remove staged `slot[keep]` filtering | No | Yes | Avoid dynamically sized boolean indexing and its implicit synchronization. |
| Fixed-shape Last-Available resolution | No | Yes | Resolve each new token across depths with request/depth indices at commit. |
| Physical adaptive exit retained | N/A | Yes | Inactive rows stop recurrent computation exactly as before. |
| Pred90 homogeneous grouping and SUF retained | N/A | Yes | Scheduler behavior and controlled predictions are unchanged. |

## Storage behavior

Original Ouro uses one geometrically allocated buffer for every recurrent-depth
and decoder-layer pair. Adaptive Pred90 uses the same logical depth/layer
separation, but every buffer is permanently indexed by global request ID.

```text
persistent KV[depth, layer, request_id, sequence_position]
                                  ^
                                  stable across all scheduled microbatches

transient microbatch = [request_id_3, request_id_9, request_id_1, ...]
```

At commit, Last-Available semantics resolve the current token with
`source_depth = min(target_depth, selected_exit_depth)` using fixed-shape GPU
indexing. Only the current sequence position is written. Historical positions
are never rebuilt or copied because batch membership changed.

The eager Hugging Face attention implementation still materializes the selected
request rows with `index_select`. Eliminating that final gather requires a
paged/request-indirect attention kernel and is outside this isolated experiment.

## Validation and profile protocol

The optimized comparison uses exactly the existing pinned workload and profile
protocol:

- `ByteDance/Ouro-2.6B-Thinking`, pinned revision;
- runtime `config.num_hidden_layers`;
- UT=4, threshold 0.2, B=4;
- eight GSM8K plus eight MATH-500 requests;
- 128 fixed output positions;
- controlled Pred90 homogeneous grouping plus SUF;
- clean wall-clock runs followed by sparse coarse CUDA profiling.

Run tests:

```bash
uv run python -m unittest \
  optimizations.request_indexed_kv.version1.test_optimized_cache -v
```

Run the profile:

```bash
uv run python optimizations/request_indexed_kv/version1/profile_optimized.py \
  --trace-dir results/predictor_trace_ut4_128_20261017 \
  --output-dir results/request_indexed_kv_profile \
  --positions 128 --warmups 1 --timing-runs 2 --resume
```

The final report will compare clean wall time and coarse KV-management time with
the unmodified control profile.
