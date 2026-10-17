# Rejected Version 2 KV Ablations

This file is the sole retained record of two Version 2 experiments built from
the frozen [Version 1](version1/README.md). Both implementations and their raw
result directories were deleted after profiling because neither improved
end-to-end adaptive decoding time.

## Common protocol

Both ablations used the same pinned Ouro-2.6B-Thinking checkpoint and workload
as Version 1:

- runtime `config.num_hidden_layers`: 48;
- 16 requests, B=4, UT=4, threshold 0.2;
- 128 positions (127 timed decode positions);
- physical adaptive exit, Last-Available KV, controlled Pred90 grouping, SUF;
- 3,990 adaptive row-recurrence steps;
- one warmup, two clean timing runs, then sparse CUDA profiling.

The Version 1 Pred90/SUF baseline was 90.2819 seconds wall time, 23.4289
seconds wall-equivalent KV CUDA time, and a 25.9508% KV share.

## Ablation 1: depth-indirected Last-Available KV

### Design

The first Version 2 stored each token's actual-depth KV once and represented
Last-Available history with source-depth metadata. Its goals were:

- replace Last-Available KV duplication with depth indirection;
- eliminate `torch.stack` during adaptive commit;
- eliminate per-commit `torch.full_like`, `torch.arange`, and temporary tensors;
- preallocate commit and gather workspaces;
- make adaptive commit a metadata update.

Prompt KV remained in the prefill cache, decoded KV used physical actual-depth
storage, and one gather workspace was shared by serial decoder layers. An
initial layout that copied prompt KV and retained one workspace per layer
exceeded A40 memory; the corrected layout avoided those allocations and
completed the full profile.

### Results

| Pred90/SUF metric | Version 1 | Depth-indirected V2 | Change |
|---|---:|---:|---:|
| Decode wall time | 90.2819 s | 97.5341 s | +8.03% |
| KV wall-equivalent CUDA time | 23.4289 s | 31.8693 s | +36.03% |
| KV share of wall time | 25.9508% | 32.6751% | +25.91% |
| `cudaMalloc` calls | 6,096 | 1,556 | -74.48% |
| `cudaStreamSynchronize` calls | 5,842 | 5,842 | 0.00% |
| `aten::index_select` calls | 399,764 | 569,436 | +42.44% |
| `aten::index_select` CUDA time | 5.668 s | 11.736 s | +107.06% |

### Conclusion

Commit-side allocation and KV duplication were successfully reduced, as shown
by the 74.5% reduction in `cudaMalloc`. However, eager Hugging Face attention
required depth-indirected history to be gathered into a contiguous tensor at
every decoder layer. The additional gather cost exceeded the commit savings,
causing an 8.03% wall-time regression. Depth indirection is only promising with
an attention kernel that consumes the indirection metadata directly.

## Ablation 2: zero-copy request-span attention

### Design

The second Version 2 retained Version 1 storage and removed this batch-shaped
history materialization before attention:

```text
key_buffer.index_select(0, global_request_rows)
```

Instead, the cache returned persistent storage plus request-span metadata. A
logical batch such as `[0, 1, 4, 5]` became two zero-copy views,
`buffer[0:2]` and `buffer[4:6]`. Attention ran once per contiguous span and only
the small attention outputs were concatenated. In this workload, just 35 of 508
scheduled microbatches were one contiguous span.

### Results

| Pred90/SUF metric | Version 1 | Request-span V2 | Change |
|---|---:|---:|---:|
| Decode wall time | 90.2819 s | 106.1372 s | +17.56% |
| KV wall-equivalent CUDA time | 23.4289 s | 20.2959 s | -13.37% |
| KV share of wall time | 25.9508% | 19.1223% | -26.31% |
| `aten::index_select` calls | 399,764 | 286,988 | -28.21% |
| Efficient-attention calls | 56,388 | 150,114 | +166.21% |
| Kernel launches | 7,753,096 | 8,441,944 | +8.88% |
| `cudaMemcpyAsync` calls | 9,652 | 97,441 | +909.54% |
| `cudaStreamSynchronize` calls | 5,842 | 7,525 | +28.81% |

Selected UT matched Version 1 exactly for all 16 requests and 127 positions.

### Conclusion

Removing batch-shaped K/V materialization reduced KV stage time by 13.37%, so
the targeted gather was a real cost. The eager per-span implementation was not
viable: attention calls increased 2.66x, and membership-to-slice translation
increased copy and synchronization activity. These costs produced a 17.56%
end-to-end regression.

## Combined conclusion

Both experiments point to the same boundary: metadata-based persistent KV is
not sufficient when conventional attention still requires a dense transient
batch tensor or multiple per-span calls. A useful successor should execute one
indexed or paged attention kernel per logical batch, consume request/depth
metadata on device, and avoid both full-history gathers and per-span launches.

Until such a kernel is available, Version 1 remains the retained implementation.
