# FlashInfer TRTLLM BF16 MoE BMM tactic sweep handoff

This folder records the May 29, 2026 investigation into a B200 illegal memory
access in pure SGLang:

```bash
python3 -m sglang.launch_server \
  --model-path /scratch/models/Qwen3-30B-A3B \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --moe-runner-backend flashinfer_trtllm \
  --cuda-graph-max-bs 1024 \
  --disable-flashinfer-autotune
```

The failure happens during piecewise CUDA graph capture on B200. The strongest
current signal is that the failing launch is FlashInfer TRTLLM BF16 MoE GEMM2
for:

```text
m=4, n=2048, k=128, numTokens=4, topK=8, localNumExperts=128
```

The pointer trace shows:

```text
GEMM1:
  m=4, n=256,  k=2048, configIndex=1343
  A = gemm1_weights
  B = hidden_states

GEMM2:
  m=4, n=2048, k=128,  configIndex=1434 by default
  A = gemm2_weights
  B = workspace.gemm1_output
```

All traced pointers were 256B-aligned. Compute sanitizer/coredump evidence
pointed at the `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128_..._sm100f`
kernel family.

## Branch State

Branch:

```text
kaixi/debug-external-moe-workspace-v0611post1
```

Relevant debug changes:

- `csrc/trtllm_batched_gemm_runner.cu`
  - host-side GEMM1/GEMM2 pointer trace
  - valid BMM kernel listing
  - env-gated blacklist for target GEMM2 kernels
- `csrc/trtllm_fused_moe_runner.cu`
  - selected MoE config trace and fused-runner pointer trace
- `csrc/trtllm_fused_moe_kernel_launcher.cu`
  - force MoE tactic / print valid MoE configs
- `flashinfer/fused_moe/core.py`
  - Python-side force tactic plumbing
- `flashinfer/jit/core.py`
  - env gate to force the TRTLLM fused MoE JIT path

## Install

Inside the SGLang dev container on a B200 node:

```bash
cd /scratch/repo/flashinfer
FLASHINFER_CUDA_ARCH_LIST=10.0a MAX_JOBS=16 \
  bash lab/bf16_bmm_ptr_trace_20260529/01.install_flashinfer_editable.sh
```

The install script verifies that `flashinfer-python` imports from
`/scratch/repo/flashinfer` and that the force-JIT marker is present.

## Main Repro

From `/scratch/repo/sglang`:

```bash
TIMEOUT_SEC=2400 \
RUN_ID=sglang-pure-piecewise-maxbs1024-bf16-valid-configs-$(date +%Y%m%d-%H%M%S) \
bash /scratch/repo/flashinfer/lab/bf16_moe_tactic_sweep_20260529/01.run_default_valid_configs.sh
```

The script writes logs under:

```text
/scratch/repro/miles-b200-qwen3-30b/logs/<RUN_ID>
```

Useful files:

- `server.log`
- `summary.txt`
- `versions.txt`
- `flashinfer_env.txt`
- `server.cmd`

## Force A Tactic

The FlashInfer Python API takes a two-element tactic:

```text
[tile_N, moe_tactic]
```

Example:

```bash
FORCE_TACTIC=8,143 \
TIMEOUT_SEC=2400 \
bash /scratch/repo/flashinfer/lab/bf16_moe_tactic_sweep_20260529/02.run_force_tactic.sh
```

Important: forcing another MoE config can change the exact `s*` schedule variant
but still remain in the same broad BMM kernel family.

## Known Results

Default selection for the target GEMM2 shape:

```text
moeConfigIndex=8
gemm1Config=1343
gemm2Config=1434
GEMM2 function includes:
  bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128_s6_..._schPd2x1x2x3_..._sm100f
```

Representative default/pointer-trace run:

```text
/scratch/repro/miles-b200-qwen3-30b/logs/sglang-pure-piecewise-maxbs1024-bmm-ptr-trace-...
```

Representative forced tactic + compute sanitizer run:

```text
/scratch/repro/miles-b200-qwen3-30b/logs/sglang-pure-piecewise-maxbs1024-bf16-forcecfg143-compute-sanitizer-20260529-204802
```

Compute sanitizer reported a warp illegal address in:

```text
bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128_s4_..._schPd2x1x2x3_..._sm100f
```

So tactic forcing changed `s6` to `s4`, but did not escape the same broad BMM
family.

## Blacklist Experiments

The blacklist is implemented in `TrtllmGenBatchedGemmRunner::getValidConfigIndices`
before a valid config is pushed into `validConfigIndices`.

Targeted shape gate:

```cpp
!mOptions.routeAct && m == 4 && n == 2048 && k == 128 && numTokens == 4
```

Two env gates exist:

```bash
FLASHINFER_DEBUG_SKIP_TRTLLM_BMM_T128X8X128_SCHPD=1
FLASHINFER_DEBUG_SKIP_TRTLLM_BMM_T128X8X128_ALL=1
```

### Skip only `schPd2x1x2x3`

Run:

```text
/scratch/repro/miles-b200-qwen3-30b/logs/sglang-pure-piecewise-maxbs1024-bf16-skip-schpd-clean-20260529-213234
```

Result:

```text
Skipped: 1434, 1436, 1438
Fallback selected: gemm2Config=1431
1431 is still t128x8x128, but scheduler=schedS
Result: still IMA
```

Conclusion: avoiding only `schPd2x1x2x3` is not sufficient.

### Skip all `t128x8x128`

Run:

```text
/scratch/repro/miles-b200-qwen3-30b/logs/sglang-pure-piecewise-maxbs1024-bf16-skip-all-t128x8x128-20260529-213812
```

Result:

```text
Skipped: 1434, 1436, 1438, 1431, 1433, 1435
Failure: No valid config found for the given problem shape
```

Conclusion: for this target GEMM2 shape, the current valid FlashInfer/TRTLLM BF16
BMM config space appears to require `t128x8x128`. There is no non-`t128x8x128`
fallback exposed by the current valid-config list.

## How To Continue

Recommended next steps for another agent:

1. Keep the pure SGLang repro. It is faster and removes Miles/RL update from the
   immediate question.
2. Do not spend more time sweeping MoE tactic values unless the sweep prints the
   underlying BMM function name. Many tactic values still map to the same broad
   BMM family.
3. If FlashInfer team wants more data, provide:
   - pointer trace for GEMM1/GEMM2
   - compute sanitizer output
   - target shape and config indices above
   - blacklist evidence showing no non-`t128x8x128` fallback
4. The most useful kernel-side question is now why the `t128x8x128` BF16 BMM
   family can hit IMA for this `m=4,n=2048,k=128,numTokens=4` graph-capture case.

## Clean-Up Notes

The repo may contain unrelated untracked third-party directories from prior
builds. Do not commit those. The relevant lab folders for this investigation are:

```text
lab/bf16_bmm_ptr_trace_20260529/
lab/bf16_moe_tactic_sweep_20260529/
```
