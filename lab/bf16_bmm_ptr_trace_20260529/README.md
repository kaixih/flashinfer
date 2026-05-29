# BF16 TRTLLM BMM Pointer Trace

Debug-only host trace for the B200 FlashInfer TRTLLM BF16 MoE IMA repro.

The trace is gated by:

```bash
FLASHINFER_DEBUG_TRTLLM_BMM_PTRS=1
FLASHINFER_DEBUG_TRTLLM_BMM_PTRS_LIMIT=4000
FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM=1
```

Every line starts with `FI_TRTLLM_BMM_PTR`.

`site=fused_moe_runner` prints the semantic mapping before the wrapper calls the batched GEMM runner:

- `stage=GEMM1`: `mPtrA=gemm1_weights`, `mPtrB=hidden_states`
- `stage=GEMM2`: `mPtrA=gemm2_weights`, `mPtrB=workspace.gemm1_output`

`site=batched_gemm_runner` prints the actual values assigned to
`gemmData.mInputBuffers.mPtrA` and `gemmData.mInputBuffers.mPtrB` immediately
after those fields are set. Both traces include `%16/%32/%64/%128/%256`
alignment mods plus problem/config metadata.

Run inside the `sglang_dev` container:

```bash
cd /scratch/repo/flashinfer
bash lab/bf16_bmm_ptr_trace_20260529/01.install_flashinfer_editable.sh

cd /scratch/repo/sglang
TIMEOUT_SEC=2400 \
RUN_ID=sglang-pure-piecewise-maxbs1024-bmm-ptr-trace-$(date +%Y%m%d-%H%M%S) \
FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM=1 \
FLASHINFER_DEBUG_TRTLLM_BMM_PTRS=1 \
FLASHINFER_DEBUG_TRTLLM_BMM_PTRS_LIMIT=4000 \
bash lab/flashinfer_trtllm_bf16_ima_repro/01.run_default_piecewise_maxbs1024.sh
```
