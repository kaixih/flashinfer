from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    WeightLayout,
)

moe_op = get_trtllm_moe_sm100_module()
args = (
    int(DtypeTrtllmGen.Bfloat16),
    int(DtypeTrtllmGen.Bfloat16),
    int(Fp8QuantizationType.NoneFp8),
    8,  # top_k
    2048,  # hidden_size
    128,  # intermediate_size per TP partition
    128,  # local_num_experts
    int(ActivationType.Swiglu),
    True,  # use_shuffled_weight
    int(WeightLayout.BlockMajorK),
    False,  # use_per_token_scaling
    4,  # num_tokens
)
pairs = moe_op.trtllm_get_valid_moe_configs(*args)
print("count", len(pairs))
for i, pair in enumerate(pairs[:300]):
    print(i, [int(pair[0]), int(pair[1])])
