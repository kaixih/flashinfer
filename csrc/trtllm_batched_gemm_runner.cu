/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"
// #include "tensorrt_llm/common/assert.h"
#include "flashinfer/exception.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmInterface.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/common.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

namespace tensorrt_llm {
namespace kernels {

using namespace batchedGemm::batchedGemm;
using namespace batchedGemm::gemm;
using namespace batchedGemm::trtllm::gen;

static BatchedGemmInterface::ModuleCache globalTrtllmGenBatchedGemmModuleCache;

namespace {

bool isBmmPtrTraceEnabled() {
  char const* value = std::getenv("FLASHINFER_DEBUG_TRTLLM_BMM_PTRS");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

int32_t bmmPtrTraceLimit() {
  char const* value = std::getenv("FLASHINFER_DEBUG_TRTLLM_BMM_PTRS_LIMIT");
  if (value == nullptr || value[0] == '\0') {
    return 4000;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed < 0) {
    return 4000;
  }
  return static_cast<int32_t>(parsed);
}

bool shouldTraceBmmPtr(int32_t* traceIndex) {
  static std::atomic<int32_t> count{0};
  if (!isBmmPtrTraceEnabled()) {
    return false;
  }
  int32_t index = count.fetch_add(1, std::memory_order_relaxed);
  if (index >= bmmPtrTraceLimit()) {
    return false;
  }
  *traceIndex = index + 1;
  return true;
}

unsigned long long ptrMod(void const* ptr, std::uintptr_t alignment) {
  return static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(ptr) % alignment);
}

void traceGemmDataInputPtrs(char const* stage, char const* ptrAName, void const* ptrA,
                            char const* ptrBName, void const* ptrB, int32_t m, int32_t n, int32_t k,
                            int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
                            int32_t configIndex, CUstream stream, int device,
                            bool transposeMmaOutput, bool routeAct) {
  int32_t traceIndex = 0;
  if (!shouldTraceBmmPtr(&traceIndex)) {
    return;
  }
  std::fprintf(stderr,
               "FI_TRTLLM_BMM_PTR site=batched_gemm_runner stage=%s traceIndex=%d "
               "gemmData.mInputBuffers.mPtrA_name=%s gemmData.mInputBuffers.mPtrA=%p "
               "mPtrA_mod16=%llu mPtrA_mod32=%llu mPtrA_mod64=%llu mPtrA_mod128=%llu "
               "mPtrA_mod256=%llu gemmData.mInputBuffers.mPtrB_name=%s "
               "gemmData.mInputBuffers.mPtrB=%p mPtrB_mod16=%llu mPtrB_mod32=%llu "
               "mPtrB_mod64=%llu mPtrB_mod128=%llu mPtrB_mod256=%llu m=%d n=%d k=%d "
               "numTokens=%d numBatches=%d maxNumCtasInBatchDim=%d configIndex=%d stream=%p "
               "device=%d transposeMmaOutput=%d routeAct=%d\n",
               stage, traceIndex, ptrAName, ptrA, ptrMod(ptrA, 16), ptrMod(ptrA, 32),
               ptrMod(ptrA, 64), ptrMod(ptrA, 128), ptrMod(ptrA, 256), ptrBName, ptrB,
               ptrMod(ptrB, 16), ptrMod(ptrB, 32), ptrMod(ptrB, 64), ptrMod(ptrB, 128),
               ptrMod(ptrB, 256), m, n, k, numTokens, numBatches, maxNumCtasInBatchDim, configIndex,
               static_cast<void*>(stream), device, static_cast<int>(transposeMmaOutput),
               static_cast<int>(routeAct));
}

}  // namespace

std::vector<int64_t> prioritizePredefinedConfigs(
    int m, int n, int k, std::vector<int64_t> const& sortedIndices,
    batchedGemm::batchedGemm::BatchedGemmConfig const* configs) {
  // Function to bubble up the pre-determined config.
  auto bubbleUpConfig = [&configs](std::vector<int64_t> const& sortedIndices,
                                   auto&& pred) -> std::vector<int64_t> {
    std::vector<int64_t> prioritizedIndices_;
    // Copy matching configs to new vector
    std::copy_if(sortedIndices.begin(), sortedIndices.end(),
                 std::back_inserter(prioritizedIndices_), [&configs, &pred](int idx) {
                   BatchedGemmConfig const& config = configs[idx];
                   return (pred(config));
                 });
    // Copy the rest of the configs to new vector, if not already copied
    std::copy_if(sortedIndices.begin(), sortedIndices.end(),
                 std::back_inserter(prioritizedIndices_), [&prioritizedIndices_](int idx) {
                   return std::find(prioritizedIndices_.begin(), prioritizedIndices_.end(), idx) ==
                          prioritizedIndices_.end();
                 });
    return prioritizedIndices_;
  };

  // Init empty vector
  std::vector<int64_t> prioritizedIndices;

  //
  // Dummy
  //

  if (n /* out_dim */ == 0 && k /* in_dim */ == 0) {
    auto pred = [](BatchedGemmConfig const& config) {
      BatchedGemmOptions const& options = config.mOptions;
      return options.mNumStagesA == 4 && options.mNumStagesB == 4 && options.mNumStagesMma == 2 &&
             options.mTileK == 256 && options.mTileScheduler == TileScheduler::Persistent;
    };
    prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
  }
  //
  // Fall back
  //
  else {
    prioritizedIndices = sortedIndices;
  }

  return prioritizedIndices;
}

TrtllmGenBatchedGemmRunner::TrtllmGenBatchedGemmRunner(
    TrtllmGenBatchedGemmRunnerOptions const& options_)
    : mOptions(options_) {
  // Select a GEMM kernel config to use
  auto const bmm = BatchedGemmInterface();
  auto const configs = bmm.getBatchedGemmConfigs();

  mPassingConfigIndices.clear();

  for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i) {
    auto const options = configs[i].mOptions;
    auto const tileSize = mOptions.transposeMmaOutput ? options.mTileN : options.mTileM;
    // When we include low-latency kernels we can set transposeMmaOutput via constructor
    if (options.mDtypeA == mOptions.dtypeA && options.mDtypeB == mOptions.dtypeB &&
        options.mDtypeC == mOptions.dtypeC && options.mUseDeepSeekFp8 == mOptions.deepSeekFp8 &&
        options.mTransposeMmaOutput == mOptions.transposeMmaOutput &&
        (!doesRouteImplUseNoRoute(options.mRouteImpl)) == mOptions.routeAct &&
        options.mFusedAct == mOptions.fusedAct && options.mIsStaticBatch == mOptions.staticBatch &&
        tileSize == mOptions.tileSize && options.mUseShuffledMatrix == mOptions.useShuffledMatrix &&
        options.mLayoutA == mOptions.weightLayout) {
      if (mOptions.usePerTokenScaling) {
        if (options.mTransposeMmaOutput && !options.mUsePerTokenSfB) continue;
        if (!options.mTransposeMmaOutput && !options.mUsePerTokenSfA) continue;
      }
      if (mOptions.usePerChannelScaling) {
        if (options.mTransposeMmaOutput && !options.mUsePerTokenSfA) continue;
        if (!options.mTransposeMmaOutput && !options.mUsePerTokenSfB) continue;
      }
      if (options.mFusedAct) {
        if (options.mActType != static_cast<batchedGemm::gemmGatedAct::ActType>(mOptions.actType)) {
          continue;
        }
      }
      if ((int64_t)options.mEltwiseActType != (int64_t)mOptions.eltwiseActType) {
        continue;
      }

      if (mOptions.transposeMmaOutput && options.mEpilogueTileM == mOptions.epilogueTileM) {
        // Skip cubins with clusterZ > 1 due to correctness issues described in
        // https://github.com/flashinfer-ai/flashinfer/issues/3197
        if (options.mClusterDimZ > 1) continue;
        mPassingConfigIndices.push_back(i);
      }
    }
  }

  std::ostringstream error_msg;
  error_msg << "No kernel found for the given options: "
            << "mDtypeA: " << tg::dtypeToString(mOptions.dtypeA)
            << ", mDtypeB: " << tg::dtypeToString(mOptions.dtypeB)
            << ", mDtypeC: " << tg::dtypeToString(mOptions.dtypeC)
            << ", mUseDeepSeekFp8: " << mOptions.deepSeekFp8
            << ", mActType: " << (int64_t)mOptions.actType
            << ", mEltwiseActType: " << (int64_t)mOptions.eltwiseActType
            << ", mTransposeMmaOutput: " << mOptions.transposeMmaOutput
            << ", mRouteAct: " << mOptions.routeAct << ", mFusedAct: " << mOptions.fusedAct
            << ", mIsStaticBatch: " << mOptions.staticBatch << ", mTileSize: " << mOptions.tileSize;
  FLASHINFER_CHECK(!mPassingConfigIndices.empty(), error_msg.str());
}

size_t TrtllmGenBatchedGemmRunner::getWorkspaceSizeInBytes(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim, int32_t configIndex) const {
  BatchedGemmData gemmData{};
  gemmData.mProblemDimensions.mNumBatches = numBatches;
  gemmData.mProblemDimensions.mNumTokens = numTokens;
  gemmData.mProblemDimensions.mBatchM = !mOptions.transposeMmaOutput;
  gemmData.mProblemDimensions.mBatchedM =
      mOptions.transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
  gemmData.mProblemDimensions.mBatchedN =
      mOptions.transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
  gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
  gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mProblemDimensions.mRank = 0;
  gemmData.mProblemDimensions.mWorldSize = 1;
  gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;

  gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
  gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
  gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;

  auto bmm = BatchedGemmInterface();

  auto const configs = bmm.getBatchedGemmConfigs();

  auto const& config = configs[configIndex];

  return bmm.getWorkspaceSizeInBytes(config, gemmData);
}

void TrtllmGenBatchedGemmRunner::run(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim, void const* a, void const* sfA, void const* b,
    void const* sfB, void const* perTokensSfA, void const* perTokensSfB, float const* scaleC,
    float const* scaleGateC, float const* ptrBias, float const* ptrAlpha, float const* ptrBeta,
    float const* ptrClampLimit, void* c, void* outSfC, int32_t const* routeMap,
    int32_t const* totalNumPaddedTokens, int32_t const* ctaIdxXyToBatchIdx,
    int32_t const* ctaIdxXyToMnLimit, int32_t const* numNonExitingCtas, void* workspace,
    CUstream stream, int device, int32_t configIndex, bool enable_pdl) {
  auto bmm = BatchedGemmInterface();

  BatchedGemmData gemmData{};

  auto const configs = bmm.getBatchedGemmConfigs();

  auto const& config = configs[configIndex];
  // printf("running config %d: %s\n", configIndex, config.mFunctionName);

  FLASHINFER_CHECK(numBatches > 0, "Batched GEMM requires numBatches > 0");
  if (!mOptions.staticBatch) {
    FLASHINFER_CHECK(totalNumPaddedTokens,
                     "Batched GEMM with dynamic batching requires totalNumPaddedTokens");
    FLASHINFER_CHECK(ctaIdxXyToBatchIdx,
                     "Batched GEMM with dynamic batching requires ctaIdxXyToBatchIdx");
    FLASHINFER_CHECK(ctaIdxXyToMnLimit,
                     "Batched GEMM with dynamic batching requires ctaIdxXyToMnLimit");
    FLASHINFER_CHECK(numNonExitingCtas,
                     "Batched GEMM with dynamic batching requires numNonExitingCtas");
  }

  if (!mOptions.staticBatch && numTokens != 0) {
    FLASHINFER_CHECK(maxNumCtasInBatchDim > 0,
                     "Batched GEMM with dynamic batching requires maxNumCtasInBatchDim > 0");
  }

  if (mOptions.routeAct) {
    FLASHINFER_CHECK(routeMap, "Batched GEMM with routeAct requires routeMap");
    FLASHINFER_CHECK(numTokens > 0, "Batched GEMM with routeAct requires numTokens > 0");
  }

  // Dims
  gemmData.mProblemDimensions.mNumBatches = numBatches;
  gemmData.mProblemDimensions.mNumTokens = numTokens;
  gemmData.mProblemDimensions.mBatchM = !mOptions.transposeMmaOutput;
  gemmData.mProblemDimensions.mBatchedM =
      mOptions.transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
  gemmData.mProblemDimensions.mBatchedN =
      mOptions.transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
  gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
  gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
  gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
  gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
  gemmData.mProblemDimensions.mRank = 0;
  gemmData.mProblemDimensions.mWorldSize = 1;

  // Inputs
  gemmData.mInputBuffers.mPtrA = mOptions.transposeMmaOutput ? b : a;
  gemmData.mInputBuffers.mPtrSfA = mOptions.transposeMmaOutput ? sfB : sfA;
  gemmData.mInputBuffers.mPtrB = mOptions.transposeMmaOutput ? a : b;
  traceGemmDataInputPtrs(
      mOptions.routeAct ? "GEMM1" : "GEMM2", mOptions.routeAct ? "gemm1_weights" : "gemm2_weights",
      gemmData.mInputBuffers.mPtrA, mOptions.routeAct ? "hidden_states" : "workspace.gemm1_output",
      gemmData.mInputBuffers.mPtrB, m, n, k, numTokens, numBatches, maxNumCtasInBatchDim,
      configIndex, stream, device, mOptions.transposeMmaOutput, mOptions.routeAct);
  gemmData.mInputBuffers.mPtrSfB = mOptions.transposeMmaOutput ? sfA : sfB;
  gemmData.mInputBuffers.mPtrScaleC = scaleC;
  gemmData.mInputBuffers.mPtrScaleGate = scaleGateC;
  // For simplicity pass set scaleAct to scaleGateC
  gemmData.mInputBuffers.mPtrScaleAct = scaleGateC;
  gemmData.mInputBuffers.mPtrPerTokenSfA =
      mOptions.transposeMmaOutput ? perTokensSfB : perTokensSfA;
  gemmData.mInputBuffers.mPtrPerTokenSfB =
      mOptions.transposeMmaOutput ? perTokensSfA : perTokensSfB;
  gemmData.mInputBuffers.mPtrBias = ptrBias;
  gemmData.mInputBuffers.mPtrGatedActAlpha = ptrAlpha;
  gemmData.mInputBuffers.mPtrGatedActBeta = ptrBeta;
  gemmData.mInputBuffers.mPtrClampLimit = ptrClampLimit;

  gemmData.mInputBuffers.mPtrRouteMap = routeMap;

  gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;

  // Pointer to total number of padded tokens
  gemmData.mInputBuffers.mPtrTotalNumPaddedTokens = totalNumPaddedTokens;
  gemmData.mInputBuffers.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
  gemmData.mInputBuffers.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
  gemmData.mInputBuffers.mPtrNumNonExitingCtas = numNonExitingCtas;

  // Outputs
  gemmData.mOutputBuffers.mPtrC = c;
  gemmData.mOutputBuffers.mPtrSfC = outSfC;

  int32_t multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

  // FIXME once we start using all-reduce in the epilogue of the bmm this can be moved elsewhere
  bmm.runInitBeforeWorldSync(config, gemmData, static_cast<void*>(stream));

  auto const err =
      bmm.run(config, workspace, gemmData, static_cast<void*>(stream), multiProcessorCount,
              enable_pdl, /*pinnedHostBuffer=*/nullptr, globalTrtllmGenBatchedGemmModuleCache);

  FLASHINFER_CHECK(err == 0,
                   "Error occurred when running GEMM!"
                   " (numBatches: ",
                   numBatches, ", GemmMNK: ", m, " ", n, " ", k, ", Kernel: ", config.mFunctionName,
                   ")");
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k,
                                     std::vector<int32_t> const& batchedTokens, void const* a,
                                     void const* sfA, void const* b, void const* sfB, void* c,
                                     void* outSfC, void* workspace, CUstream stream, int device,
                                     int32_t configIndex, bool enable_pdl) {
  // Dispatch with block scaling factors and with static batching.
  run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0,
      a, sfA, b, sfB,
      /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
      /* scaleC */ nullptr, /* scaleGateC */ nullptr, /* ptrBias */ nullptr, /* ptrAlpha */ nullptr,
      /* ptrBeta */ nullptr, /* ptrClampLimit */ nullptr, c, outSfC,
      /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
      /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
      /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex, enable_pdl);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k,
                                     std::vector<int32_t> const& batchedTokens, void const* a,
                                     void const* sfA, void const* b, void const* sfB,
                                     float const* ptrBias, float const* ptrAlpha,
                                     float const* ptrBeta, float const* ptrClampLimit, void* c,
                                     void* outSfC, void* workspace, CUstream stream, int device,
                                     int32_t configIndex, bool enable_pdl) {
  // Dispatch with block scaling factors and with static batching.
  run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0,
      a, sfA, b, sfB,
      /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
      /* scaleC */ nullptr, /* scaleGateC */ nullptr, ptrBias, ptrAlpha, ptrBeta, ptrClampLimit, c,
      outSfC,
      /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
      /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
      /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex, enable_pdl);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k,
                                     std::vector<int32_t> const& batchedTokens, void const* a,
                                     void const* b, float const* scaleC, float const* scaleGateC,
                                     void* c, void* workspace, CUstream stream, int device,
                                     int32_t configIndex, bool enable_pdl) {
  // Dispatch with block scaling factors and with static batching.
  run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0,
      a,
      /* sfA */ nullptr, b, /* sfB */ nullptr, /* perTokensSfA */ nullptr,
      /* perTokensSfB */ nullptr, scaleC, scaleGateC, /* ptrBias */ nullptr, /* ptrAlpha */ nullptr,
      /* ptrBeta */ nullptr, /* ptrClampLimit */ nullptr, c,
      /* outSfC */ nullptr,
      /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
      /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
      /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex, enable_pdl);
}

std::vector<int64_t> TrtllmGenBatchedGemmRunner::getValidConfigIndices(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim) const {
  auto const bmm = BatchedGemmInterface();
  auto const configs = bmm.getBatchedGemmConfigs();

  int32_t multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

  BatchedGemmData gemmData{};
  // Dims
  gemmData.mProblemDimensions.mNumBatches = numBatches;
  gemmData.mProblemDimensions.mNumTokens = numTokens;
  gemmData.mProblemDimensions.mBatchM = !mOptions.transposeMmaOutput;
  gemmData.mProblemDimensions.mBatchedM =
      mOptions.transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
  gemmData.mProblemDimensions.mBatchedN =
      mOptions.transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
  gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
  gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mProblemDimensions.mRank = 0;
  gemmData.mProblemDimensions.mWorldSize = 1;
  gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;

  gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
  gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
  gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;

  auto cmpFunc = [&configs, &gemmData, &bmm, &multiProcessorCount](int64_t idx0, int64_t idx1) {
    auto const& optionsA = configs[idx0].mOptions;
    auto const& optionsB = configs[idx1].mOptions;
    int32_t sizeK = gemmData.mProblemDimensions.mK;

    // Tier 0: K < tileK, prefer higher efficiency.
    if (optionsA.mTileK != optionsB.mTileK) {
      // Both waste computation, prefer higher efficiency.
      if (sizeK <= optionsA.mTileK && sizeK <= optionsB.mTileK) {
        double eff_a = (double)sizeK / optionsA.mTileK;
        double eff_b = (double)sizeK / optionsB.mTileK;
        return eff_a > eff_b;
      }
      // If either can be utilized, sort by tileK.
      else {
        return optionsA.mTileK > optionsB.mTileK;
      }
    }

    // Tier 1: When tileK is the same, prefer unroll loop 2x for mma.
    if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma) {
      return optionsA.mUseUnrollLoop2xForMma;
    }

    // Tier 2+: When previous comparators are the same, prefer higher tileM.
    if (optionsA.mTileM != optionsB.mTileM) {
      return optionsA.mTileM > optionsB.mTileM;
    }

    // Tier 2+: When previous comparators are the same, prefer higher tileN.
    if (optionsA.mTileN != optionsB.mTileN) {
      return optionsA.mTileN > optionsB.mTileN;
    }

    // Tier 2+: When previous comparators are the same, and when the number of estimated CTAs is on
    // the larger side, prefer persistent tile scheduler.
    if (optionsA.mTileScheduler != optionsB.mTileScheduler) {
      auto options = bmm.getOptionsFromConfigAndData(configs[idx0], gemmData);
      auto numCtas = bmm.getNumCtas(options, gemmData.mProblemDimensions.mMaxNumCtasInTokenDim);
      if (numCtas > multiProcessorCount) {
        return optionsA.mTileScheduler == batchedGemm::gemm::TileScheduler::Persistent;
      } else {
        return optionsB.mTileScheduler == batchedGemm::gemm::TileScheduler::Persistent;
      }
    }

    return false;
  };

  // Sort configs by options.
  std::vector<int64_t> sortedIndices = mPassingConfigIndices;
  std::sort(sortedIndices.begin(), sortedIndices.end(), cmpFunc);

  // Special rules for corner cases, if applicable.
  std::vector<int64_t> prioritizedIndices =
      prioritizePredefinedConfigs(m, n, k, sortedIndices, configs);

  // Filter out invalid configs.
  std::vector<int64_t> validConfigIndices;
  for (auto const& configIndex : prioritizedIndices) {
    auto isValidConfig = bmm.isValidConfig(configs[configIndex], gemmData);
    if (isValidConfig) {
      if (!mOptions.routeAct && m == 4 && n == 2048 && k == 128 && numTokens == 4) {
        auto const& options = configs[configIndex].mOptions;
        auto const* function_name = configs[configIndex].mFunctionName;
        bool const is_t128x8x128 =
            options.mTileM == 128 && options.mTileN == 8 && options.mTileK == 128;
        bool const skip_schpd =
            std::getenv("FLASHINFER_DEBUG_SKIP_TRTLLM_BMM_T128X8X128_SCHPD") != nullptr &&
            is_t128x8x128 && std::strstr(function_name, "schPd2x1x2x3") != nullptr;
        bool const skip_all_t128x8x128 =
            std::getenv("FLASHINFER_DEBUG_SKIP_TRTLLM_BMM_T128X8X128_ALL") != nullptr &&
            is_t128x8x128;
        if (skip_schpd || skip_all_t128x8x128) {
          std::fprintf(stderr,
                       "FI_TRTLLM_BMM_SKIP_KERNEL stage=GEMM2 configIndex=%ld "
                       "m=%d n=%d k=%d numTokens=%d tileM=%d tileN=%d tileK=%d "
                       "functionName=%s\n",
                       static_cast<long>(configIndex), m, n, k, numTokens, options.mTileM,
                       options.mTileN, options.mTileK, function_name);
          continue;
        }
      }
      validConfigIndices.push_back(configIndex);
    }
  }

  FLASHINFER_CHECK(!validConfigIndices.empty(),
                   "No valid config found for the given problem shape");

  if (std::getenv("FLASHINFER_DEBUG_TRTLLM_BMM_VALID_KERNELS") != nullptr) {
    int32_t print_limit = 16;
    if (char const* limit_env = std::getenv("FLASHINFER_DEBUG_TRTLLM_BMM_VALID_KERNELS_LIMIT")) {
      char* end = nullptr;
      long parsed = std::strtol(limit_env, &end, 10);
      if (end != limit_env && parsed >= 0) {
        print_limit = static_cast<int32_t>(parsed);
      }
    }
    bool const target_gemm2_shape =
        !mOptions.routeAct && m == 4 && n == 2048 && k == 128 && numTokens == 4;
    bool const print_all = std::getenv("FLASHINFER_DEBUG_TRTLLM_BMM_VALID_KERNELS_ALL") != nullptr;
    if (print_all || target_gemm2_shape) {
      static std::atomic<int32_t> valid_kernel_print_count{0};
      int32_t print_index = valid_kernel_print_count.fetch_add(1, std::memory_order_relaxed);
      if (print_index < print_limit) {
        std::fprintf(stderr,
                     "FI_TRTLLM_BMM_VALID_KERNELS traceIndex=%d stage=%s m=%d n=%d k=%d "
                     "numTokens=%d numBatches=%d maxNumCtasInBatchDim=%d valid_count=%zu\n",
                     print_index + 1, mOptions.routeAct ? "GEMM1" : "GEMM2", m, n, k, numTokens,
                     numBatches, maxNumCtasInBatchDim, validConfigIndices.size());
        for (auto const& valid_index : validConfigIndices) {
          auto const& valid_config = configs[valid_index];
          auto const& options = valid_config.mOptions;
          std::fprintf(stderr,
                       "FI_TRTLLM_BMM_VALID_KERNEL stage=%s configIndex=%ld "
                       "tileM=%d tileN=%d tileK=%d epilogueTileM=%d clusterDimZ=%d "
                       "scheduler=%d unroll2x=%d functionName=%s\n",
                       mOptions.routeAct ? "GEMM1" : "GEMM2", static_cast<long>(valid_index),
                       options.mTileM, options.mTileN, options.mTileK, options.mEpilogueTileM,
                       options.mClusterDimZ, static_cast<int>(options.mTileScheduler),
                       static_cast<int>(options.mUseUnrollLoop2xForMma),
                       valid_config.mFunctionName);
        }
      }
    }
  }

  return validConfigIndices;
}

int64_t TrtllmGenBatchedGemmRunner::getDefaultValidConfigIndex(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim) const {
  auto const validConfigIndices =
      getValidConfigIndices(m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim);

  return validConfigIndices[0];
}

bool TrtllmGenBatchedGemmRunner::isValidConfigIndex(int32_t configIndex, int32_t m, int32_t n,
                                                    int32_t k,
                                                    std::vector<int32_t> const& batchedTokens,
                                                    int32_t numTokens, int32_t numBatches,
                                                    int32_t maxNumCtasInBatchDim) const {
  auto const bmm = BatchedGemmInterface();
  auto const configs = bmm.getBatchedGemmConfigs();

  BatchedGemmData gemmData{};
  // Dims
  gemmData.mProblemDimensions.mNumBatches = numBatches;
  gemmData.mProblemDimensions.mNumTokens = numTokens;
  gemmData.mProblemDimensions.mBatchM = !mOptions.transposeMmaOutput;
  gemmData.mProblemDimensions.mBatchedM =
      mOptions.transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
  gemmData.mProblemDimensions.mBatchedN =
      mOptions.transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
  gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
  gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
  gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
  gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
  gemmData.mProblemDimensions.mRank = 0;
  gemmData.mProblemDimensions.mWorldSize = 1;
  gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;
  gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
  gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
  gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;

  auto const& config = configs[configIndex];

  return bmm.isValidConfig(config, gemmData);
}

}  // namespace kernels
}  // namespace tensorrt_llm
