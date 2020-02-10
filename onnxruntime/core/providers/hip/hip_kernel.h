// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
//#include <hcc/hc_defines.h>

#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/framework/data_transfer_manager.h"

#include "core/providers/hip/hip_execution_provider.h"

namespace onnxruntime {
namespace hip {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

// Type mapping for MLFloat16 to half
template <typename T>
class ToHipType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) {
    return static_cast<T>(f);
  }
};

// template <>
// class ToHipType<MLFloat16> {
//  public:
//   typedef hc::half MappedType;
//   static MappedType FromFloat(float f) {
//     uint16_t h = math::floatToHalf(f);
//     return *reinterpret_cast<MappedType*>(&h);
//   }
// };

// -----------------------------------------------------------------------
// Base class for HIP kernels
// -----------------------------------------------------------------------
class HipKernel : public OpKernel {
 public:
  explicit HipKernel(const OpKernelInfo& info)
      : OpKernel(info),
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<HIPExecutionProvider*>(dynamic_cast<const HIPExecutionProvider*>(info.GetExecutionProvider()))) {
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);
    // use this to precisely locate the node where HIP failure comes from
    //  if (hipSuccess != hipDeviceSynchronize())
    //    __debugbreak();

    if (s.IsOK()) {
      auto err = hipGetLastError();
      if (err != hipSuccess) {
        s = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIP error ", hipGetErrorName(err), ":", hipGetErrorString(err));
      }
    }

    return s;
  }

  virtual Status ComputeInternal(OpKernelContext* p_op_kernel_context) const = 0;

  // template <typename T>
  // inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
  //   AllocatorPtr allocator = provider_->GetAllocator(CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
  //   if (!allocator)
  //     return nullptr;
  //   return IAllocator::MakeUniquePtr<T>(allocator, count_or_bytes);
  // }

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    return provider_->GetScratchBuffer<T>(count_or_bytes);
  }

  // inline void AddDeferredReleaseCPUPtr(void* p) const {
  //   provider_->AddDeferredReleaseCPUPtr(p);
  // }

 protected:
  inline hipblasHandle_t HipblasHandle() const {
    return provider_->PerThreadHipblasHandle();
  }

  // inline hipdnnHandle_t CudnnHandle() const {
  //   return provider_->PerThreadCudnnHandle();
  // }
  // inline hiprandGenerator_t CurandGenerator() const {
  //   return provider_->PerThreadCurandGenerator();
  // }

  // template <typename T>
  // inline const T* GetConstOnes(size_t count) const {
  //   return provider_->template GetConstOnes<T>(count);
  // }

  inline Status CopyTensor(const Tensor& src, Tensor& dst) const {
    return Info().GetDataTransferManager().CopyTensor(src, dst);
  }

  inline int GetDeviceId() const { return provider_->GetDeviceId(); }

 private:
  HIPExecutionProvider* provider_;
};

}  // namespace hip
}  // namespace onnxruntime
