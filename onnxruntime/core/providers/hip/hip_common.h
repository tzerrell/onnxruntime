// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
//#include <hiprand/hiprand.h>

#include "core/common/status.h"
#include "core/framework/data_transfer_manager.h"
#include "core/graph/graph_viewer.h"
#include "core/util/math.h"

#include "core/providers/hip/fast_divmod.h"
#include "core/providers/hip/hip_call.h"
#include "core/providers/hip/hip_execution_provider.h"
#include "core/providers/hip/hip_kernel.h"


namespace onnxruntime {
namespace hip {

#define HIP_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(HIP_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIP error executing ", #expr))

#define HIPBLAS_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(HIPBLAS_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPBLAS error executing ", #expr))
/*
#define CUSPARSE_RETURN_IF_ERROR(expr)           \
  ORT_RETURN_IF_ERROR(CUSPARSE_CALL(expr)        \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUSPARSE error executing ", #expr))

#define CURAND_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(CURAND_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CURAND error executing ", #expr))

#define CUDNN_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(CUDNN_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDNN error executing ", #expr))

#define CUDNN2_RETURN_IF_ERROR(expr, m)          \
  ORT_RETURN_IF_ERROR(CUDNN_CALL2(expr, m)       \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDNN2 error executing ", #expr))
*/



inline bool CalculateFdmStrides(gsl::span<fast_divmod> p, const std::vector<int64_t>& dims) {
  int stride = 1;
  if (dims.empty() || p.size() < dims.size())
    return false;
  auto rank = p.size();
  for (size_t i = 0; i < rank; i++) {
    p[rank - 1 - i] = fast_divmod(stride);
    if (i < dims.size() - 1) {
      stride *= static_cast<int>(dims[dims.size() - 1 - i]);
    }
  }
  return true;
}

struct DeviceProp {
  static const std::vector<hipDeviceProp_t>& GetCachedDeviceProps() {
    std::call_once(s_cachedDevicePropsInitFlag, [=] {
      int numDevices;
      // must wait GPU idle, otherwise hipGetDeviceProperties might fail
      HIP_CALL_THROW(hipDeviceSynchronize());
      HIP_CALL_THROW(hipGetDeviceCount(&numDevices));
      s_cachedDeviceProps.resize(numDevices);
      for (int i = 0; i < numDevices; i++)
        HIP_CALL_THROW(hipGetDeviceProperties(&s_cachedDeviceProps[i], i));
    });

    return s_cachedDeviceProps;
  }

  static size_t GetCurrentDeviceId() {
    int deviceId;
    hipGetDevice(&deviceId);
    return (size_t)deviceId;
  }

  // get device properties of current device
  static const hipDeviceProp_t& GetDeviceProps() {
    const auto& cachedDevicesProps = GetCachedDeviceProps();
    return cachedDevicesProps[GetCurrentDeviceId()];
  }

 private:
  static std::vector<hipDeviceProp_t> s_cachedDeviceProps;
  static std::once_flag s_cachedDevicePropsInitFlag;
};

}  // namespace hip
}  // namespace onnxruntime
