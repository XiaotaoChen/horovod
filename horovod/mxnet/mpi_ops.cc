// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <chrono>
#include <memory>
#include <thread>
#include <stdio.h>

#include "../common/bf16.h"
#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "mpi_ops.h"
#include "tensor_util.h"

namespace horovod {
namespace mxnet {

static HandleManager handle_manager;

namespace {

std::string GetOpName(std::string prefix, char* name) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }
  return std::string();
}

std::string GetOpNameHandle(std::string prefix, const std::string& name, int handle) {
  if (name.length() != 0) {
    return name;
  }
  return prefix + ".noname." + std::to_string(handle);
}

} // namespace

int DoAllreduce(NDArray* tensor, NDArray* output, int average, const std::string& name, Callback cb, int rank, int count) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle(cb);
  auto device = TensorUtil::GetDevice(tensor);
//  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_tensor = std::make_shared<MXBF16Tensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = hvd_tensor;
  if (tensor->var() != output->var()){
    hvd_output = std::make_shared<MXBF16Tensor<NDArray>>(output);
//    hvd_output = std::make_shared<MXTensor<NDArray>>(output);
  }

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, nullptr,
      GetOpNameHandle("allreduce", name, handle), device,
      [handle, average, output, hvd_output](const Status& status) {
        // convert bf16_tensor to fp32, assign to output
        BF16ToFloat(reinterpret_cast<const unsigned short*>(hvd_output->data()),
                    reinterpret_cast<float*>(hvd_output->source_data()),
                    output->shape().Size(),
                    2);
        handle_manager.MarkDone(handle, status);
        handle_manager.ExecuteCallback(handle);
      });
  ThrowIfError(enqueue_result);
  return handle;
}

int DoAllgather(NDArray* tensor, NDArray* output, std::string& name, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);

  auto handle = handle_manager.AllocateHandle(cb);
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_tensor, nullptr, GetOpNameHandle("allgather", name, handle),
      device, [handle](const Status& status) {
        handle_manager.MarkDone(handle, status);
        handle_manager.ExecuteCallback(handle);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoBroadcast(NDArray* tensor, NDArray* output, int root_rank, std::string& name, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<MXTensor<NDArray>>(output);
  }

  auto handle = handle_manager.AllocateHandle(cb);
  auto enqueue_result =
      EnqueueTensorBroadcast(hvd_context, hvd_tensor, hvd_output, root_rank,
                             nullptr, GetOpNameHandle("broadcast", name, handle),
                             device, [handle](const Status& status) {
                               handle_manager.MarkDone(handle, status);
                               handle_manager.ExecuteCallback(handle);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

// Do AllReduce on GPU only if src and dst are on GPU
// Otherwise do AllReduce on CPU
extern "C" int horovod_mxnet_allreduce_async(
    NDArray* input, NDArray* output, int average, char* name, int rank, int count) {

  std::string new_name = GetOpName("allreduce", name);
  auto allreduce_async_fn = [input, output, new_name, average, rank, count](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
      DoAllreduce(input, output, average, new_name, cb, rank, count);
  };
    // Not in-place
    if (input->var() != output->var()) {
      Engine::Get()->PushAsync(
        allreduce_async_fn,
        input->ctx(),
        {input->var()},
        {output->var()},
        FnProperty::kNormal,
        0,
        "HorovodAllreduce");
    // In-place
    } else {
      Engine::Get()->PushAsync(
        allreduce_async_fn,
        input->ctx(),
        {},
        {output->var()},
        FnProperty::kNormal,
        0,
        "HorovodAllreduce");
    }
  return 0;
}

extern "C" int horovod_mxnet_allgather_async(
    NDArray* input, NDArray* output, char* name) {

  std::string new_name = GetOpName("allgather", name);
  auto allgather_async_fn = [input, output, new_name](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
    DoAllgather(input, output, new_name, cb);
  };
    if (input->var() != output->var()) {
      Engine::Get()->PushAsync(
        allgather_async_fn,
        Context::CPU(0),
        {input->var()},
        {output->var()},
        FnProperty::kNormal,
        0,
        "HorovodAllgather");
  // In-place
    } else {
      Engine::Get()->PushAsync(
        allgather_async_fn,
        Context::CPU(0),
        {},
        {output->var()},
        FnProperty::kNormal,
        0,
        "HorovodAllgather");
    }
  return 0;
}

extern "C" int horovod_mxnet_broadcast_async(
    NDArray* input, NDArray* output, int root_rank, char* name) {
   
  std::string new_name = GetOpName("broadcast", name);
  auto broadcast_async_fn = [input, output, new_name, root_rank](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
    DoBroadcast(input, output, root_rank, new_name, cb);
  };
    // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(
      broadcast_async_fn,
      Context::CPU(0),
      {input->var()},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodBroadcast");
  } else {
    Engine::Get()->PushAsync(
      broadcast_async_fn,
      Context::CPU(0),
      {},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodBroadcast");
  }
  return 0;
}

extern "C" int horovod_mxnet_poll(int handle) {
  return handle_manager.PollHandle(handle) ? 1 : 0;
}

extern "C" void horovod_mxnet_wait_and_clear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

} // namespace mxnet
} // namespace horovod
