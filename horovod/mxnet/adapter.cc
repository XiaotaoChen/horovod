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

#if HAVE_CUDA
#include "cuda.h"
#endif

#include <mxnet/base.h>
#include <cstring> // memcpy

#include "adapter.h"
#include "cuda_util.h"
#include "tensor_util.h"
#include "../common/bf16.h"
#include "../common/half.h"

namespace horovod {
namespace mxnet {

// This class intentionally does not have destructor at the moment.
//
// Unfortunately, by the time this destructor would be called in normal
// circumstances (application shutdown), CUDA context would already be destroyed
// and cudaFree() operations would print nasty errors in the log - in a pretty
// normal termination scenario.
//
// If we add functionality to terminate Horovod without terminating the
// application, we should revisit this logic.
MXPersistentBuffer::MXPersistentBuffer(int device, int64_t size)
    : device_(device) {
  with_device device_context(device_);
  if (device_ == CPU_DEVICE_ID) {
    buffer_ = new char[size];
  } else {
#if HAVE_CUDA
    CUDA_CALL(cudaMalloc((void**)&buffer_, size));
#else
    throw std::logic_error("Internal error. Requested MXPersistentBuffer "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

const void*
MXPersistentBuffer::AccessData(std::shared_ptr<OpContext> context) const {
  return buffer_;
}

template <> MXTensor<NDArray>::MXTensor(NDArray* tensor) : tensor_(tensor) {
  int len = tensor->shape().Size();
  size_t fp16_size = len * sizeof(uint16_t);
//  // create fp16 tensor from tensor
//  uint16_t* fp16dptr_ = reinterpret_cast<uint16_t*>(bf16_alloc(fp16_size));
//  uint16_t* fp16dptr_mask_ = reinterpret_cast<uint16_t*>(bf16_alloc(fp16_size));
//
//  float* src = reinterpret_cast<float*>(tensor->data().dptr<float>());
//  // convert fp32 to fp16
//  FP32ToFP16(src, fp16dptr_, len, 0);
//  // mask fp16 to uint8
//  memcpy(fp16dptr_mask_, fp16dptr_, fp16_size);
//  mask_fp16(fp16dptr_mask_, len, 8);
//  bool equal_flag = true;
//  for(int i = 0; i < len; i++) {
//    equal_flag = check_equal(fp16dptr_[i], fp16dptr_mask_[i]);
//    if(!equal_flag) {
//      printf("i: %d, fp16, fp16_mask: %x, %x\n", i, fp16dptr_[i], fp16dptr_mask_[i]);
//      break;
//    }
//  }
//  if(equal_flag) {
//    printf("len: %d mask is equal\n", len);
//  }
//  // convert fp16 to fp32
//  FP16ToFP32(fp16dptr_mask_, src, len, 0);
//  // free fp16dptr
//  free(fp16dptr_);
//  free(fp16dptr_mask_);



  // create fp16 tensor from tensor
  uint16_t* fp16dptr_ = reinterpret_cast<uint16_t*>(bf16_alloc(fp16_size));

  float* src = reinterpret_cast<float*>(tensor->data().dptr<float>());
  // convert fp32 to fp16
  FP32ToFP16(src, fp16dptr_, len, 0);
  // mask fp16 to uint8
  mask_fp16(fp16dptr_, len, 8);
  // convert fp16 to fp32
  FP16ToFP32(fp16dptr_, src, len, 0);
  // free fp16dptr
  free(fp16dptr_);
}

template <class T> const MPIDataType MXTensor<T>::dtype() const {
  return TensorUtil::GetDType(tensor_);
}

template <class T> const TensorShape MXTensor<T>::shape() const {
  auto shape = TensorUtil::GetShape(tensor_);
  if (shape.dims() == 0) {
    // Tensor with empty shape is a Tensor with no values in MXNet, unlike a
    // constant in TensorFlow. So, we inject a dummy zero dimension to make sure
    // that the number-of-elements calculation is correct.
    shape.AddDim(0);
  }
  return shape;
}

template <class T> const void* MXTensor<T>::data() const {
  return TensorUtil::GetData(tensor_);
}

template <class T> int64_t MXTensor<T>::size() const {
  return TensorUtil::GetSize(tensor_);
}

template <class T>
MXOpContext<T>::MXOpContext(int device, T* output)
    : device_(device), output_(output) {}

template <class T>
Status MXOpContext<T>::AllocatePersistent(
    int64_t size, std::shared_ptr<PersistentBuffer>* tensor) {
  // Allocation errors are handled using PyMX exceptions.
  *tensor = std::make_shared<MXPersistentBuffer>(device_, size);
  return Status::OK();
}

template <class T>
Status MXOpContext<T>::AllocateOutput(TensorShape shape,
                                      std::shared_ptr<Tensor>* tensor) {
  int64_t* shape_array = new int64_t[shape.dims()];
  for (int idx = 0; idx < shape.dims(); idx++) {
    shape_array[idx] = shape.dim_size(idx);
  }
  TensorUtil::ResizeNd(output_, shape.dims(), shape_array);
  delete[] shape_array;
  *tensor = std::make_shared<MXTensor<T>>(output_);
  return Status::OK();
}

template <class T> Framework MXOpContext<T>::framework() const {
  return Framework::MXNET;
}

void ThrowIfError(Status status) {
  switch (status.type()) {
  case StatusType::OK:
    return;
  case StatusType::PRECONDITION_ERROR:
    throw std::logic_error(status.reason());
  case StatusType::ABORTED:
    throw std::runtime_error(status.reason());
  default: // Includes UNKNOWN_ERROR
    throw std::runtime_error(status.reason());
  }
}

template <> MXBF16Tensor<NDArray>::MXBF16Tensor(NDArray* tensor) : MXTensor<NDArray>(tensor) {
  int len = tensor->shape().Size();
  size_t bf16_size = len * sizeof(uint16_t);
  // create bf16 tensor from tensor
  this->bf16dptr_ = bf16_alloc(bf16_size);
//  FloatToBF16(reinterpret_cast<const float*>(TensorUtil::GetData(tensor)), this->bf16dptr_, len, 2);
  FloatToBF16(reinterpret_cast<const float*>(tensor->data().dptr<float>()), this->bf16dptr_, len, 0);
  // to check equal
  const unsigned int* src_p = reinterpret_cast<const unsigned int*>
                              (tensor->data().dptr<float>());
  const unsigned short* dst_p = reinterpret_cast<const uint16_t*>(this->bf16dptr_);
  for(int i=0; i < len; i++){
    if(!check_equal(src_p[i], dst_p[i])){
      printf("float to bf16, check equal error: %d, %x, %x\n",
              i, src_p[i], dst_p[i]);
    }
  }
}

template <> const MPIDataType MXBF16Tensor<NDArray>::dtype() const {
  return MPIDataType::HOROVOD_BF16;
}

template <> const void* MXBF16Tensor<NDArray>::data() const {
  return reinterpret_cast<void*>(this->bf16dptr_);
}

template <> int64_t MXBF16Tensor<NDArray>::size() const {
  return (int64_t)(this->tensor_->shape().Size()) * sizeof(uint16_t);
}

template<> void* MXBF16Tensor<NDArray>::source_data() {
  return const_cast<void*>(MXTensor<NDArray>::data());
}

template <> MXBF16Tensor<NDArray>::~MXBF16Tensor(){
  free(this->bf16dptr_);
}

template <> MXFP16Tensor<NDArray>::MXFP16Tensor(NDArray* tensor) : MXTensor<NDArray>(tensor) {
  int len = tensor->shape().Size();
  size_t fp16_size = len * sizeof(unsigned short);
  // create fp16 tensor from tensor
  this->fp16dptr_ = bf16_alloc(fp16_size);

  float* src = reinterpret_cast<float*>(tensor->data().dptr<float>());
  // convert fp32 to fp16
  FP32ToFP16(src, this->fp16dptr_, len, 0);
}

template <> const MPIDataType MXFP16Tensor<NDArray>::dtype() const {
  return MPIDataType::HOROVOD_FLOAT16;
}

template <> const void* MXFP16Tensor<NDArray>::data() const {
  return reinterpret_cast<const void*>(this->fp16dptr_);
}

template <> int64_t MXFP16Tensor<NDArray>::size() const {
  return (int64_t)(this->tensor_->shape().Size()) * sizeof(uint16_t);
}

template<> void* MXFP16Tensor<NDArray>::source_data() {
  return const_cast<void*>(MXTensor<NDArray>::data());
}

template <> MXFP16Tensor<NDArray>::~MXFP16Tensor(){
  free(this->fp16dptr_);
}
template class MXTensor<NDArray>;
template class MXOpContext<NDArray>;
template class MXBF16Tensor<NDArray>;
template class MXFP16Tensor<NDArray>;

} // namespace mxnet
} // namespace horovod
