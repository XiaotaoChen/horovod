/*!
 *  Copyright (c) 2015 by Contributors
 * \file quantize_uint8.h
 * \brief definition of quantize fp32 to uint8.
 *
 * \author Xiaotao Chen
 */

#ifndef HOROVOD_QUANTIZE_UINT8_H
#define HOROVOD_QUANTIZE_UINT8_H

#define OMPI_SKIP_MPICXX
#include "mpi.h"

namespace horovod {
namespace common {

void* alloc_mem(size_t size, int alignment);

inline void free_mem(void* ptr) {
  free(ptr);
}

inline void fp32_sum(float* invec, float* inoutvec, int* len) {
  for (int i = 0; i < *len; i++) inoutvec[i] += invec[i];
}

void quantize(const float* src, uint8_t* dst, int len);

void dequantize(const uint8_t* src, float* dst, int len);

void quantize_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

} // namespace common
} // namespace horovod

#endif // HOROVOD_QUANTIZE_UINT8_H