/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

#ifndef HOROVOD_BF16_H
#define HOROVOD_BF16_H

#include "mpi.h"

namespace horovod {
namespace common {

inline unsigned short* bf16_alloc(size_t size);

inline void BF16ToFloat(const unsigned short* src, float* dest, int len, int type_flag);

inline void FloatToBF16(const float* src, unsigned short* dest, int len, int type_flag);

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

} // namespace common
} // namespace horovod

#endif // HOROVOD_BF16_H