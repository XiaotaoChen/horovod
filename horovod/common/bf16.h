/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

#ifndef HOROVOD_BF16_H
#define HOROVOD_BF16_H

namespace horovod {
namespace common {

inline void BF16ToFloat(unsigned short* src, float* dest, int len);

inline void FloatToBF16(float* src, unsigned short* dest, int len);

void BF16_sum(void* invec, void* inoutvec, int* len);

} // namespace common
} // namespace horovod

#endif // HOROVOD_BF16_H