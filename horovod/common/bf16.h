/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

#ifndef HOROVOD_BF16_H
#define HOROVOD_BF16_H

#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>  // uint16_t
#define OMPI_SKIP_MPICXX
#include "mpi.h"

namespace horovod {
namespace common {

inline uint16_t* bf16_alloc(size_t size){
  // TODO set alignment to 256/512, it's performance may be better.
  //size must be an integral multiple of 64.
  if (size % 64 == 0) {
    return reinterpret_cast<uint16_t*>(aligned_alloc(64, size));
  }
  else {
    size_t new_size = (size / 64 + 1 ) * 64;
    return reinterpret_cast<uint16_t*>(aligned_alloc(64, new_size));
  }
}

inline void convert_f32_to_b16(const void* src, void* dst)
{
  __m512i y = _mm512_bsrli_epi128(_mm512_loadu_si512(src), 2);
  _mm256_storeu_si256((__m256i*)(dst), _mm512_cvtepi32_epi16(y));
}

// when the pointer is aligned, it needn't to call movdqu
inline void convert_f32_to_b16(__m512i* src, __m256i* dst)
{
  __m512i y = _mm512_bsrli_epi128(*src, 2);
  *dst = _mm512_cvtepi32_epi16(y);
}

inline void convert_b16_to_f32(const void* src, void* dst)
{
  __m512i y = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src));
  _mm512_storeu_si512(dst, _mm512_bslli_epi128(y, 2));
}

bool check_equal(const unsigned int a, const uint16_t b);

bool is_aligned(const void* ptr, int alignment);

float cal_var_range(const unsigned int a, const uint16_t b);

void cal_min_max_var(const unsigned int* fp32_p,
                     const uint16_t* bf16_p,
                     int len,
                     float* min_var,
                     float* max_var);

void BF16ToFloat(const uint16_t* src, float* dst, int len, int type_flag);

void FloatToBF16(const float* src, uint16_t* dst, int len, int type_flag);

void BFloat16ToFloat(const uint16_t* src, float* dst, int size, int type_flag);

void FloatToBFloat16(const float* src, uint16_t* dst, int size, int type_flag);

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

} // namespace common
} // namespace horovod

#endif // HOROVOD_BF16_H