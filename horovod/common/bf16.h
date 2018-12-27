/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

#ifndef HOROVOD_BF16_H
#define HOROVOD_BF16_H

//#include <immintrin.h>
#include <stdlib.h>
#include <assert.h>
#define OMPI_SKIP_MPICXX
#include "mpi.h"

namespace horovod {
namespace common {

inline unsigned short* bf16_alloc(size_t size){
  //size must be an integral multiple of 64.
  assert(size % 64 == 0 && "size must be an integral multiple of 64 that needed by immintrin for bf16 convertion.");
  return reinterpret_cast<unsigned short*>(aligned_alloc(64, size));
}

//inline void convert_f32_to_b16(__m512i src, __m256i* dest)
//{
//  __m512i y = _mm512_bsrli_epi128(src, 2);
//  *dest = _mm512_cvtepi32_epi16(y);
//}
//
//inline void convert_b16_to_f32(__m256i src, __m512i* dest)
//{
//  __m512i y = _mm512_cvtepu16_epi32(src);
//  *dest = _mm512_bslli_epi128(y, 2);
//}
//
//inline void convert_f32_to_b16(__m256i src0, __m256i src1, __m256i *dst)
//{
//    src0 = _mm256_srli_epi32(src0, 16);
//    src1 = _mm256_srli_epi32(src1, 16);
//    *dst = _mm256_packus_epi32(src0, src1);
//}
//
//inline void convert_b16_to_f32(__m256i src, __m256i *dst0, __m256i *dst1)
//{
//    int zero[8] = {0,0,0,0,0,0,0,0};
//    __m256i zeros = *(__m256i*)zero;
//    *dst0 = _mm256_unpacklo_epi16(zeros, src);
//    *dst1 = _mm256_unpackhi_epi16(zeros, src);
//}

void BF16ToFloat(const unsigned short* src, float* dest, int len, int type_flag);

void FloatToBF16(const float* src, unsigned short* dest, int len, int type_flag);

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

} // namespace common
} // namespace horovod

#endif // HOROVOD_BF16_H