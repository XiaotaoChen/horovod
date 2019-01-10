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
  // TODO set alignment to 256/512, it's performance may be better.
  //size must be an integral multiple of 64.
  if (size % 64 == 0) {
    return reinterpret_cast<unsigned short*>(aligned_alloc(64, size));
  }
  else {
    size_t new_size = (size / 64 + 1 ) * 64;
    return reinterpret_cast<unsigned short*>(aligned_alloc(64, new_size));
  }
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

bool check_equal(const unsigned int a, const unsigned short b);

bool check_masked(float* src, int size);

float cal_var_range(const unsigned int a, const unsigned short b);

void cal_min_max_var(const unsigned int* fp32_p,
                     const unsigned short* bf16_p,
                     int len,
                     float* min_var,
                     float* max_var);

void BF16ToFloat(const unsigned short* src, float* dst, int len, int type_flag);

void FloatToBF16(const float* src, unsigned short* dst, int len, int type_flag);

void mask_fp32(float* fp, int size);

void mask_fp32(float* src, int size, int low_bits);

void BFloat16ToFloat(const unsigned short* src, float* dst, int size, int type_flag);

void FloatToBFloat16(const float* src, unsigned short* dst, int size, int type_flag);

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

} // namespace common
} // namespace horovod

#endif // HOROVOD_BF16_H