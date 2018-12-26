/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

#include <x86intrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <assert.h>
#include "bf16.h"

namespace horovod {
namespace common {

inline unsigned short* bf16_alloc(size_t size){
  //size must be an integral multiple of 64.
  assert(size % 64 == 0 && "size must be an integral multiple of 64 that needed by immintrin for bf16 convertion.");
  return reinterpret_cast<unsigned short*>(aligned_alloc(64, size));
}

inline void convert_f32_to_b16(__m512i src, __m256i* dest)
{
  __m512i y = _mm512_bsrli_epi128(src, 2);
  *dest = _mm512_cvtepi32_epi16(y);
}

inline void convert_b16_to_f32(__m256i src, __m512i* dest)
{
  __m512i y = _mm512_cvtepu16_epi32(src);
  *dest = _mm512_bslli_epi128(y, 2);
}

inline void convert_f32_to_b16(__m256i src0, __m256i src1, __m256i *dst)
{
    src0 = _mm256_srli_epi32(src0, 16);
    src1 = _mm256_srli_epi32(src1, 16);
    *dst = _mm256_packus_epi32(src0, src1);
}

inline void convert_b16_to_f32(__m256i src, __m256i *dst0, __m256i *dst1)
{
    int zero[8] = {0,0,0,0,0,0,0,0};
    __m256i zeros = *(__m256i*)zero;
    *dst0 = _mm256_unpacklo_epi16(zeros, src);
    *dst1 = _mm256_unpackhi_epi16(zeros, src);
}

inline void BF16ToFloat(const unsigned short* src, float* dest, int len, int type_flag){
 switch (type_flag)
 {
   case 0:
     convert_b16_to_f32(*(__m256i*)(src), (__m512i*)(dest));
     break;
   case 1:
     convert_b16_to_f32(*(__m256i*)(src), (__m256i*)(dest), (__m256i*)(dest+8));
     break;
   default:
     unsigned int* dest_unsigned = reinterpret_cast<unsigned int*>(dest);
     for(int i=0; i<16; i++){
       *(dest_unsigned+i) = *(src+i)<<16;
     }
     break;
 }
}

inline void FloatToBF16(const float* src, unsigned short* dest, int len, int type_flag){
 switch (type_flag)
 {
   case 0:
     convert_f32_to_b16(*(__m512i*)(src), (__m256i*)(dest));
     break;
   case 1:
     convert_f32_to_b16(*(__m256i*)(src), *(__m256i*)(src+8), (__m256i*)(dest));
     break;
  default:
     unsigned int* src_unsigned = reinterpret_cast<unsigned int*>(src);
     for(int i=0; i<16; i++){
       *(dest+i) = *(src_unsigned+i)>>16;
     }
     break;
 }
}

bool check_equal(unsigned int a, unsigned short b){
  unsigned short short_a = a>>16;
  return short_a == b;
}

bool check_equal(unsigned int a, unsigned int b){
  unsigned short short_a = a>>16;
  unsigned short short_b = b>>16;
  return short_a == short_b;
}


void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype){
  int type_flag = 0;
  int i=0;
  if(type_flag == 0){
    for(; i < (*len / 16) * 16; i += 16)
    {
      // convert in & inout to m512
      __m512i in_m512 = _mm512_bslli_epi128(_mm512_cvtepu16_epi32(invec+i), 2);
      __m512i out_m512 = _mm512_bslli_epi128(_mm512_cvtepu16_epi32(inoutvec+i), 2);
      // add them together to new_inout_m256
      _m512 newout_m512 = _mm512_add_ps(in_m512, out_m512);
      // convert back and store in inout
      (__m128i*)(inoutvec + i) = _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(newout_m512, 2));
    }
  } else if(type_flag == 1){
    for(; i< (*len / 16) * 16; i += 16){
      // convert in & out to m256
      __m256i invec0 = _mm256_unpacklo_epi16(0, invec+i);
      __m256i invec1 = _mm256_unpackhi_epi16(0, invec+i);
      __m256i outvec0 = _mm256_unpacklo_epi16(0, outvec+i);
      __m256i outvec1 = _mm256_unpackhi_epi16(0, outvec+i);
      // add them together to new_inout_m256
      __m256 new_inout0_m256 = _mm256_add_ps(invec0, outvec0);
      __m256 new_inout1_m256 = _mm256_add_ps(invec1, outvec1);
      // convert back and store in inout
      new_inout0_m256 = _mm256_srli_epi32(new_inout0_m256, 16);
      new_inout1_m256 = _mm256_srli_epi32(new_inout1_m256, 16);
      (__m256i*)(inoutvec + i) = _mm256_packus_epi32(new_inout0_m256, new_inout1_m256);
    }
  }
  // process the remaining data
  for(; i < *len; i++){
    float in_float;
    float inout_float;
    in_float = reinterpret_cast<float>((*reinterpret_cast<unsigned short*>(invec + i))<<16);
    inout_float = reinterpret_cast<float>((*reinterpret_cast<unsigned short*>(outvec + i))<<16);
    inout_float += in_float;
    *(outvec + i) = (reinterpret_cast<unsigned int>(inout_float))>>16;
  }
}

} // namespace common
} // namespace horovod