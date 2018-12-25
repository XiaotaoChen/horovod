/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

#include <x86intrin.h>
#include "bf16.h"

namespace horovod {
namespace common {

inline void cvt_f32_b16(__m512i src, __m256i* dest)
{
  y = _mm512_bsrli_epi128(src, 2);
  *dest = _mm512_cvtepi32_epi16(y);
}

inline void cvt_b16_f32(__m256i src, __m512i* dest)
{
  __m512i y = _mm512_cvtepu16_epi32(src);
  *dest = _mm512_bslli_epi128(y, 2);
}

inline void BF16ToFloat(unsigned short* src, float* dest, int len){
  for(int i=0; i < len; i+=16){
    cvt_b16_f32(*(__m256i*)(src+i), (__m512i*)(dest+i));
  }
}

inline void FloatToBF16(float* src, unsigned short* dest, int len){
  for(int i=0; i< len; i+=16){
    cvt_f32_b16(*(__m512i*)(src+i), (__m256i*)(dest+i));
  }
}

void BF16_sum(void* invec, void* inoutvec, int* len){

}

} // namespace common
} // namespace horovod