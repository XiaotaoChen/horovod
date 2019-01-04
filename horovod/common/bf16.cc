/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

#include <cmath>
#include <stdio.h>
#include "bf16.h"

namespace horovod {
namespace common {


bool check_equal(const unsigned int a, const uint16_t b){
  uint16_t short_a = a>>16;
  return short_a == b;
}

float cal_var_range(const unsigned int a, const uint16_t b){
  const float* fp_a = reinterpret_cast<const float*>(&a);
  const unsigned int int_b = b<<16;
  const float* fp_b = reinterpret_cast<const float*>(&int_b);
  float abs_err = fabs((*fp_a) - (*fp_b));
  float var = 0;
  if (fabs(*fp_a) > 0.00001){
    var = abs_err / fabs(*fp_a);
  }
  if (var > 0.1) {
    printf("range is larger than 0.1 , (a, b) %f, %f, %x, %x, abs_err: %f, range: %f\n", *fp_a, *fp_b, a, int_b, abs_err, var);
  }
  return var;
}

void cal_min_max_var(const unsigned int* fp32_p,
                     const uint16_t* bf16_p,
                     int len,
                     float* min_var,
                     float* max_var){
  for (int i = 0; i < len; i++) {
    float temp = cal_var_range(*(fp32_p + i), *(bf16_p + i));
    if (temp < *min_var) {
      *min_var = temp;
    }
    if (temp > *max_var) {
      *max_var = temp;
    }
  }
}

void BF16ToFloat(const uint16_t* src, float* dst, int len, int type_flag){
// switch (type_flag)
// {
//   case 0:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_b16_to_f32(*(__m256i*)(src+i), (__m512i*)(dst+i));
//       }
//       // process the remaining data
//       unsigned int* dst_unsigned = reinterpret_cast<unsigned int*>(dst);
//       for(; i < len; i++){
//         *(dst_unsigned+i) = *(src+i)<<16;
//       }
//     }
//     break;
//   case 1:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_b16_to_f32(*(__m256i*)(src+i), (__m256i*)(dst+i), (__m256i*)(dst+i+8));
//       }
//       // process the remaining data
//       unsigned int* dst_unsigned = reinterpret_cast<unsigned int*>(dst);
//       for(; i < len; i++){
//         *(dst_unsigned+i) = *(src+i)<<16;
//       }
//     }
//     break;
//   default:
//     {
//       unsigned int* dst_unsigned = reinterpret_cast<unsigned int*>(dst);
//       for(int i=0; i < len; i++){
//         *(dst_unsigned+i) = *(src+i)<<16;
//       }
//     }
//     break;
// }
 unsigned int* dst_unsigned = reinterpret_cast<unsigned int*>(dst);
 for(int i=0; i < len; i++){
   *(dst_unsigned+i) = *(src+i)<<16;
 }
}

void FloatToBF16(const float* src, uint16_t* dst, int len, int type_flag){
// switch (type_flag)
// {
//   case 0:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_f32_to_b16(*(__m512i*)(src+i), (__m256i*)(dst+i));
//       }
//       // process the remaining data
//       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
//       for(; i < len; i++){
//         *(dst+i) = *(src_unsigned+i)>>16;
//       }
//     }
//     break;
//   case 1:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_f32_to_b16(*(__m256i*)(src+i), *(__m256i*)(src+i+8), (__m256i*)(dst+i));
//       }
//       // process the remaining data
//       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
//       for(; i < len; i++){
//         *(dst+i) = *(src_unsigned+i)>>16;
//       }
//     }
//     break;
//   default:
//     {
//       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
//       for(int i=0; i < len; i++){
//         *(dst+i) = *(src_unsigned+i)>>16;
//       }
//     }
//     break;
// }
 const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
 for(int i=0; i < len; i++){
   *(dst+i) = *(src_unsigned+i)>>16;
 }
}

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype){
  int type_flag = 0;
  int i=0;
  uint16_t* invec_16 = reinterpret_cast<uint16_t*>(invec);
  uint16_t* inoutvec_16 = reinterpret_cast<uint16_t*>(inoutvec);
//  if(type_flag == 0){
//    for(; i < (*len / 16) * 16; i += 16)
//    {
//      // convert in & inout to m512
//      __m512i in_m512 = _mm512_bslli_epi128(_mm512_cvtepu16_epi32(*(__m256i*)(invec_16+i)), 2);
//      __m512i out_m512 = _mm512_bslli_epi128(_mm512_cvtepu16_epi32(*(__m256i*)(inoutvec_16+i)), 2);
//      // add them together to new_inout_m256
//      __m512 newout_m512 = _mm512_add_ps((__m512)in_m512, (__m512)out_m512);
//      // convert back and store in inout
//      *(__m256i*)(inoutvec_16 + i) = _mm512_cvtepi32_epi16(_mm512_bsrli_epi128((__m512i)newout_m512, 2));
//    }
//  } else if(type_flag == 1){
//    // alignas 64 for __m256i requested
//    alignas(64) int zero[8] = {0,0,0,0,0,0,0,0};
//    __m256i zeros = *(__m256i*)zero;
//    for(; i< (*len / 16) * 16; i += 16){
//      // convert in & out to m256
//      __m256i invec0 = _mm256_unpacklo_epi16(zeros, *(__m256i*)(invec_16 + i));
//      __m256i invec1 = _mm256_unpackhi_epi16(zeros, *(__m256i*)(invec_16 + i));
//      __m256i outvec0 = _mm256_unpacklo_epi16(zeros, *(__m256i*)(inoutvec_16 + i));
//      __m256i outvec1 = _mm256_unpackhi_epi16(zeros, *(__m256i*)(inoutvec_16 + i));
//      // add them together to new_inout_m256
//      __m256 new_inout0_m256 = _mm256_add_ps((__m256)invec0, (__m256)outvec0);
//      __m256 new_inout1_m256 = _mm256_add_ps((__m256)invec1, (__m256)outvec1);
//      // convert back and store in inout
//      __m256i inout0_m256i = _mm256_srli_epi32((__m256i)new_inout0_m256, 16);
//      __m256i inout1_m256i = _mm256_srli_epi32((__m256i)new_inout1_m256, 16);
//      *(__m256i*)(inoutvec_16 + i) = _mm256_packus_epi32(inout0_m256i, inout1_m256i);
//    }
//  }
  // process the remaining data
  for(; i < *len; i++){
    unsigned int tmp_in = (*(invec_16 + i)) << 16;
    unsigned int tmp_out = (*(inoutvec_16 + i)) << 16;
    float in_float = *reinterpret_cast<float*>(&tmp_in);
    float inout_float = *reinterpret_cast<float*>(&tmp_out);
    inout_float += in_float;
    *(inoutvec_16 + i) = *reinterpret_cast<unsigned int*>(&inout_float)>>16;
  }
}

// ref to tensorflow implementation
void BFloat16ToFloat(const uint16_t* src, float* dst, int size, int type_flag) {
  const uint16_t* p = src;
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
  for (; size != 0; p++, q += 2, size--) {
    q[0] = 0;
    q[1] = *p;
  }
}

void FloatToBFloat16(const float* src, uint16_t* dst, int size, int type_flag) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = dst;
  for (; size != 0; p += 2, q++, size--) {
    *q = p[1];
  }
}

} // namespace common
} // namespace horovod