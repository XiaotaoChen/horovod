/*!
 *  Copyright (c) 2015 by Contributors
 * \file bf16.h
 * \brief definition of bf16 type.
 *
 * \author Xiaotao Chen
 */

//#include <immintrin.h>
#include "bf16.h"

namespace horovod {
namespace common {


bool check_equal(unsigned int a, unsigned short b){
  unsigned short short_a = a>>16;
  return short_a == b;
}

bool check_equal(unsigned int a, unsigned int b){
  unsigned short short_a = a>>16;
  unsigned short short_b = b>>16;
  return short_a == short_b;
}

//void BF16ToFloat(const unsigned short* src, float* dest, int len, int type_flag){
// switch (type_flag)
// {
//   case 0:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_b16_to_f32(*(__m256i*)(src+i), (__m512i*)(dest+i));
//       }
//       // process the remaining data
//       unsigned int* dest_unsigned = reinterpret_cast<unsigned int*>(dest);
//       for(; i < len; i++){
//         *(dest_unsigned+i) = *(src+i)<<16;
//       }
//     }
//     break;
//   case 1:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_b16_to_f32(*(__m256i*)(src+i), (__m256i*)(dest+i), (__m256i*)(dest+i+8));
//       }
//       // process the remaining data
//       unsigned int* dest_unsigned = reinterpret_cast<unsigned int*>(dest);
//       for(; i < len; i++){
//         *(dest_unsigned+i) = *(src+i)<<16;
//       }
//     }
//     break;
//   default:
//     unsigned int* dest_unsigned = reinterpret_cast<unsigned int*>(dest);
//     for(int i=0; i < len; i++){
//       *(dest_unsigned+i) = *(src+i)<<16;
//     }
//     break;
// }
//}
//
//void FloatToBF16(const float* src, unsigned short* dest, int len, int type_flag){
// switch (type_flag)
// {
//   case 0:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_f32_to_b16(*(__m512i*)(src+i), (__m256i*)(dest+i));
//       }
//       // process the remaining data
//       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
//       for(; i < len; i++){
//         *(dest+i) = *(src_unsigned+i)>>16;
//       }
//     }
//     break;
//   case 1:
//     {
//       int i = 0;
//       for(; i < (len / 16) * 16; i += 16){
//         convert_f32_to_b16(*(__m256i*)(src+i), *(__m256i*)(src+i+8), (__m256i*)(dest+i));
//       }
//       // process the remaining data
//       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
//       for(; i < len; i++){
//         *(dest+i) = *(src_unsigned+i)>>16;
//       }
//     }
//     break;
//   default:
//     {
//       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
//       for(int i=0; i < len; i++){
//         *(dest+i) = *(src_unsigned+i)>>16;
//       }
//     }
//     break;
// }
//}
//
//void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype){
//  int type_flag = 2;
//  int i=0;
//  if(type_flag == 0){
//    for(; i < (*len / 16) * 16; i += 16)
//    {
//      // convert in & inout to m512
//      __m512i in_m512 = _mm512_bslli_epi128(_mm512_cvtepu16_epi32(*(__m256i*)(invec+i)), 2);
//      __m512i out_m512 = _mm512_bslli_epi128(_mm512_cvtepu16_epi32(*(__m256i*)(inoutvec+i)), 2);
//      // add them together to new_inout_m256
//      __m512 newout_m512 = _mm512_add_ps((__m512)in_m512, (__m512)out_m512);
//      // convert back and store in inout
//      *(__m256i*)(inoutvec + i) = _mm512_cvtepi32_epi16(_mm512_bsrli_epi128((__m512i)newout_m512, 2));
//    }
//  } else if(type_flag == 1){
//    int zero[8] = {0,0,0,0,0,0,0,0};
//    __m256i zeros = *(__m256i*)zero;
//    for(; i< (*len / 16) * 16; i += 16){
//      // convert in & out to m256
//      __m256i invec0 = _mm256_unpacklo_epi16(zeros, *(__m256i*)(invec+i));
//      __m256i invec1 = _mm256_unpackhi_epi16(zeros, *(__m256i*)(invec+i));
//      __m256i outvec0 = _mm256_unpacklo_epi16(zeros, *(__m256i*)(inoutvec+i));
//      __m256i outvec1 = _mm256_unpackhi_epi16(zeros, *(__m256i*)(inoutvec+i));
//      // add them together to new_inout_m256
//      __m256 new_inout0_m256 = _mm256_add_ps((__m256)invec0, (__m256)outvec0);
//      __m256 new_inout1_m256 = _mm256_add_ps((__m256)invec1, (__m256)outvec1);
//      // convert back and store in inout
//      __m256i inout0_m256i = _mm256_srli_epi32((__m256i)new_inout0_m256, 16);
//      __m256i inout1_m256i = _mm256_srli_epi32((__m256i)new_inout1_m256, 16);
//      *(__m256i*)(inoutvec + i) = _mm256_packus_epi32(inout0_m256i, inout1_m256i);
//    }
//  }
//  // process the remaining data
//  for(; i < *len; i++){
//    float in_float;
//    float inout_float;
//    unsigned int tmp_in = (*reinterpret_cast<unsigned short*>(invec + i)) << 16;
//    unsigned int tmp_out = (*reinterpret_cast<unsigned short*>(inoutvec + i)) << 16;
//    in_float = *reinterpret_cast<float*>(&tmp_in);
//    inout_float = *reinterpret_cast<float*>(&tmp_out);
//    inout_float += in_float;
//    *(unsigned short*)(inoutvec + i) = *reinterpret_cast<unsigned int*>(&inout_float)>>16;
//  }
//}

void BF16ToFloat(const unsigned short* src, float* dest, int len, int type_flag){
  unsigned int* dest_unsigned = reinterpret_cast<unsigned int*>(dest);
  for(int i=0; i < len; i++){
    *(dest_unsigned+i) = *(src+i)<<16;
  }
}

void FloatToBF16(const float* src, unsigned short* dest, int len, int type_flag){
  const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
  for(int i=0; i < len; i++){
    *(dest+i) = *(src_unsigned+i)>>16;
  }
}

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype){
  int i=0;
  // process the remaining data
  for(; i < *len; i++){
    float in_float;
    float inout_float;
    unsigned int tmp_in = (*reinterpret_cast<unsigned short*>(invec + i)) << 16;
    unsigned int tmp_out = (*reinterpret_cast<unsigned short*>(inoutvec + i)) << 16;
    in_float = *reinterpret_cast<float*>(&tmp_in);
    inout_float = *reinterpret_cast<float*>(&tmp_out);
    inout_float += in_float;
    *(unsigned short*)(inoutvec + i) = *reinterpret_cast<unsigned int*>(&inout_float)>>16;
  }
}

} // namespace common
} // namespace horovod