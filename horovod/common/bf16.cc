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

bool is_aligned(const void* ptr, int alignment) {
  auto iptr = reinterpret_cast<uintptr_t>(ptr);
  return !(iptr % alignment);
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
 switch (type_flag)
 {
   case 0:
     {
       int i;
       for(i = 0; i < (len / 16) * 16; i += 16){
         convert_b16_to_f32((const void*)(src+i), (void*)(dst+i));
       }
       // process the remaining data
       unsigned int* dst_unsigned = reinterpret_cast<unsigned int*>(dst);
       for(i = (len / 16) * 16; i < len; i++){
         *(dst_unsigned+i) = *(src+i)<<16;
       }
     }
     break;
   default:
     {
       unsigned int* dst_unsigned = reinterpret_cast<unsigned int*>(dst);
       int i;
       for(i=0; i < len; i++){
         *(dst_unsigned+i) = *(src+i)<<16;
       }
     }
     break;
 }
}

void FloatToBF16(const float* src, uint16_t* dst, int len, int type_flag){
 bool aligned_flag = is_aligned(reinterpret_cast<const void*>(src), 64)
                     && is_aligned(reinterpret_cast<const void*>(dst), 64);
 switch (type_flag)
 {
   case 0:
     {
       int i;
       if (aligned_flag) {
         for(i = 0; i < (len / 16) * 16; i += 16){
           convert_f32_to_b16((__m512i*)(src+i), (__m256i*)(dst+i));
         }
       } else {
         for(i = 0; i < (len / 16) * 16; i += 16){
           convert_f32_to_b16((const void*)(src+i), (void*)(dst+i));
         }
       }
       // process the remaining data
       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
       for(i = (len / 16) * 16; i < len; i++){
         *(dst+i) = *(src_unsigned+i)>>16;
       }
     }
     break;
   default:
     {
       const unsigned int* src_unsigned = reinterpret_cast<const unsigned int*>(src);
       int i;
       for(i=0; i < len; i++){
         *(dst+i) = *(src_unsigned+i)>>16;
       }
     }
     break;
 }
}

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype){
  bool aligned_flag = is_aligned(reinterpret_cast<const void*>(invec), 64)
                      && is_aligned(reinterpret_cast<const void*>(inoutvec), 64);
  int i;
  uint16_t* invec_16 = reinterpret_cast<uint16_t*>(invec);
  uint16_t* inoutvec_16 = reinterpret_cast<uint16_t*>(inoutvec);
  int type_flag = 0;
  if(type_flag == 0){
    if (aligned_flag) {
      for(i=0; i < (*len / 16) * 16; i += 16)
      {
        // convert in & inout to m512
        __m512i in_m512, out_m512;
        convert_b16_to_f32((const void*)(invec_16+i), (void*)(&in_m512));
        convert_b16_to_f32((const void*)(inoutvec_16+i), (void*)(&out_m512));
        // add them together to new_inout_m256
        __m512 newout_m512 = _mm512_add_ps((__m512)in_m512, (__m512)out_m512);
        // convert back and store in inout
        convert_f32_to_b16((__m512i*)(&newout_m512), (__m256i*)(inoutvec_16+i));
      }
    } else {
      for(i=0; i < (*len / 16) * 16; i += 16)
      {
        // convert in & inout to m512
        __m512i in_m512, out_m512;
        convert_b16_to_f32((const void*)(invec_16+i), (void*)(&in_m512));
        convert_b16_to_f32((const void*)(inoutvec_16+i), (void*)(&out_m512));
        // add them together to new_inout_m256
        __m512 newout_m512 = _mm512_add_ps((__m512)in_m512, (__m512)out_m512);
        // convert back and store in inout
        convert_f32_to_b16((const void *)(&newout_m512), (void*)(inoutvec_16+i));
      }
    }
  }
  // process the remaining data
  for(i=(*len / 16) * 16; i < *len; i++){
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