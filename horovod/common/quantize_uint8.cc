#include <stdint.h> //uint8_t
#include <limits> //numeric_limits
#include <stdlib.h>  // aligned_alloc
#include <cmath>  // fabs
#include <cstdio>  // printf
#include "quantize_uint8.h"

namespace horovod {
namespace common {

void* alloc_mem(size_t size, int alignment) {
  if (size % alignment == 0) {
    return aligned_alloc(alignment, size);
  }
  else {
    size_t new_size = (size / alignment + 1 ) * alignment;
    return aligned_alloc(alignment, new_size);
  }
}

void quantize(const float* src, uint8_t* dst, int len) {
  // dst size must be: len*sizeof(uint8_t) + 2*sizeof(float);
  float maximum = std::numeric_limits<float>::min();
  float minimum = std::numeric_limits<float>::max();
  for (int i = 0; i < len; i++) {
    maximum = src[i] > maximum ? src[i] : maximum;
    minimum = src[i] < minimum ? src[i] : minimum;
  }
  float max_dist = maximum - minimum;
  for (int i = 0; i < len; i++) {
    dst[i + 8] = (uint8_t)(255 * (src[i] - minimum) / max_dist);
  }
  float* fp_ptr = reinterpret_cast<float*>(dst);
  *fp_ptr = maximum;
  *(fp_ptr + 1) = minimum;

//  printf("quantize max, min: %f, %f\n", maximum, minimum);
}

void dequantize(const uint8_t* src, float* dst, int len) {
  const float* fp_ptr = reinterpret_cast<const float*>(src);
  float maximum = *fp_ptr;
  float minimum = *(fp_ptr + 1);
  float max_dist = maximum - minimum;
  for(int i = 0; i < len; i++) {
    dst[i] = max_dist * src[i + 8] / 255 + minimum;
  }
}

void quantize_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
  int real_len = *len - 2 * sizeof(float);
  uint8_t* invec_ui_ptr = reinterpret_cast<uint8_t*>(invec);
  uint8_t* inoutvec_ui_ptr = reinterpret_cast<uint8_t*>(inoutvec);
  float* invec_fp = reinterpret_cast<float*>(alloc_mem(real_len * sizeof(float), 4));
  float* inoutvec_fp = reinterpret_cast<float*>(alloc_mem(real_len * sizeof(float), 4));

  // dequantize
  dequantize(invec_ui_ptr, invec_fp, real_len);
  dequantize(inoutvec_ui_ptr, inoutvec_fp, real_len);

  // do summation
  fp32_sum(invec_fp, inoutvec_fp, &real_len);

  // quantize
  quantize(inoutvec_fp, inoutvec_ui_ptr, real_len);

  free(invec_fp);
  free(inoutvec_fp);
}

void variance_range(const float* src, const float* dequantized_src, int len) {
  float max_range = std::numeric_limits<float>::min();
  float min_range = std::numeric_limits<float>::max();
  int count=0;
  for(int i = 0; i < len; i++) {
    float abs_range = src[i] - dequantized_src[i];
    max_range = max_range > abs_range ? max_range : abs_range;
    min_range = min_range < abs_range ? min_range : abs_range;
    if(fabs(abs_range/src[i]) > 0.1) {
      count += 1;
//      printf("src, deq_src: %f, %f, abs range: %f\n", src[i], dequantized_src[i], abs_range);
    }
  }
  printf("len: %d, exceed 0.1: %d, percent: %f, max, min range: %f, %f\n", len, count, 1.0 * count / len, max_range, min_range);
}

} // namespace common
} // namespace horovod