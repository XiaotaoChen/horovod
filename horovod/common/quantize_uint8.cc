#include <stdint.h> //uint8_t
#include <limits> //numeric_limits
#include <stdlib.h>  // aligned_alloc
#include "quantize_uint8.h"
#include <chrono>
#include <thread>

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
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  uint8_t* invec_ui_ptr = reinterpret_cast<uint8_t*>(invec);
  uint8_t* inoutvec_ui_ptr = reinterpret_cast<uint8_t*>(inoutvec);
  float* invec_fp = reinterpret_cast<float*>(alloc_mem(*len * sizeof(float), 4));
  float* inoutvec_fp = reinterpret_cast<float*>(alloc_mem(*len * sizeof(float), 4));

  // dequantize
  dequantize(invec_ui_ptr, invec_fp, *len);
  dequantize(inoutvec_ui_ptr, inoutvec_fp, *len);

  // do summation
  fp32_sum(invec_fp, inoutvec_fp, len);

  // quantize
  quantize(inoutvec_fp, inoutvec_ui_ptr, *len);

  free(invec_fp);
  free(inoutvec_fp);
}

} // namespace common
} // namespace horovod