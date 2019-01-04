/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications copyright (C) 2018 Uber Technologies, Inc.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#ifndef HOROVOD_HALF_H
#define HOROVOD_HALF_H

#include <stdint.h>  // uint16_t
#include <immintrin.h>

#define OMPI_SKIP_MPICXX
#include "mpi.h"

namespace horovod {
namespace common {

inline void convert_f32_to_f16(__m512 src, __m256i* dst) {
  // round to nearest, and suppress exceptions
  *dst = _mm512_cvtps_ph(src, _MM_FROUND_TO_NEAREST_INT);
}

inline void convert_f32_to_f16(__m256 src, __m128i* dst) {
  // round to nearest, and suppress exceptions
  *dst = _mm256_cvtps_ph(src, _MM_FROUND_TO_NEAREST_INT);
}

inline void convert_f32_to_f16(float src, unsigned short* dst) {
  *dst = _cvtss_sh(src, 0);
}

inline void convert_f16_to_f32(__m256i src, __m512* dst) {
  *dst = _mm512_cvtph_ps(src);
}

inline void convert_f16_to_f32(__m128i src, __m256* dst) {
  *dst = _mm256_cvtph_ps(src);
}

inline void convert_f16_to_f32(unsigned short src, float* dst) {
  *dst = _cvtsh_ss(src);
}
void FP32ToFP16(const float* src, uint16_t* dst, int len, int type_flag);

void FP16ToFP32(const uint16_t* src, float* dst, int len, int type_flag);

inline void HalfBits2Float(unsigned short* src, float* res) {
  unsigned h = *src;
  int sign = ((h >> 15) & 1);
  int exp = ((h >> 10) & 0x1f);
  int mantissa = (h & 0x3ff);
  unsigned f = 0;

  if (exp > 0 && exp < 31) {
    // normal
    exp += 112;
    f = (sign << 31) | (exp << 23) | (mantissa << 13);
  } else if (exp == 0) {
    if (mantissa) {
      // subnormal
      exp += 113;
      while ((mantissa & (1 << 10)) == 0) {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    } else {
      // sign-preserving zero
      f = (sign << 31);
    }
  } else if (exp == 31) {
    if (mantissa) {
      f = 0x7fffffff;  // not a number
    } else {
      f = (0xff << 23) | (sign << 31);  //  inf
    }
  }

  *res = *reinterpret_cast<float const*>(&f);
}

inline void Float2HalfBits(float* src, unsigned short* dest) {
  // software implementation rounds toward nearest even
  unsigned const& s = *reinterpret_cast<unsigned const*>(src);
  uint16_t sign = uint16_t((s >> 16) & 0x8000);
  int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
  int mantissa = s & 0x7fffff;
  uint16_t u = 0;

  if ((s & 0x7fffffff) == 0) {
    // sign-preserving zero
    *dest = sign;
    return;
  }

  if (exp > 15) {
    if (exp == 128 && mantissa) {
      // not a number
      u = 0x7fff;
    } else {
      // overflow to infinity
      u = sign | 0x7c00;
    }
    *dest = u;
    return;
  }

  int sticky_bit = 0;

  if (exp >= -14) {
    // normal fp32 to normal fp16
    exp = uint16_t(exp + uint16_t(15));
    u = uint16_t(((exp & 0x1f) << 10));
    u = uint16_t(u | (mantissa >> 13));
  } else {
    // normal single-precision to subnormal half_t-precision representation
    int rshift = (-14 - exp);
    if (rshift < 32) {
      mantissa |= (1 << 23);

      sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

      mantissa = (mantissa >> rshift);
      u = (uint16_t(mantissa >> 13) & 0x3ff);
    } else {
      mantissa = 0;
      u = 0;
    }
  }

  // round to nearest even
  int round_bit = ((mantissa >> 12) & 1);
  sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

  if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
    u = uint16_t(u + 1);
  }

  u |= sign;

  *dest = u;
}

void float16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

} // namespace common
} // namespace horovod

#endif // HOROVOD_HALF_H
