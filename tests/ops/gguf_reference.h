#pragma once

// An independent CPU decode of the ggml K-quant block types a GGUF release stores, and the FP64
// weight values its tests' oracles read.
//
// The layouts are ggml-common.h's block structs (QK_K = 256 values per super-block) and the
// arithmetic is ggml-quants.c's dequantize_row_{q2,q3,q4,q6}_K, restated here in FP64 so no
// production kernel, decoder or shared helper can hide a mistake in either. A test that needs the
// represented weight values of these formats reads them from here, not from the device.

#include "core/weight.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::test::gguf_reference {

constexpr int kSuperBlock = 256;

// One binary16 value as FP64, from its bits.
inline double half_to_double(std::uint16_t bits) {
    const std::uint32_t sign     = (bits >> 15) & 1U;
    const std::uint32_t exponent = (bits >> 10) & 0x1FU;
    const std::uint32_t mantissa = bits & 0x3FFU;
    const double magnitude       = sign ? -1.0 : 1.0;
    if (exponent == 0) { return magnitude * std::ldexp(static_cast<double>(mantissa), -24); }
    if (exponent == 31) {
        return mantissa != 0 ? std::numeric_limits<double>::quiet_NaN()
                             : magnitude * std::numeric_limits<double>::infinity();
    }
    return magnitude * std::ldexp(static_cast<double>(mantissa | 0x400U),
                                  static_cast<int>(exponent) - 25);
}

inline std::uint16_t half_bits(const std::uint8_t* p) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, p, sizeof(bits));
    return bits;
}

inline std::uint32_t word(const std::uint8_t* p) {
    std::uint32_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

// Q2_K (84 bytes): scales[16], qs[64], d, dmin. Each of 16 groups of 16 values has a 4-bit scale
// and a 4-bit minimum, and its two bits sit at 2 * (j / 64) of the group's 32 bytes.
inline void decode_q2_k(const std::uint8_t* block, double* out) {
    const double d    = half_to_double(half_bits(block + 80));
    const double dmin = half_to_double(half_bits(block + 82));
    const std::uint8_t* scales = block;
    const std::uint8_t* qs     = block + 16;
    int is = 0;
    for (int n = 0; n < kSuperBlock; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                const std::uint8_t sc = scales[is++];
                const double dl       = d * double(sc & 0xF);
                const double ml       = dmin * double(sc >> 4);
                const std::uint8_t* lane = qs + half * 16;
                for (int l = 0; l < 16; ++l) {
                    *out++ = dl * double((lane[l] >> shift) & 3U) - ml;
                }
            }
            shift += 2;
        }
        qs += 32;
    }
}

// Q3_K (110 bytes): hmask[32], qs[64] (low two bits), scales[12] (six bits each, packed in four
// 32-bit words), d. Value j is scale * (low - (hmask bit ? 0 : 4)).
inline void decode_q3_k(const std::uint8_t* block, double* out) {
    const double d = half_to_double(half_bits(block + 108));
    const std::uint8_t* hmask      = block;
    const std::uint8_t* qs         = block + 32;
    const std::uint8_t* packed     = block + 96;
    constexpr std::uint32_t kmask1 = 0x03030303U;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0fU;
    std::uint32_t aux[4]           = {word(packed), word(packed + 4), word(packed + 8), 0};
    const std::uint32_t tmp        = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::int8_t scales[16];
    std::memcpy(scales, aux, sizeof(scales));

    int is = 0;
    std::uint8_t mask = 1;
    for (int n = 0; n < kSuperBlock; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                const double dl          = d * double(scales[is++] - 32);
                const std::uint8_t* lane = qs + half * 16;
                const std::uint8_t* hm   = hmask + half * 16;
                for (int l = 0; l < 16; ++l) {
                    const int low = int((lane[l] >> shift) & 3U) - ((hm[l] & mask) ? 0 : 4);
                    *out++        = dl * double(low);
                }
            }
            shift += 2;
            mask = static_cast<std::uint8_t>(mask << 1);
        }
        qs += 32;
    }
}

inline void q4_k_scale_min(int j, const std::uint8_t* scales, int* sc, int* m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
        return;
    }
    *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    *m  = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
}

// Q4_K (144 bytes): d, dmin, scales[12], qs[128]. Each of 8 groups of 32 values has a 6-bit scale
// and minimum; the group's 32 bytes hold its low and high nibbles.
inline void decode_q4_k(const std::uint8_t* block, double* out) {
    const double d    = half_to_double(half_bits(block));
    const double dmin = half_to_double(half_bits(block + 2));
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* qs     = block + 16;
    int is = 0;
    for (int j = 0; j < kSuperBlock; j += 64) {
        int sc = 0, m = 0;
        q4_k_scale_min(is + 0, scales, &sc, &m);
        const double d1 = d * double(sc), m1 = dmin * double(m);
        q4_k_scale_min(is + 1, scales, &sc, &m);
        const double d2 = d * double(sc), m2 = dmin * double(m);
        for (int l = 0; l < 32; ++l) { *out++ = d1 * double(qs[l] & 0xF) - m1; }
        for (int l = 0; l < 32; ++l) { *out++ = d2 * double(qs[l] >> 4) - m2; }
        qs += 32;
        is += 2;
    }
}

// Q6_K (210 bytes): ql[128] (low four bits), qh[64] (high two bits), scales[16] (int8), d.
inline void decode_q6_k(const std::uint8_t* block, double* out) {
    const double d            = half_to_double(half_bits(block + 208));
    const std::uint8_t* ql    = block;
    const std::uint8_t* qh    = block + 128;
    const std::int8_t* scales = reinterpret_cast<const std::int8_t*>(block + 192);
    for (int n = 0; n < kSuperBlock; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int((ql[l] & 0xF) | ((qh[l] & 3U) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3U) << 4)) - 32;
            const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3U) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3U) << 4)) - 32;
            out[l]       = d * double(scales[is + 0]) * double(q1);
            out[l + 32]  = d * double(scales[is + 2]) * double(q2);
            out[l + 64]  = d * double(scales[is + 4]) * double(q3);
            out[l + 96]  = d * double(scales[is + 6]) * double(q4);
        }
        out += 128;
        ql += 64;
        qh += 32;
        scales += 8;
    }
}

// Whether this header decodes the type.
inline bool decodes(QType qtype) {
    return qtype == QType::GGUF_Q2_K || qtype == QType::GGUF_Q3_K || qtype == QType::GGUF_Q4_K ||
           qtype == QType::GGUF_Q6_K;
}

// One super-block of `qtype` as its 256 represented values.
inline void decode_block(QType qtype, const std::uint8_t* block, double* out) {
    switch (qtype) {
    case QType::GGUF_Q2_K: return decode_q2_k(block, out);
    case QType::GGUF_Q3_K: return decode_q3_k(block, out);
    case QType::GGUF_Q4_K: return decode_q4_k(block, out);
    case QType::GGUF_Q6_K: return decode_q6_k(block, out);
    default: throw std::invalid_argument("gguf_reference: unsupported block type");
    }
}

// Row-major values of `rows` rows of `columns` values (a whole number of super-blocks).
inline std::vector<double> decode_matrix(QType qtype, const std::uint8_t* bytes, std::int64_t rows,
                                         std::int64_t columns) {
    const auto block = gguf_block_shape(qtype);
    if (!decodes(qtype) || block.elements != kSuperBlock || columns % kSuperBlock != 0) {
        throw std::invalid_argument("gguf_reference: unsupported matrix geometry");
    }
    const std::int64_t per_row = columns / kSuperBlock;
    const std::int64_t stride  = per_row * block.bytes;
    std::vector<double> out(static_cast<std::size_t>(rows * columns));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t b = 0; b < per_row; ++b) {
            decode_block(qtype, bytes + row * stride + b * block.bytes,
                         out.data() + row * columns + b * kSuperBlock);
        }
    }
    return out;
}

} // namespace ninfer::test::gguf_reference
