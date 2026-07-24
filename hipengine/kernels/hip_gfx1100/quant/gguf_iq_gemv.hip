// Raw selected-MoE GEMV kernels for GGUF IQ2_XS, IQ3_XXS, and IQ4_XS experts.
//
// Quant decoding follows llama.cpp@1ebf790cd. Decode scheduling starts from
// qwen-kernel@52e240f: IQ3 maps one 8-value group to each work item and fuses
// gate+up+SiLU; IQ2 reuses that schedule with its llama.cpp block decoder. IQ4
// weighted down reduces and BF16-rounds each selected route
// before applying its routing weight in fallback order.  hipEngine keeps single
// selected primitives as the required unfused fallback.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int IQ2_XS_BLOCK_BYTES = 74;
constexpr int IQ3_XXS_BLOCK_BYTES = 98;
constexpr int IQ4_XS_BLOCK_BYTES = 136;

// Compressed llama.cpp iq2xs_grid. Each u16 contains eight 2-bit selectors
// mapped to magnitudes 8, 25, and 43.
__device__ __constant__ uint16_t IQ2_XS_GRID_PACKED[512] = {
    0x0000u, 0x0002u, 0x0005u, 0x0008u, 0x000au, 0x0011u, 0x0014u, 0x0016u, 0x0019u, 0x0020u, 0x0022u, 0x0025u, 0x0028u, 0x0041u, 0x0044u, 0x0046u,
    0x0049u, 0x0050u, 0x0052u, 0x0055u, 0x0058u, 0x0061u, 0x0064u, 0x0080u, 0x0082u, 0x0085u, 0x0088u, 0x0091u, 0x0094u, 0x0099u, 0x00a0u, 0x0101u,
    0x0104u, 0x0106u, 0x0109u, 0x0110u, 0x0112u, 0x0115u, 0x0118u, 0x011au, 0x0121u, 0x0124u, 0x0140u, 0x0142u, 0x0145u, 0x0148u, 0x0151u, 0x0154u,
    0x0160u, 0x0168u, 0x0181u, 0x0184u, 0x0190u, 0x0200u, 0x0202u, 0x0205u, 0x0208u, 0x0211u, 0x0214u, 0x0220u, 0x0241u, 0x0244u, 0x0250u, 0x0255u,
    0x0280u, 0x028au, 0x0401u, 0x0404u, 0x0406u, 0x0409u, 0x0410u, 0x0412u, 0x0415u, 0x0418u, 0x0421u, 0x0424u, 0x0440u, 0x0442u, 0x0445u, 0x0448u,
    0x0451u, 0x0454u, 0x0456u, 0x0460u, 0x0481u, 0x0484u, 0x0490u, 0x0500u, 0x0502u, 0x0505u, 0x0508u, 0x0511u, 0x0514u, 0x0520u, 0x0541u, 0x0544u,
    0x0550u, 0x0561u, 0x0580u, 0x0601u, 0x0604u, 0x0610u, 0x0626u, 0x0640u, 0x0642u, 0x0684u, 0x0800u, 0x0802u, 0x0805u, 0x0808u, 0x080au, 0x0811u,
    0x0814u, 0x0820u, 0x0825u, 0x0841u, 0x0844u, 0x0850u, 0x0858u, 0x0880u, 0x08a0u, 0x08aau, 0x0901u, 0x0904u, 0x0910u, 0x0940u, 0x0981u, 0x0989u,
    0x0a00u, 0x0a20u, 0x0a28u, 0x0a96u, 0x0aa0u, 0x1001u, 0x1004u, 0x1006u, 0x1009u, 0x1010u, 0x1012u, 0x1015u, 0x1018u, 0x1021u, 0x1024u, 0x1040u,
    0x1042u, 0x1045u, 0x1048u, 0x1051u, 0x1054u, 0x1060u, 0x106au, 0x1081u, 0x1084u, 0x1090u, 0x1100u, 0x1102u, 0x1105u, 0x1108u, 0x1111u, 0x1114u,
    0x1120u, 0x1141u, 0x1144u, 0x1150u, 0x1180u, 0x1194u, 0x1196u, 0x1201u, 0x1204u, 0x1206u, 0x1210u, 0x1240u, 0x1260u, 0x1400u, 0x1402u, 0x1405u,
    0x1408u, 0x1411u, 0x1414u, 0x1420u, 0x1441u, 0x1444u, 0x1449u, 0x1450u, 0x1464u, 0x1480u, 0x1501u, 0x1504u, 0x1510u, 0x1540u, 0x1600u, 0x1614u,
    0x1649u, 0x1801u, 0x1804u, 0x1810u, 0x1812u, 0x1840u, 0x1854u, 0x1886u, 0x1900u, 0x1905u, 0x1966u, 0x1a51u, 0x1aa9u, 0x2000u, 0x2002u, 0x2005u,
    0x2008u, 0x200au, 0x2011u, 0x2014u, 0x2020u, 0x2041u, 0x2044u, 0x2050u, 0x2080u, 0x20a0u, 0x2101u, 0x2104u, 0x2110u, 0x2140u, 0x2148u, 0x2165u,
    0x2200u, 0x2222u, 0x2280u, 0x22a8u, 0x2401u, 0x2404u, 0x2410u, 0x2429u, 0x2440u, 0x2500u, 0x2541u, 0x2552u, 0x2599u, 0x2601u, 0x261au, 0x26a6u,
    0x2800u, 0x2808u, 0x280au, 0x2820u, 0x2855u, 0x2888u, 0x28a2u, 0x2968u, 0x2990u, 0x2a08u, 0x2a20u, 0x2a82u, 0x2a88u, 0x2a8au, 0x4001u, 0x4004u,
    0x4006u, 0x4009u, 0x4010u, 0x4012u, 0x4015u, 0x4018u, 0x4021u, 0x4024u, 0x4040u, 0x4042u, 0x4045u, 0x4048u, 0x404au, 0x4051u, 0x4054u, 0x4060u,
    0x4065u, 0x4081u, 0x4084u, 0x4090u, 0x4100u, 0x4102u, 0x4105u, 0x4108u, 0x4111u, 0x4114u, 0x4120u, 0x4141u, 0x4144u, 0x4150u, 0x4180u, 0x4185u,
    0x41a2u, 0x4201u, 0x4204u, 0x4210u, 0x4212u, 0x4229u, 0x4240u, 0x4400u, 0x4402u, 0x4405u, 0x4408u, 0x4411u, 0x4414u, 0x4419u, 0x4420u, 0x4441u,
    0x4444u, 0x4450u, 0x4480u, 0x4494u, 0x4501u, 0x4504u, 0x4510u, 0x4524u, 0x4540u, 0x459au, 0x4600u, 0x460au, 0x4644u, 0x4650u, 0x4801u, 0x4804u,
    0x4810u, 0x4840u, 0x4845u, 0x4854u, 0x4862u, 0x4900u, 0x4911u, 0x4944u, 0x4950u, 0x4969u, 0x4a04u, 0x5000u, 0x5002u, 0x5005u, 0x5008u, 0x5011u,
    0x5014u, 0x5020u, 0x5028u, 0x5041u, 0x5044u, 0x5050u, 0x5080u, 0x5101u, 0x5104u, 0x5110u, 0x5115u, 0x5140u, 0x5142u, 0x5200u, 0x5244u, 0x52aau,
    0x5401u, 0x5404u, 0x5410u, 0x5421u, 0x5440u, 0x5460u, 0x5481u, 0x54a1u, 0x5500u, 0x5508u, 0x5580u, 0x5588u, 0x5621u, 0x5668u, 0x56a1u, 0x5800u,
    0x5814u, 0x5841u, 0x5850u, 0x5899u, 0x591au, 0x5940u, 0x5942u, 0x5a85u, 0x6001u, 0x6004u, 0x6010u, 0x6040u, 0x6054u, 0x6062u, 0x6086u, 0x60a9u,
    0x6100u, 0x6224u, 0x624au, 0x6292u, 0x6400u, 0x6416u, 0x6510u, 0x6540u, 0x6545u, 0x65a4u, 0x6801u, 0x686au, 0x6925u, 0x6a06u, 0x6a54u, 0x6a62u,
    0x8000u, 0x8002u, 0x8005u, 0x8008u, 0x8011u, 0x8014u, 0x8020u, 0x802au, 0x8041u, 0x8044u, 0x8050u, 0x8080u, 0x8082u, 0x80a8u, 0x80aau, 0x8101u,
    0x8104u, 0x8106u, 0x8110u, 0x8140u, 0x8151u, 0x8159u, 0x8200u, 0x8220u, 0x8280u, 0x8282u, 0x82a0u, 0x82a8u, 0x8401u, 0x8404u, 0x8410u, 0x8412u,
    0x8415u, 0x8440u, 0x8460u, 0x8489u, 0x8500u, 0x8544u, 0x85a5u, 0x8618u, 0x866au, 0x8800u, 0x8808u, 0x8825u, 0x885au, 0x8880u, 0x8882u, 0x88a8u,
    0x8906u, 0x8a22u, 0x8a80u, 0x8a88u, 0x8a96u, 0x8aa8u, 0x9001u, 0x9004u, 0x9010u, 0x9040u, 0x9056u, 0x9084u, 0x9100u, 0x9122u, 0x9164u, 0x9256u,
    0x9289u, 0x9400u, 0x9405u, 0x9444u, 0x9450u, 0x9458u, 0x9529u, 0x9590u, 0x9592u, 0x9641u, 0x9851u, 0x98a6u, 0x9949u, 0x9a15u, 0x9a60u, 0xa000u,
    0xa002u, 0xa008u, 0xa00au, 0xa020u, 0xa02au, 0xa0a0u, 0xa151u, 0xa159u, 0xa1a6u, 0xa200u, 0xa202u, 0xa208u, 0xa22au, 0xa280u, 0xa2a0u, 0xa440u,
    0xa495u, 0xa665u, 0xa698u, 0xa80au, 0xa820u, 0xa822u, 0xa828u, 0xa8a0u, 0xa8a8u, 0xa904u, 0xa984u, 0xa986u, 0xaa28u, 0xaa2au, 0xaa91u, 0xaaaau,
};

// Generated from llama.cpp ggml/src/ggml-common.h (MIT).  Each u32 packs four
// unsigned IQ3 grid magnitudes in little-endian byte order.
__device__ __constant__ uint32_t IQ3_XXS_GRID[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};

__device__ __constant__ int8_t IQ4_NL_VALUES[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
};

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

__device__ inline uint16_t load_u16_le(const uint8_t* ptr) {
  return static_cast<uint16_t>(ptr[0]) |
         (static_cast<uint16_t>(ptr[1]) << 8);
}

__device__ inline uint32_t load_u32_le(const uint8_t* ptr) {
  return static_cast<uint32_t>(ptr[0]) |
         (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

__device__ inline uint32_t iq3_sign_byte(uint32_t selector) {
  return selector | ((static_cast<uint32_t>(__popc(selector)) & 1U) << 7);
}

// IQ2_XS_GRID_PACKED only contains selector codes 0, 1, and 2. This exact
// arithmetic mapping avoids divergent control flow for magnitudes 8, 25, 43.
__device__ inline float iq2_xs_signed_magnitude(uint32_t code, uint32_t sign) {
  const uint32_t magnitude = 8U + 17U * code + (code >> 1);
  union FloatBits {
    float value;
    uint32_t bits;
  } signed_magnitude;
  signed_magnitude.value = static_cast<float>(magnitude);
  signed_magnitude.bits |= (sign & 1U) << 31;
  return signed_magnitude.value;
}

__device__ inline float iq2_xs_group_dot(
    const uint8_t* __restrict__ block,
    int group8,
    const float* __restrict__ x_values) {
  const int group32 = group8 >> 2;
  const int local8 = group8 & 3;
  const uint16_t q = load_u16_le(block + 2 + 2 * (4 * group32 + local8));
  const uint8_t scale_byte = block[66 + group32];
  const uint32_t scale =
      (static_cast<uint32_t>(scale_byte) >> (4 * (local8 >> 1))) & 0x0FU;
  const float db = fp16_bytes_to_float(block) *
                   (0.5f + static_cast<float>(scale)) * 0.25f;
  const uint32_t signs = iq3_sign_byte(static_cast<uint32_t>(q >> 9));
  const uint16_t grid = IQ2_XS_GRID_PACKED[q & 511U];

  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t code = (static_cast<uint32_t>(grid) >> (2 * j)) & 3U;
    const float signed_magnitude =
        iq2_xs_signed_magnitude(code, (signs >> j) & 1U);
    sum += signed_magnitude * x_values[j];
  }
  return db * sum;
}

__device__ inline float iq2_xs_pair16_dot(
    const uint8_t* __restrict__ block,
    int group8_even,
    const float* __restrict__ x_values) {
  const int group32 = group8_even >> 2;
  const int local8 = group8_even & 3;
  const uint32_t packed_q =
      load_u32_le(block + 2 + 2 * (4 * group32 + local8));
  const uint16_t q0 = static_cast<uint16_t>(packed_q);
  const uint16_t q1 = static_cast<uint16_t>(packed_q >> 16);
  const uint8_t scale_byte = block[66 + group32];
  const uint32_t scale =
      (static_cast<uint32_t>(scale_byte) >> (4 * (local8 >> 1))) & 0x0FU;
  const float db = fp16_bytes_to_float(block) *
                   (0.5f + static_cast<float>(scale)) * 0.25f;
  const uint32_t signs0 = iq3_sign_byte(static_cast<uint32_t>(q0 >> 9));
  const uint32_t signs1 = iq3_sign_byte(static_cast<uint32_t>(q1 >> 9));
  const uint16_t grid0 = IQ2_XS_GRID_PACKED[q0 & 511U];
  const uint16_t grid1 = IQ2_XS_GRID_PACKED[q1 & 511U];
  float sum0 = 0.0f;
  float sum1 = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t code0 = (static_cast<uint32_t>(grid0) >> (2 * j)) & 3U;
    const uint32_t code1 = (static_cast<uint32_t>(grid1) >> (2 * j)) & 3U;
    sum0 += iq2_xs_signed_magnitude(code0, (signs0 >> j) & 1U) * x_values[j];
    sum1 += iq2_xs_signed_magnitude(code1, (signs1 >> j) & 1U) * x_values[8 + j];
  }
  return db * sum0 + db * sum1;
}

__device__ inline float iq3_xxs_group_dot(
    const uint8_t* __restrict__ block,
    int group8,
    const float* __restrict__ x_values) {
  const uint8_t* __restrict__ qs = block + 2;
  const int group32 = group8 >> 2;
  const int local8 = group8 & 3;
  const uint32_t aux = load_u32_le(qs + 64 + 4 * group32);
  const float d = fp16_bytes_to_float(block);
  const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
  const uint32_t signs = iq3_sign_byte((aux >> (7 * local8)) & 127U);
  const uint32_t grid1 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8]];
  const uint32_t grid2 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8 + 1]];

  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float magnitude1 = static_cast<float>((grid1 >> (8 * j)) & 255U);
    const float magnitude2 = static_cast<float>((grid2 >> (8 * j)) & 255U);
    sum += ((signs & (1U << j)) != 0U ? -magnitude1 : magnitude1) * x_values[j];
    sum += ((signs & (1U << (j + 4))) != 0U ? -magnitude2 : magnitude2) *
           x_values[j + 4];
  }
  return db * sum;
}

__device__ inline float iq4_xs_subblock_dot(
    const uint8_t* __restrict__ block,
    int subblock,
    const uint16_t* __restrict__ x) {
  const float d = fp16_bytes_to_float(block);
  const uint16_t scales_h = load_u16_le(block + 2);
  const uint8_t scale_l_byte = block[4 + (subblock >> 1)];
  const uint32_t scale_l =
      (static_cast<uint32_t>(scale_l_byte) >> (4 * (subblock & 1))) & 0x0FU;
  const uint32_t scale_h =
      (static_cast<uint32_t>(scales_h) >> (2 * subblock)) & 0x03U;
  const int scale = static_cast<int>(scale_l | (scale_h << 4)) - 32;
  const float dl = d * static_cast<float>(scale);
  const uint8_t* __restrict__ qs = block + 8 + subblock * 16;

  float sum_lo = 0.0f;
  float sum_hi = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint8_t packed = qs[j];
    sum_lo += static_cast<float>(IQ4_NL_VALUES[packed & 0x0FU]) *
              bf16_bits_to_float(x[j]);
    sum_hi += static_cast<float>(IQ4_NL_VALUES[packed >> 4]) *
              bf16_bits_to_float(x[j + 16]);
  }
  return dl * (sum_lo + sum_hi);
}

__device__ inline float reduce_block_sum(float value) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  __shared__ float wave_sums[8];
  if (lane == 0) {
    wave_sums[wave] = value;
  }
  __syncthreads();
  float total = 0.0f;
  if (tid == 0) {
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      total += wave_sums[i];
    }
  }
  return total;
}

__device__ inline void reduce_block_pair(float& value_a, float& value_b) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value_a += __shfl_down(value_a, offset);
    value_b += __shfl_down(value_b, offset);
  }
  __shared__ float wave_sums_a[8];
  __shared__ float wave_sums_b[8];
  if (lane == 0) {
    wave_sums_a[wave] = value_a;
    wave_sums_b[wave] = value_b;
  }
  __syncthreads();
  if (tid == 0) {
    value_a = 0.0f;
    value_b = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      value_a += wave_sums_a[i];
      value_b += wave_sums_b[i];
    }
  }
}

__device__ inline void reduce_block_quad(
    float& value_a,
    float& value_b,
    float& value_c,
    float& value_d) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value_a += __shfl_down(value_a, offset);
    value_b += __shfl_down(value_b, offset);
    value_c += __shfl_down(value_c, offset);
    value_d += __shfl_down(value_d, offset);
  }
  __shared__ float wave_sums_a[8];
  __shared__ float wave_sums_b[8];
  __shared__ float wave_sums_c[8];
  __shared__ float wave_sums_d[8];
  if (lane == 0) {
    wave_sums_a[wave] = value_a;
    wave_sums_b[wave] = value_b;
    wave_sums_c[wave] = value_c;
    wave_sums_d[wave] = value_d;
  }
  __syncthreads();
  if (tid == 0) {
    value_a = 0.0f;
    value_b = 0.0f;
    value_c = 0.0f;
    value_d = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      value_a += wave_sums_a[i];
      value_b += wave_sums_b[i];
      value_c += wave_sums_c[i];
      value_d += wave_sums_d[i];
    }
  }
}

__device__ inline int selected_x_row(int row, int x_rows, int rows) {
  const int lanes_per_x_row = rows / x_rows;
  return x_rows == 1 ? 0 : row / lanes_per_x_row;
}

__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  if (valid_expert) {
    const int pairs16 = in_features / 16;
    for (int pair16 = threadIdx.x; pair16 < pairs16;
         pair16 += blockDim.x) {
      const int block_idx = pair16 >> 4;
      const int group8_even = (pair16 & 15) * 2;
      const int x_offset = pair16 * 16;
      float x_values[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const uint8_t* __restrict__ block =
          weight_row + static_cast<int64_t>(block_idx) * IQ2_XS_BLOCK_BYTES;
      acc += iq2_xs_pair16_dot(block, group8_even, x_values);
    }
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(total);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_dual_silu_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ gate_row =
      valid_expert ? gate_weight + weight_offset : gate_weight;
  const uint8_t* __restrict__ up_row =
      valid_expert ? up_weight + weight_offset : up_weight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  if (valid_expert) {
    const int pairs16 = in_features / 16;
    for (int pair16 = threadIdx.x; pair16 < pairs16;
         pair16 += blockDim.x) {
      const int block_idx = pair16 >> 4;
      const int group8_even = (pair16 & 15) * 2;
      const int x_offset = pair16 * 16;
      float x_values[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const int64_t block_offset =
          static_cast<int64_t>(block_idx) * IQ2_XS_BLOCK_BYTES;
      gate_acc += iq2_xs_pair16_dot(
          gate_row + block_offset, group8_even, x_values);
      up_acc += iq2_xs_pair16_dot(
          up_row + block_offset, group8_even, x_values);
    }
  }
  reduce_block_pair(gate_acc, up_acc);
  if (threadIdx.x == 0) {
    const float gate = bf16_bits_to_float(float_to_bf16_bits(gate_acc));
    const float up = bf16_bits_to_float(float_to_bf16_bits(up_acc));
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(gate * sig * up);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_gemv_tile2_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = 2 * static_cast<int>(blockIdx.x);
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const bool has_second = out_col + 1 < out_features;
  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ weight_row0 =
      valid_expert ? qweight + weight_offset : qweight;
  const uint8_t* __restrict__ weight_row1 =
      valid_expert && has_second ? weight_row0 + row_bytes : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  if (valid_expert) {
    const int pairs16 = in_features / 16;
    for (int pair16 = threadIdx.x; pair16 < pairs16;
         pair16 += blockDim.x) {
      const int block_idx = pair16 >> 4;
      const int group8_even = (pair16 & 15) * 2;
      const int x_offset = pair16 * 16;
      float x_values[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const int64_t block_offset =
          static_cast<int64_t>(block_idx) * IQ2_XS_BLOCK_BYTES;
      acc0 += iq2_xs_pair16_dot(
          weight_row0 + block_offset, group8_even, x_values);
      if (has_second) {
        acc1 += iq2_xs_pair16_dot(
            weight_row1 + block_offset, group8_even, x_values);
      }
    }
  }
  reduce_block_pair(acc0, acc1);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(acc0);
    if (has_second) {
      out[static_cast<int64_t>(row) * out_features + out_col + 1] =
          float_to_bf16_bits(acc1);
    }
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_dual_silu_gemv_tile2_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = 2 * static_cast<int>(blockIdx.x);
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const bool has_second = out_col + 1 < out_features;
  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ gate_row0 =
      valid_expert ? gate_weight + weight_offset : gate_weight;
  const uint8_t* __restrict__ gate_row1 =
      valid_expert && has_second ? gate_row0 + row_bytes : gate_weight;
  const uint8_t* __restrict__ up_row0 =
      valid_expert ? up_weight + weight_offset : up_weight;
  const uint8_t* __restrict__ up_row1 =
      valid_expert && has_second ? up_row0 + row_bytes : up_weight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float gate_acc0 = 0.0f;
  float up_acc0 = 0.0f;
  float gate_acc1 = 0.0f;
  float up_acc1 = 0.0f;
  if (valid_expert) {
    const int pairs16 = in_features / 16;
    for (int pair16 = threadIdx.x; pair16 < pairs16;
         pair16 += blockDim.x) {
      const int block_idx = pair16 >> 4;
      const int group8_even = (pair16 & 15) * 2;
      const int x_offset = pair16 * 16;
      float x_values[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const int64_t block_offset =
          static_cast<int64_t>(block_idx) * IQ2_XS_BLOCK_BYTES;
      gate_acc0 += iq2_xs_pair16_dot(
          gate_row0 + block_offset, group8_even, x_values);
      up_acc0 += iq2_xs_pair16_dot(
          up_row0 + block_offset, group8_even, x_values);
      if (has_second) {
        gate_acc1 += iq2_xs_pair16_dot(
            gate_row1 + block_offset, group8_even, x_values);
        up_acc1 += iq2_xs_pair16_dot(
            up_row1 + block_offset, group8_even, x_values);
      }
    }
  }
  reduce_block_quad(gate_acc0, up_acc0, gate_acc1, up_acc1);
  if (threadIdx.x == 0) {
    const float gate0 = bf16_bits_to_float(float_to_bf16_bits(gate_acc0));
    const float up0 = bf16_bits_to_float(float_to_bf16_bits(up_acc0));
    const float sig0 = 1.0f / (1.0f + expf(-gate0));
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(gate0 * sig0 * up0);
    if (has_second) {
      const float gate1 = bf16_bits_to_float(float_to_bf16_bits(gate_acc1));
      const float up1 = bf16_bits_to_float(float_to_bf16_bits(up_acc1));
      const float sig1 = 1.0f / (1.0f + expf(-gate1));
      out[static_cast<int64_t>(row) * out_features + out_col + 1] =
          float_to_bf16_bits(gate1 * sig1 * up1);
    }
  }
}

__launch_bounds__(32, 16)
__global__ void gguf_iq3_xxs_selected_gemv_k1024_wave4_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  constexpr int IN_FEATURES = 1024;
  constexpr int BLOCKS = IN_FEATURES / QK_K;
  const int lane = threadIdx.x;
  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  constexpr int ROW_BYTES = BLOCKS * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * ROW_BYTES;
  const uint8_t* __restrict__ weight_row = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * ROW_BYTES
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * IN_FEATURES;

  // Reconstruct the local128 baseline's four independent wave partitions in
  // one wave. Each accumulator keeps the same 32-lane dot and shuffle tree;
  // lane zero then adds wave totals in the original 0..3 order.
  float partials[BLOCKS] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (valid_expert) {
#pragma unroll
    for (int block_idx = 0; block_idx < BLOCKS; ++block_idx) {
      const int x_offset = block_idx * QK_K + lane * 8;
      float x_values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const uint8_t* __restrict__ block =
          weight_row + static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
      partials[block_idx] = iq3_xxs_group_dot(block, lane, x_values);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int block_idx = 0; block_idx < BLOCKS; ++block_idx) {
      partials[block_idx] += __shfl_down(partials[block_idx], offset);
    }
  }
  if (lane == 0) {
    float total = 0.0f;
#pragma unroll
    for (int block_idx = 0; block_idx < BLOCKS; ++block_idx) {
      total += partials[block_idx];
    }
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(total);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_selected_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  if (valid_expert) {
    const int groups8 = in_features / 8;
    for (int group8_linear = threadIdx.x; group8_linear < groups8;
         group8_linear += blockDim.x) {
      // Every wave owns one 256-value super-block at a time. Match the
      // selected-dual path's uniform base so the compiler keeps it in SGPRs.
      const int block_idx =
          __builtin_amdgcn_readfirstlane(group8_linear >> 5);
      const int group8 = group8_linear & 31;
      const int x_offset = block_idx * QK_K + group8 * 8;
      float x_values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const uint8_t* __restrict__ block =
          weight_row + static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
      acc += iq3_xxs_group_dot(block, group8, x_values);
    }
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(total);
  }
}

template <int OUTPUT_TILE>
__launch_bounds__(128, 2)
__global__ void gguf_iq3_xxs_selected_gemv_output_tile_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  static_assert(OUTPUT_TILE == 2 || OUTPUT_TILE == 4, "IQ3 output tile must be 2 or 4");
  const int out_col = OUTPUT_TILE * static_cast<int>(blockIdx.x);
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int valid_columns = min(OUTPUT_TILE, out_features - out_col);
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_base = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc[OUTPUT_TILE];
#pragma unroll
  for (int column = 0; column < OUTPUT_TILE; ++column) {
    acc[column] = 0.0f;
  }
  if (valid_expert) {
    const int groups8 = in_features / 8;
    for (int group8_linear = threadIdx.x; group8_linear < groups8;
         group8_linear += blockDim.x) {
      const int block_idx =
          __builtin_amdgcn_readfirstlane(group8_linear >> 5);
      const int group8 = group8_linear & 31;
      const int x_offset = block_idx * QK_K + group8 * 8;
      float x_values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const int64_t block_offset =
          static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
#pragma unroll
      for (int column = 0; column < OUTPUT_TILE; ++column) {
        if (column < valid_columns) {
          const uint8_t* __restrict__ weight_row =
              weight_base + static_cast<int64_t>(column) * row_bytes;
          acc[column] += iq3_xxs_group_dot(
              weight_row + block_offset, group8, x_values);
        }
      }
    }
  }
  if constexpr (OUTPUT_TILE == 2) {
    reduce_block_pair(acc[0], acc[1]);
  } else {
    reduce_block_quad(acc[0], acc[1], acc[2], acc[3]);
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int column = 0; column < OUTPUT_TILE; ++column) {
      if (column < valid_columns) {
        out[static_cast<int64_t>(row) * out_features + out_col + column] =
            float_to_bf16_bits(acc[column]);
      }
    }
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_selected_dual_silu_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ gate_row =
      valid_expert ? gate_weight + weight_offset : gate_weight;
  const uint8_t* __restrict__ up_row =
      valid_expert ? up_weight + weight_offset : up_weight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  if (valid_expert) {
    const int groups8 = in_features / 8;
    for (int group8_linear = threadIdx.x; group8_linear < groups8;
         group8_linear += blockDim.x) {
      // Every wave owns one 256-value super-block at a time. Make that
      // wave-uniform relationship explicit so block bases stay in SGPRs
      // instead of repeating 64-bit address arithmetic in every lane.
      const int block_idx =
          __builtin_amdgcn_readfirstlane(group8_linear >> 5);
      const int group8 = group8_linear & 31;
      const int x_offset = block_idx * QK_K + group8 * 8;
      float x_values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const int64_t block_offset =
          static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
      gate_acc += iq3_xxs_group_dot(gate_row + block_offset, group8, x_values);
      up_acc += iq3_xxs_group_dot(up_row + block_offset, group8, x_values);
    }
  }
  reduce_block_pair(gate_acc, up_acc);
  if (threadIdx.x == 0) {
    // Preserve the registered unfused chain's two projection rounding
    // boundaries before applying the existing BF16 SiLU contract.
    const float gate = bf16_bits_to_float(float_to_bf16_bits(gate_acc));
    const float up = bf16_bits_to_float(float_to_bf16_bits(up_acc));
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(gate * sig * up);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq4_xs_selected_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  if (valid_expert) {
    const int subblocks = in_features / 32;
    for (int unit = threadIdx.x; unit < subblocks; unit += blockDim.x) {
      const int block_idx = unit >> 3;
      const int subblock = unit & 7;
      const uint8_t* __restrict__ block =
          weight_row + static_cast<int64_t>(block_idx) * IQ4_XS_BLOCK_BYTES;
      const uint16_t* __restrict__ x_subblock =
          x_row_ptr + block_idx * QK_K + subblock * 32;
      acc += iq4_xs_subblock_dot(block, subblock, x_subblock);
    }
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(total);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_weighted_selected_down_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const float* __restrict__ routing_weights,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int tokens,
    int top_k,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int token = blockIdx.y;
  if (out_col >= out_features || token >= tokens) return;

  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int groups8 = in_features / 8;
  const int64_t route_base = static_cast<int64_t>(token) * top_k;

  // Preserve the unfused path exactly: reduce and round every selected route
  // to BF16, then accumulate scaled routing weights in slot order before the
  // final BF16 rounding. The trailing barrier protects the reused LDS wave
  // sums before the next route starts its reduction.
  float weighted_total = 0.0f;
  for (int slot = 0; slot < top_k; ++slot) {
    const int64_t route = route_base + slot;
    const int64_t expert_id = selected[route];
    float route_acc = 0.0f;
    if (expert_id >= 0 && expert_id < num_experts) {
      const uint8_t* __restrict__ weight_row =
          qweight + expert_id * expert_bytes +
          static_cast<int64_t>(out_col) * row_bytes;
      const uint16_t* __restrict__ x_row_ptr =
          x + route * in_features;
      for (int group8_linear = threadIdx.x; group8_linear < groups8;
           group8_linear += blockDim.x) {
        const int block_idx =
            __builtin_amdgcn_readfirstlane(group8_linear >> 5);
        const int group8 = group8_linear & 31;
        const int x_offset = block_idx * QK_K + group8 * 8;
        float x_values[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
        }
        const uint8_t* __restrict__ block =
            weight_row + static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
        route_acc += iq3_xxs_group_dot(block, group8, x_values);
      }
    }
    const float route_total = reduce_block_sum(route_acc);
    if (threadIdx.x == 0) {
      const float route_bf16 =
          bf16_bits_to_float(float_to_bf16_bits(route_total));
      weighted_total =
          fmaf(route_bf16, routing_weights[route], weighted_total);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(token) * out_features + out_col] =
        float_to_bf16_bits(weighted_total);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq4_xs_weighted_selected_down_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const float* __restrict__ routing_weights,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int tokens,
    int top_k,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int token = blockIdx.y;
  if (out_col >= out_features || token >= tokens) return;

  const int row_bytes = (in_features / QK_K) * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int subblocks = in_features / 32;
  const int64_t route_base = static_cast<int64_t>(token) * top_k;

  // Preserve the selected-single fallback's numerical boundary: each route
  // projection is reduced and rounded to BF16 before its routing weight is
  // applied. Reusing the block reduction requires a trailing barrier so the
  // next route cannot overwrite wave_sums while thread 0 is consuming it.
  float weighted_total = 0.0f;
  for (int slot = 0; slot < top_k; ++slot) {
    const int64_t expert_id = selected[route_base + slot];
    float route_acc = 0.0f;
    if (expert_id >= 0 && expert_id < num_experts) {
      const uint8_t* __restrict__ weight_row =
          qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
      for (int unit = threadIdx.x; unit < subblocks; unit += blockDim.x) {
        const int block_idx = unit >> 3;
        const int subblock = unit & 7;
        const uint8_t* __restrict__ block =
            weight_row + static_cast<int64_t>(block_idx) * IQ4_XS_BLOCK_BYTES;
        const uint16_t* __restrict__ x_subblock =
            x + (route_base + slot) * in_features + block_idx * QK_K + subblock * 32;
        route_acc += iq4_xs_subblock_dot(block, subblock, x_subblock);
      }
    }
    const float route_total = reduce_block_sum(route_acc);
    if (threadIdx.x == 0) {
      const float route_bf16 =
          bf16_bits_to_float(float_to_bf16_bits(route_total));
      weighted_total += routing_weights[route_base + slot] * route_bf16;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(token) * out_features + out_col] =
        float_to_bf16_bits(weighted_total);
  }
}

inline bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

template <typename Kernel>
int launch_selected(
    Kernel kernel,
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || qweight == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || rows % x_rows != 0 || num_experts <= 0 ||
      in_features <= 0 || in_features % QK_K != 0 || out_features <= 0 ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <int OUTPUT_TILE, typename Kernel>
int launch_selected_output_tiled(
    Kernel kernel,
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || qweight == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || rows % x_rows != 0 || num_experts <= 0 ||
      in_features <= 0 || in_features % QK_K != 0 || out_features <= 0 ||
      threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((out_features + OUTPUT_TILE - 1) / OUTPUT_TILE),
      static_cast<unsigned int>(rows));
  kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_iq2_xs_selected_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected(
      gguf_iq2_xs_selected_gemv_kernel,
      x,
      selected,
      qweight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_gemv_tile2_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || qweight == nullptr ||
      out == nullptr || x_rows <= 0 || rows <= 0 || rows % x_rows != 0 ||
      num_experts <= 0 || in_features <= 0 || in_features % QK_K != 0 ||
      out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>((out_features + 1) / 2),
      static_cast<unsigned int>(rows));
  gguf_iq2_xs_selected_gemv_tile2_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || gate_weight == nullptr ||
      up_weight == nullptr || out == nullptr || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || num_experts <= 0 || in_features <= 0 ||
      in_features % QK_K != 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_iq2_xs_selected_dual_silu_gemv_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      gate_weight,
      up_weight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_silu_gemv_tile2_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || gate_weight == nullptr ||
      up_weight == nullptr || out == nullptr || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || num_experts <= 0 || in_features <= 0 ||
      in_features % QK_K != 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>((out_features + 1) / 2),
      static_cast<unsigned int>(rows));
  gguf_iq2_xs_selected_dual_silu_gemv_tile2_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      gate_weight,
      up_weight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq3_xxs_selected_gemv_k1024_wave4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || qweight == nullptr ||
      out == nullptr || x_rows <= 0 || rows <= 0 || rows % x_rows != 0 ||
      num_experts <= 0 || in_features != 1024 || out_features <= 0 ||
      threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32U);
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_iq3_xxs_selected_gemv_k1024_wave4_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq3_xxs_selected_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected(
      gguf_iq3_xxs_selected_gemv_kernel,
      x,
      selected,
      qweight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_gemv_tile4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_output_tiled<4>(
      gguf_iq3_xxs_selected_gemv_output_tile_kernel<4>,
      x,
      selected,
      qweight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected(
      gguf_iq4_xs_selected_gemv_kernel,
      x,
      selected,
      qweight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || gate_weight == nullptr ||
      up_weight == nullptr || out == nullptr || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || num_experts <= 0 || in_features <= 0 ||
      in_features % QK_K != 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_iq3_xxs_selected_dual_silu_gemv_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      gate_weight,
      up_weight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq3_xxs_weighted_selected_down_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const float* routing_weights,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || routing_weights == nullptr ||
      qweight == nullptr || out == nullptr || tokens <= 0 || top_k <= 0 ||
      num_experts <= 0 || in_features <= 0 || in_features % QK_K != 0 ||
      out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens));
  gguf_iq3_xxs_weighted_selected_down_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      routing_weights,
      qweight,
      out,
      static_cast<int>(tokens),
      static_cast<int>(top_k),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_xs_weighted_selected_down_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const float* routing_weights,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || routing_weights == nullptr ||
      qweight == nullptr || out == nullptr || tokens <= 0 || top_k <= 0 ||
      num_experts <= 0 || in_features <= 0 || in_features % QK_K != 0 ||
      out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens));
  gguf_iq4_xs_weighted_selected_down_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      routing_weights,
      qweight,
      out,
      static_cast<int>(tokens),
      static_cast<int>(top_k),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}
