#ifndef COLORS_COLOR_CONVERSION_H_
#define COLORS_COLOR_CONVERSION_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using std::cbrt;
using std::pow;

namespace {

template <typename Ch>
inline constexpr Ch SRGBChannelToLinear(const Ch& ch) {
  return ch <= Ch(0.04045) ? ch / Ch(12.92)
                           : pow((ch + Ch(0.055)) / Ch(1.055), Ch(2.4));
}

template <typename Ch>
inline constexpr Ch XYZChannelToLABChannel(const Ch& ch) {
  return ch > Ch(216.0 / 24389.0) ? cbrt(ch)
                                  : ch * Ch(7.787) + Ch(16.0 / 116.0);
}

}  // namespace

namespace color {

template <typename Ch>
using Color = std::array<Ch, 3>;

template <typename Ch>
inline constexpr Color<Ch> ExtractSRGB(uint32_t srgb) {
  return {{((srgb & 0x00ff0000U) >> 16U) / Ch(256.0),
           ((srgb & 0x0000ff00U) >> 8U) / Ch(256.0),
           (srgb & 0x000000ffU) / Ch(256.0)}};
}

template <typename Ch>
inline constexpr uint32_t RenderARGB(const Color<Ch>& from,
                                     u_char opacity = 0xff) {
  const Ch& r = from[0];
  const Ch& g = from[1];
  const Ch& b = from[2];
  return static_cast<uint32_t>(opacity << 24U) |
         static_cast<uint32_t>(static_cast<u_char>(r * 256) << 16U) |
         static_cast<uint32_t>(static_cast<u_char>(g * 256) << 8U) |
         static_cast<uint32_t>(static_cast<u_char>(b * 256));
}

static_assert(RenderARGB(ExtractSRGB<double>(0x123456), 0) == 0x123456,
              "RenderARGB and ExtractSRGB should be inverse");

inline constexpr uint32_t RenderABGR(uint32_t int_srgb, u_char opacity = 0xff) {
  const uint32_t b = (int_srgb & 0x00ff0000U) >> 16U;
  const uint32_t g = (int_srgb & 0x0000ff00U);
  const uint32_t r = (int_srgb & 0x000000ffU) << 16U;
  const uint32_t a = static_cast<uint32_t>(opacity) << 24U;
  return a | r | g | b;
}

template <typename Ch>
inline constexpr Color<Ch> SRGBToLinearRGB(const Color<Ch>& from) {
  const Ch& r = from[0];
  const Ch& g = from[1];
  const Ch& b = from[2];
  return {{SRGBChannelToLinear(r),  //
           SRGBChannelToLinear(g),  //
           SRGBChannelToLinear(b)}};
}

template <typename Ch>
inline constexpr Color<Ch> LinearRGBToXYZ(const Color<Ch>& from) {
  const Ch& r = from[0];
  const Ch& g = from[1];
  const Ch& b = from[2];
  Ch x = r * Ch(0.412424) + g * Ch(0.212656) + b * Ch(0.0193324);
  Ch y = r * Ch(0.357579) + g * Ch(0.715158) + b * Ch(0.119193);
  Ch z = r * Ch(0.180464) + g * Ch(0.0721856) + b * Ch(0.950444);
  return {{x, y, z}};
}

template <typename Ch>
inline constexpr Color<Ch> XYZToLAB(const Color<Ch>& from) {
  Ch x = XYZChannelToLABChannel(from[0]);
  Ch y = XYZChannelToLABChannel(from[1]);
  Ch z = XYZChannelToLABChannel(from[2]);
  Ch l = y * Ch(116.0) - Ch(16.0);
  Ch a = (x - y) * Ch(500.0);
  Ch b = (y - z) * Ch(200.0);
  return {{l, a, b}};
}

template <typename Ch>
inline constexpr Color<Ch> XYZToLUV(const Color<Ch>& from) {
  Ch x = XYZChannelToLABChannel(from[0]);
  Ch y = XYZChannelToLABChannel(from[1]);
  Ch z = XYZChannelToLABChannel(from[2]);
  Ch denom = x + Ch(15.0) * y + Ch(3.0) * z;
  Ch u, v;
  if (denom == Ch(0.0)) {
    u = Ch(0.0);
    v = Ch(0.0);
  } else {
    u = Ch(4.0) * x / denom;
    v = Ch(9.0) * y / denom;
  }
  Ch lab_y = XYZChannelToLABChannel(y);
  Ch l = lab_y * Ch(116.0) + Ch(16.0);
  u = l * Ch(13.0) * (u - Ch(4.0 / 19.0));
  v = l * Ch(13.0) * (v - Ch(9.0 / 19.0));
  return {{l, u, v}};
}

}  // namespace color

#endif  // COLORS_COLOR_CONVERSION_H_
