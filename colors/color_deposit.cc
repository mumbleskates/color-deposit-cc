#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/types/variant.h"
#include "colors/color_conversion.h"
#include "third_party/lodepng/lodepng.h"
#include "widders/container/kd_metrics.h"
#include "widders/container/scapegoat_kd_map.h"
#include "widders/util/progress.h"

// Commandline flags
ABSL_FLAG(bool, log_progress, false, "Log progress to stdout.");
ABSL_FLAG(std::string, color_metric, "lab",
          "The type of color metric to determine similar colors (one of lab, "
          "luv, xyz, rgb, srgb).");
// TODO(widders): Hilbert... and zigzag?
ABSL_FLAG(
    std::string, ordering, "shuffle",
    "The order in which colors are added to the image. Possible values are "
    "'shuffle' for random, or some variation of 'ordered:+r+g+b', "
    "'ordered:-g+b-r', etc. for a specific ordering from highest to lowest "
    "denomination. Components for specific orderings include r, g, b (SRGB "
    "red, green, and blue channels), l, u, v (CIE LUV color space "
    "coordinates), x, y, and z (CIE XYZ color space) and are each preceded "
    "either by '+' (ascending) or '-' (descending). There can be up to three "
    "such components; for example, 'ordered:-r+b' would be the equivalent of "
    "all colors shuffled, then ordered by ascending blue value, then finally "
    "ordered by descending red value (with green left randomized).");
ABSL_FLAG(
    std::string, origin, "center",
    "The origin in the image at which to start depositing. Possible values are "
    "'center' for the center of the image, 'random' for a random location, "
    "'x,y' with exact integer pixel coordinates, or 'x.xx%,y.yy%' for floating "
    "point percentage coordinates relative to the size of the image.");
// TODO(widders): dimensions: square, tall, wide

using std::cout;
using std::endl;
using std::flush;
using std::string;
using std::vector;

constexpr size_t kColorDims = 3;
constexpr uint32_t kEmpty = ~0U;

constexpr int kAllColorImageWidth = 1U << 12U;
constexpr int kAllColorImageHeight = 1U << 12U;

struct Position {
  int x;
  int y;

  bool operator==(const Position& other) const {
    return x == other.x && y == other.y;
  }
  bool operator<(const Position& other) const {
    return y < other.y || (y == other.y && x < other.x);
  }
  Position operator+(const Position& other) const {
    return {x + other.x, y + other.y};
  }
  template <typename H>
  friend H AbslHashValue(H h, const Position& c) {
    return H::combine(std::move(h), c.x, c.y);
  }
};

using Canvas = vector<uint32_t>;

struct ImageBuffer {
  int width;
  int height;
  Canvas buffer;

  static ImageBuffer OfSize(int width, int height) {
    ImageBuffer result{
        .width = width,
        .height = height,
    };
    result.erase();
    return result;
  }

  bool InBounds(Position pos) const {
    return pos.x >= 0 && pos.y >= 0 && pos.x < width && pos.y < height;
  }

  uint32_t& operator[](Position pos) { return buffer[pos.y * height + pos.x]; }

  const uint32_t& operator[](Position pos) const {
    return buffer[pos.y * height + pos.x];
  }

  size_t size() const { return buffer.size(); }
  void erase() { buffer = Canvas(width * height, kEmpty); }
};

// Type for each color channel field.
using Field = double;
// Type for the frontier set.
using ColorMap = widders::ScapegoatKdMap<kColorDims, Field, Position>;
using Color = color::Color<Field>;
using ColorPair = std::pair<uint32_t, Color>;
using ColorMetric = std::function<Color(uint32_t int_srgb)>;

// Offsets from any occupied pixel to any unoccupied pixel that is valid for
// placement.
const Position kFrontierOffsets[] = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
// Sampling convolution; (offset, weight) pairs that are summed together for all
// occupied pixels near a frontier pixel that determine its 'neighboring' color
// value, and thus the color value that would most like to be placed there.
const std::pair<Position, Field> kSamples[] =       // NOLINT(cert-err58-cpp)
    {{{-1, +0}, 1}, {{-1, -1}, 1 / std::sqrt(2)},   //
     {{+0, -1}, 1}, {{+1, -1}, 1 / std::sqrt(2)},   //
     {{+1, +0}, 1}, {{+1, +1}, 1 / std::sqrt(2)},   //
     {{+0, +1}, 1}, {{-1, +1}, 1 / std::sqrt(2)}};  //

// A pair of functions for generating and then (if necessary) reordering colors
// before placement on the canvas.
struct ColorOrder {
  std::function<uint32_t(uint32_t)> map_channels;
  bool should_shuffle;
  ColorMetric sort_metric;
  std::function<void(vector<ColorPair>*)> do_sort;
};

// Returns a mapping from [0,1) SRGB values to the corresponding color space
// named by the color_metric flag.
ColorMetric GetColorMetric(const string& metric_name) {
  return absl::flat_hash_map<string, ColorMetric>{
      {"srgb",
       [](uint32_t int_srgb) {
         Color srgb = color::ExtractSRGB<Field>(int_srgb);
         return srgb;
       }},
      {"rgb",
       [](uint32_t int_srgb) {
         Color srgb = color::ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = color::SRGBToLinearRGB(srgb);
         return linear_srgb;
       }},
      {"xyz",
       [](uint32_t int_srgb) {
         Color srgb = color::ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = color::SRGBToLinearRGB(srgb);
         Color xyz = color::LinearRGBToXYZ(linear_srgb);
         return xyz;
       }},
      {"lab",
       [](uint32_t int_srgb) {
         Color srgb = color::ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = color::SRGBToLinearRGB(srgb);
         Color xyz = color::LinearRGBToXYZ(linear_srgb);
         Color lab = color::XYZToLAB(xyz);
         return lab;
       }},
      {"luv", [](uint32_t int_srgb) {
         Color srgb = color::ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = color::SRGBToLinearRGB(srgb);
         Color xyz = color::LinearRGBToXYZ(linear_srgb);
         Color luv = color::XYZToLUV(xyz);
         return luv;
       }}}[metric_name];
}

// Parses values for the 'ordering' flag and returns functions for producing the
// corresponding ordering of colors for placement.
absl::variant<string, ColorOrder> ParseOrdering(const string& ordering_name) {
  const auto identity = [](uint32_t c) { return c; };
  enum OrderClass {
    UNKNOWN,
    RGB,
    LUV,
    XYZ,
  };
  constexpr char kOrderClassError[] =
      "Ordering channels must all come from the same colorspace (rgb, luv, "
      "xyz).";

  struct OrderPart {
    int channel = 0;
    bool descending = false;
  };

  if (ordering_name == "shuffle") {
    return ColorOrder{identity, true, nullptr, nullptr};
  } else {
    std::pair<string, string> split =
        absl::StrSplit(ordering_name, absl::MaxSplits(absl::ByChar(':'), 1));
    if (split.second.empty()) {
      return "Expected 'shuffle' or 'class:options' for ordering.";
    } else if (split.first == "ordered") {
      vector<string> options = absl::StrSplit(split.second, absl::ByLength(2));
      if (options.size() > 3) {
        return "Too many ordering options; expected 1-3";
      }

      // Parse each channel and ordering in the options string.
      OrderClass order_class = UNKNOWN;
      vector<OrderPart> ordering_parts;
      // Set of hopefully unique channels.
      absl::flat_hash_set<char> channels_named;
      ordering_parts.reserve(options.size());
      for (const string& part_str : options) {
        if (part_str.size() != 2) {
          return "Ordering options must be two characters each";
        }
        OrderPart part;
        const char part_ascdesc = part_str[0];
        const char part_channel = part_str[1];
        switch (part_ascdesc) {
          case '+': {
            part.descending = false;
            break;
          }
          case '-': {
            part.descending = true;
            break;
          }
          default:
            return "The first character of each ordering option, the order, "
                   "must be one of '+-'";
        }
        if (!channels_named.insert(part_channel).second) {
          // This channel was already named.
          return absl::StrCat("The ordering option with channel '",
                              static_cast<u_char>(part_channel),
                              "' was named twice.");
        }
        switch (part_channel) {
          case 'r': {
            if (order_class == UNKNOWN || order_class == RGB) {
              order_class = RGB;
              part.channel = 0;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'g': {
            if (order_class == UNKNOWN || order_class == RGB) {
              order_class = RGB;
              part.channel = 1;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'b': {
            if (order_class == UNKNOWN || order_class == RGB) {
              order_class = RGB;
              part.channel = 2;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'l': {
            if (order_class == UNKNOWN || order_class == LUV) {
              order_class = LUV;
              part.channel = 0;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'u': {
            if (order_class == UNKNOWN || order_class == LUV) {
              order_class = LUV;
              part.channel = 1;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'v': {
            if (order_class == UNKNOWN || order_class == LUV) {
              order_class = LUV;
              part.channel = 2;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'x': {
            if (order_class == UNKNOWN || order_class == XYZ) {
              order_class = XYZ;
              part.channel = 0;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'y': {
            if (order_class == UNKNOWN || order_class == XYZ) {
              order_class = XYZ;
              part.channel = 1;
              break;
            } else {
              return kOrderClassError;
            }
          }
          case 'z': {
            if (order_class == UNKNOWN || order_class == XYZ) {
              order_class = XYZ;
              part.channel = 2;
              break;
            } else {
              return kOrderClassError;
            }
          }
          default: {
            return "The second character of each ordering option, the channel, "
                   "must be one of 'rgbluvxyz'";
          }
        }
        ordering_parts.push_back(part);
      }

      if (order_class == RGB && ordering_parts.size() == 3) {
        // All three channels are represented; we can just map the input colors,
        // without shuffling or sorting.
        return ColorOrder{
            [ordering_parts](uint32_t input) {
              uint32_t result = 0;
              for (int i = 0; i < 3; ++i) {
                const auto& part = ordering_parts[i];
                uint32_t input_ch = (input & (0xffU << (8U * i))) >> (8U * i);
                result |= (part.descending ? 255U - input_ch : input_ch)
                          << (8 * part.channel);
              }
              return result;
            },
            false, nullptr, nullptr};
      } else {
        // Some channels are not represented; shuffle the input colors first,
        // then sort.
        ColorMetric metric;
        switch (order_class) {
          case RGB: {
            metric = GetColorMetric("raw");
            break;
          }
          case LUV: {
            metric = GetColorMetric("luv");
            break;
          }
          case XYZ: {
            metric = GetColorMetric("xyz");
            break;
          }
          default: {
            return "There is a bug in the sorting color metric function.";
          }
        }

        return ColorOrder{identity, true, metric,
                          [ordering_parts](vector<ColorPair>* colors) -> void {
                            absl::c_stable_sort(
                                *colors,
                                [ordering_parts](const ColorPair& a,
                                                 const ColorPair& b) -> bool {
                                  for (const auto& part : ordering_parts) {
                                    Field a_chan = a.second[part.channel];
                                    Field b_chan = b.second[part.channel];
                                    if (a_chan == b_chan) continue;
                                    return (a_chan < b_chan) ^ part.descending;
                                  }
                                  return false;  // Completely equal.
                                });
                          }};
      }
    } else {
      return "Unrecognized ordering class: " + split.first;
    }
  }
}

// Parses the origin location for the deposit process.
absl::variant<string, Position> ParseOrigin(const string& origin_string,
                                            const ImageBuffer& image,
                                            absl::BitGen& rng) {
  if (origin_string == "center") {
    return Position{image.width / 2, image.height / 2};
  }
  if (origin_string == "random") {
    return Position{
        absl::Uniform(absl::IntervalClosedOpen, rng, 0, image.width),
        absl::Uniform(absl::IntervalClosedOpen, rng, 0, image.height),
    };
  }
  std::pair<string, string> coords =
      absl::StrSplit(origin_string, absl::MaxSplits(absl::ByChar(','), 1));
  std::string &x = coords.first, y = coords.second;
  absl::StripAsciiWhitespace(&x);
  absl::StripAsciiWhitespace(&y);
  if (y.empty() || (absl::EndsWith(x, "%") != absl::EndsWith(y, "%"))) {
    return "Expected 'center', 'x,y', or 'x.xx%,y.yy%' format for origin "
           "coordinates.";
  }
  if (absl::EndsWith(x, "%")) {
    // Remove trailing %
    x.resize(x.size() - 1);
    y.resize(y.size() - 1);
    double x_percent;
    double y_percent;
    if (absl::SimpleAtod(x, &x_percent) && absl::SimpleAtod(y, &y_percent)) {
      if (x_percent < 0 || x_percent > 100) {
        return "Origin x percentage must be between 0% and 100%.";
      }
      if (y_percent < 0 || y_percent > 100) {
        return "Origin y percentage must be between 0% and 100%.";
      }
      int x_exact = static_cast<int>(image.width * (x_percent / 100.0));
      int y_exact = static_cast<int>(image.height * (y_percent / 100.0));
      return Position{
          std::min(image.width - 1, x_exact),
          std::min(image.height - 1, y_exact),
      };
    } else {
      return "Could not parse origin percentages.";
    }
  } else {
    // Exact origin coordinates.
    int x_exact;
    int y_exact;
    if (absl::SimpleAtoi(x, &x_exact) && absl::SimpleAtoi(y, &y_exact)) {
      if (x_exact < 0 || x_exact >= image.width) {
        return absl::StrCat("Origin x coordinate must be between 0 and ",
                            image.width - 1);
      }
      if (y_exact < 0 || y_exact >= image.height) {
        return absl::StrCat("Origin x coordinate must be between 0 and ",
                            image.height - 1);
      }
      return Position{
          .x = x_exact,
          .y = y_exact,
      };
    } else {
      return "Could not parse origin coordinates.";
    }
  }
}

// RNG instance
static absl::BitGen rng;

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage("Color deposit: growing images like crystals.");
  const std::vector<char*> cmd_args = absl::ParseCommandLine(argc, argv);

  const ColorMetric metric = GetColorMetric(absl::GetFlag(FLAGS_color_metric));
  if (!metric) {
    cout << "Invalid color metric name" << endl;
    return 1;
  }
  const auto maybe_ordering = ParseOrdering(absl::GetFlag(FLAGS_ordering));
  if (absl::holds_alternative<string>(maybe_ordering)) {
    cout << absl::get<string>(maybe_ordering) << endl;
    return 1;
  }
  const auto& ordering = absl::get<ColorOrder>(maybe_ordering);

  vector<ColorPair> color_values;
  color_values.reserve(kAllColorImageWidth * kAllColorImageHeight);

  cout << "Generating colors..." << flush;
  for (uint32_t int_srgb = 0; int_srgb < 0x1000000U; ++int_srgb) {
    uint32_t mapped = ordering.map_channels(int_srgb);
    color_values.emplace_back<ColorPair>({mapped, {}});
  }
  cout << "done" << endl;

  if (ordering.should_shuffle) {
    cout << "Shuffling..." << flush;
    std::shuffle(color_values.begin(), color_values.end(), rng);
    cout << "done" << endl;
  }
  if (ordering.sort_metric) {
    cout << "Mapping color values for sort..." << flush;
    for (ColorPair& pair : color_values) {
      pair.second = ordering.sort_metric(pair.first);
    }
    cout << "done" << endl;
  }
  if (ordering.do_sort) {
    cout << "Sorting..." << flush;
    ordering.do_sort(&color_values);
    cout << "done" << endl;
  }

  cout << "Mapping color values for fill..." << flush;
  for (ColorPair& pair : color_values) {
    pair.second = metric(pair.first);
  }
  cout << "done" << endl;

  auto image = ImageBuffer::OfSize(kAllColorImageWidth, kAllColorImageHeight);
  auto maybe_origin = ParseOrigin(absl::GetFlag(FLAGS_origin), image, rng);
  if (absl::holds_alternative<string>(maybe_origin)) {
    cout << "Error: " << absl::get<string>(maybe_origin) << endl;
    return 1;
  }
  auto origin = absl::get<Position>(maybe_origin);

  cout << "Filling..." << flush;
  auto frontier = ColorMap(1.4);
  ProgressOptions progress_options;
  progress_options.output_function =
      [&frontier, image_size = image.size()](ProgressStats stats) {
        cout << stats.progress << " pixels done; "     // absolute progress
             << stats.recent_update_rate << " px/s; "  // rate
             << "ETA " << absl::Trunc(stats.eta(image_size), absl::Seconds(0.1))
             << "; "                                          // ETA
             << "frontier size " << frontier.size() << endl;  // frontier stats
      };
  Progress progress(progress_options);

  using Distance =
      widders::TiebreakingDistance<widders::EuclideanDistance<long double>,
                                   widders::RandomTiebreak<decltype(rng), rng>>;

  // Create initial frontier.
  frontier.set({origin.x, origin.y}, Color());
  // Repeatedly place a pixel in the image, updating the frontier set and all
  // affected color values of neighboring frontier pixels.
  const bool logging_enabled = absl::GetFlag(FLAGS_log_progress);
  for (size_t i = 0; i < image.size(); ++i) {
    if (logging_enabled) progress.update(i);
    auto result = frontier.nearest<Distance>(color_values[i].second);
#ifdef KD_TREE_DEBUG
#ifdef NDEBUG
#error NDEBUG disables compilation of k-d tree validation.
#endif
    frontier.validate();
#endif
    const Position& pos = result.key;
    frontier.erase(pos);
    image[pos] = static_cast<uint32_t>(i);
    for (const Position& f_delta : kFrontierOffsets) {
      Position frontier_pos = pos + f_delta;
      // Do nothing if this is not an empty frontier point.
      if (!image.InBounds(frontier_pos) || image[frontier_pos] != kEmpty) {
        continue;
      }
      // Sum surrounding points for the new value of this frontier.
      auto new_mean = Color();
      Field total_weight = 0;
      for (const auto& sample : kSamples) {
        const Position& s_delta = sample.first;
        const Field& sample_weight = sample.second;
        Position sample_pos = frontier_pos + s_delta;
        if (!image.InBounds(sample_pos)) continue;
        const uint32_t sample_idx = image[sample_pos];
        // Sum values of non-empty neighbors into new_mean.
        if (sample_idx == kEmpty) continue;
        total_weight += sample_weight;
        const Color& sample_value = color_values[sample_idx].second;
        for (size_t ch = 0; ch < kColorDims; ++ch) {
          new_mean[ch] += sample_value[ch] * sample_weight;
        }
      }
      // Normalize value from total sampled weight.
      for (auto& d : new_mean) d /= total_weight;
      frontier.set(frontier_pos, new_mean);
#ifdef KD_TREE_DEBUG
      frontier.validate();
#endif
    }
  }
  absl::Duration elapsed = progress.elapsed();
  cout << "done; Total " << absl::Trunc(elapsed, absl::Seconds(0.1))
       << "; Average " << image.size() / absl::ToDoubleSeconds(elapsed)
       << " px/s" << endl;

  cout << "Unwrapping colors..." << flush;
  for (auto& pixel : image.buffer) {
    pixel = color::RenderABGR(color_values[pixel].first);
  }
  cout << "done" << endl;

  // Release memory for color values.
  color_values.clear();

  cout << "Encoding to file..." << flush;
  const string output_filename =
      cmd_args.size() <= 1 ? "output.png" : cmd_args[1];
  auto error = lodepng::encode(
      output_filename, reinterpret_cast<unsigned char*>(image.buffer.data()),
      image.width, image.height);
  if (error) {
    cout << "encoder error " << error << ": " << lodepng_error_text(error)
         << endl;
  } else {
    cout << "done" << endl;
  }

  return error;
}
