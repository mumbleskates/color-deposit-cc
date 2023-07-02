#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <variant>
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
// TODO(widders): Mode to read from existing file in some order and re-deposit
//  instead of using all-colors
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
ABSL_FLAG(
    bool, legacy_stale_diagonals, false,
    "Enable a long-standing, technically-buggy behavior where frontier pixels "
    "that were diagonal of a newly-placed pixel did not get updated with a "
    "new color average.");
// TODO(widders): dimensions: square, tall, wide
// TODO(widders): try out slightly fuzzing choice of best placement
//  (cheapening lookup)

using std::cout;
using std::endl;
using std::flush;
using std::size_t;
using std::string;
using std::vector;
using widders::Progress;
using widders::ProgressOptions;
using widders::ProgressStats;
using widders::color::ExtractSRGB;
using widders::color::LinearRGBToXYZ;
using widders::color::RenderABGR;
using widders::color::SRGBToLinearRGB;
using widders::color::XYZToLAB;
using widders::color::XYZToLUV;

constexpr size_t kColorDims = 3;
constexpr uint32_t kEmpty = ~0U;

constexpr int kAllColorImageWidth = 1U << 12U;
constexpr int kAllColorImageHeight = 1U << 12U;

struct Position {
  int32_t x;
  int32_t y;

  constexpr bool operator==(const Position& other) const = default;
  constexpr auto operator<=>(const Position& other) const = default;
  Position operator+(const Position& other) const {
    return {.x = x + other.x, .y = y + other.y};
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
using Color = widders::color::Color<Field>;
using ColorMap = widders::ScapegoatKdMap<Position, Color, widders::NoStatistic,
                                         std::ratio<6, 5>>;
using ColorPair = std::pair<uint32_t, Color>;
using ColorMetric = std::function<Color(uint32_t int_srgb)>;

// Sampling convolution; (offset, weight, placeable) tuples. Weights and colors
// of already-placed pixels are summed together for all pixels that are a
// placeable offset fromn already-placed pixel (so, on the frontier) and the
// resulting color value is the color that would most like to be placed there.
const std::tuple<Position, Field, bool> kSamplingOffsets[] = {
    {{.x = -1, .y = +0}, 1, true},                  //
    {{.x = -1, .y = -1}, 1 / std::sqrt(2), false},  //
    {{.x = +0, .y = -1}, 1, true},                  //
    {{.x = +1, .y = -1}, 1 / std::sqrt(2), false},  //
    {{.x = +1, .y = +0}, 1, true},                  //
    {{.x = +1, .y = +1}, 1 / std::sqrt(2), false},  //
    {{.x = +0, .y = +1}, 1, true},                  //
    {{.x = -1, .y = +1}, 1 / std::sqrt(2), false},  //
};

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
         Color srgb = ExtractSRGB<Field>(int_srgb);
         return srgb;
       }},
      {"rgb",
       [](uint32_t int_srgb) {
         Color srgb = ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = SRGBToLinearRGB(srgb);
         return linear_srgb;
       }},
      {"xyz",
       [](uint32_t int_srgb) {
         Color srgb = ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = SRGBToLinearRGB(srgb);
         Color xyz = LinearRGBToXYZ(linear_srgb);
         return xyz;
       }},
      {"lab",
       [](uint32_t int_srgb) {
         Color srgb = ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = SRGBToLinearRGB(srgb);
         Color xyz = LinearRGBToXYZ(linear_srgb);
         Color lab = XYZToLAB(xyz);
         return lab;
       }},
      {"luv", [](uint32_t int_srgb) {
         Color srgb = ExtractSRGB<Field>(int_srgb);
         Color linear_srgb = SRGBToLinearRGB(srgb);
         Color xyz = LinearRGBToXYZ(linear_srgb);
         Color luv = XYZToLUV(xyz);
         return luv;
       }}}[metric_name];
}

// Parses values for the 'ordering' flag and returns functions for producing the
// corresponding ordering of colors for placement.
std::variant<string, ColorOrder> ParseOrdering(const string& ordering_name) {
  const auto identity = [](uint32_t c) { return c; };
  enum OrderClass {
    UNKNOWN,
    RGB,
    LUV,
    XYZ,
  };
  constexpr std::string_view kOrderClassError =
      "Ordering channels must all come from the same colorspace (rgb, luv, "
      "xyz).";

  struct OrderPart {
    int channel = 0;
    bool descending = false;
  };

  if (ordering_name == "shuffle") {
    return ColorOrder{.map_channels = identity,
                      .should_shuffle = true,
                      .sort_metric = nullptr,
                      .do_sort = nullptr};
  } else {
    std::pair<string, string> split =
        absl::StrSplit(ordering_name, absl::MaxSplits(absl::ByChar(':'), 1));
    auto& [ordering_class, ordering_options] = split;
    if (ordering_options.empty()) {
      return "Expected 'shuffle' or 'class:options' for ordering.";
    } else if (ordering_class == "ordered") {
      vector<string> options =
          absl::StrSplit(ordering_options, absl::ByLength(2));
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
              return string(kOrderClassError);
            }
          }
          case 'g': {
            if (order_class == UNKNOWN || order_class == RGB) {
              order_class = RGB;
              part.channel = 1;
              break;
            } else {
              return string(kOrderClassError);
            }
          }
          case 'b': {
            if (order_class == UNKNOWN || order_class == RGB) {
              order_class = RGB;
              part.channel = 2;
              break;
            } else {
              return string(kOrderClassError);
            }
          }
          case 'l': {
            if (order_class == UNKNOWN || order_class == LUV) {
              order_class = LUV;
              part.channel = 0;
              break;
            } else {
              return string(kOrderClassError);
            }
          }
          case 'u': {
            if (order_class == UNKNOWN || order_class == LUV) {
              order_class = LUV;
              part.channel = 1;
              break;
            } else {
              return string(kOrderClassError);
            }
          }
          case 'v': {
            if (order_class == UNKNOWN || order_class == LUV) {
              order_class = LUV;
              part.channel = 2;
              break;
            } else {
              return string(kOrderClassError);
            }
          }
          case 'x': {
            if (order_class == UNKNOWN || order_class == XYZ) {
              order_class = XYZ;
              part.channel = 0;
              break;
            } else {
              return string(kOrderClassError);
            }
          }
          case 'y': {
            if (order_class == UNKNOWN || order_class == XYZ) {
              order_class = XYZ;
              part.channel = 1;
              break;
            } else {
              return string(kOrderClassError);
            }
          }
          case 'z': {
            if (order_class == UNKNOWN || order_class == XYZ) {
              order_class = XYZ;
              part.channel = 2;
              break;
            } else {
              return string(kOrderClassError);
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
            .map_channels =
                [ordering_parts](uint32_t input) {
                  uint32_t result = 0;
                  for (int i = 0; i < 3; ++i) {
                    const auto& part = ordering_parts[i];
                    uint32_t input_ch =
                        (input & (0xffU << (8U * i))) >> (8U * i);
                    result |= (part.descending ? 255U - input_ch : input_ch)
                              << (8 * part.channel);
                  }
                  return result;
                },
            .should_shuffle = false,
            .sort_metric = nullptr,
            .do_sort = nullptr};
      } else {
        // Some channels are not represented; shuffle the input colors first,
        // then sort.
        ColorMetric metric;
        switch (order_class) {
          case RGB: {
            metric = GetColorMetric("srgb");
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

        return ColorOrder{
            .map_channels = identity,
            .should_shuffle = true,
            .sort_metric = metric,
            .do_sort = [ordering_parts](vector<ColorPair>* colors) -> void {
              absl::c_stable_sort(*colors,
                                  [ordering_parts](const ColorPair& a,
                                                   const ColorPair& b) -> bool {
                                    for (const auto& part : ordering_parts) {
                                      Field a_chan = a.second[part.channel];
                                      Field b_chan = b.second[part.channel];
                                      if (a_chan == b_chan) continue;
                                      return (a_chan < b_chan) ^
                                             part.descending;
                                    }
                                    return false;  // Completely equal.
                                  });
            }};
      }
    } else {
      return "Unrecognized ordering class: " + ordering_class;
    }
  }
}

// Parses the origin location for the deposit process.
std::variant<string, Position> ParseOrigin(const string& origin_string,
                                           const ImageBuffer& image,
                                           absl::BitGen& rng) {
  if (origin_string == "center") {
    return Position{.x = image.width / 2, .y = image.height / 2};
  }
  if (origin_string == "random") {
    return Position{
        .x = absl::Uniform(absl::IntervalClosedOpen, rng, 0, image.width),
        .y = absl::Uniform(absl::IntervalClosedOpen, rng, 0, image.height),
    };
  }
  std::pair<string, string> coords =
      absl::StrSplit(origin_string, absl::MaxSplits(absl::ByChar(','), 1));
  auto& [x, y] = coords;
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
          .x = std::min(image.width - 1, x_exact),
          .y = std::min(image.height - 1, y_exact),
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

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage("Color deposit: growing images like crystals.");
  std::vector<char*> cmd_args = absl::ParseCommandLine(argc, argv);

  // TODO(widders): variations for all-color generation vs permuting existing
  //  images

  const ColorMetric metric = GetColorMetric(absl::GetFlag(FLAGS_color_metric));
  if (!metric) {
    cout << "Invalid color metric name" << endl;
    return 1;
  }
  const auto maybe_ordering = ParseOrdering(absl::GetFlag(FLAGS_ordering));
  if (std::holds_alternative<string>(maybe_ordering)) {
    cout << std::get<string>(maybe_ordering) << endl;
    return 1;
  }
  const auto& ordering = std::get<ColorOrder>(maybe_ordering);

  vector<ColorPair> color_values;
  color_values.reserve(kAllColorImageWidth * kAllColorImageHeight);

  cout << "Generating colors..." << flush;
  for (uint32_t int_srgb = 0; int_srgb < 0x1000000U; ++int_srgb) {
    uint32_t mapped = ordering.map_channels(int_srgb);
    color_values.emplace_back<ColorPair>({mapped, {}});
  }
  cout << "done" << endl;

  // RNG instance
  static absl::BitGen rng_inst;
  struct rng {
    auto operator()() { return rng_inst(); }
  };

  if (ordering.should_shuffle) {
    cout << "Shuffling..." << flush;
    std::shuffle(color_values.begin(), color_values.end(), rng_inst);
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
  auto maybe_origin = ParseOrigin(absl::GetFlag(FLAGS_origin), image, rng_inst);
  if (std::holds_alternative<string>(maybe_origin)) {
    cout << "Error: " << std::get<string>(maybe_origin) << endl;
    return 1;
  }
  auto origin = std::get<Position>(maybe_origin);

  cout << "Filling..." << flush;
  auto frontier = ColorMap();
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

  const bool stale_diagonals = absl::GetFlag(FLAGS_legacy_stale_diagonals);

  // Create initial frontier.
  frontier.set(origin, Color());
  // Repeatedly place a pixel in the image, updating the frontier set and all
  // affected color values of neighboring frontier pixels.
  const bool logging_enabled = absl::GetFlag(FLAGS_log_progress);
  for (size_t i = 0; i < image.size(); ++i) {
    if (logging_enabled) progress.update(i);
    using Distance = widders::EuclideanDistanceMetric<double>;
    auto search =
        frontier.search<widders::NearestRandomTiebreak, Distance, rng>(
            color_values[i].second);
#ifdef KD_TREE_DEBUG
#ifdef NDEBUG
#error NDEBUG disables compilation of k-d tree validation.
#endif  // NDEBUG
    frontier.validate();
#endif  // KD_TREE_DEBUG
    auto result = *search.result();
    const auto& [pos, val] = result;
    frontier.erase(pos);
    image[pos] = static_cast<uint32_t>(i);
    for (const auto& [f_delta, _, placeable] : kSamplingOffsets) {
      if (stale_diagonals && !placeable) continue;
      const Position frontier_pos = pos + f_delta;
      // Do nothing if this is not an empty frontier point.
      if (!image.InBounds(frontier_pos) || image[frontier_pos] != kEmpty) {
        continue;
      }
      // If frontier_pos is not placeable as a neighbor of the current pixel
      // and hasn't already been added to the frontier as a neighbor of another,
      // don't add it to the frontier set.
      if (!placeable && !frontier.contains(frontier_pos)) {
        continue;
      }
      // From this point, frontier_pos is either an existing or to-be-added
      // pixel in the frontier. Sum its surrounding points for the new value of
      // the color there.
      auto new_mean = Color();
      Field total_weight = 0;
      for (const auto& [s_delta, sample_weight, _] : kSamplingOffsets) {
        const Position sample_pos = frontier_pos + s_delta;
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
#endif  // KD_TREE_DEBUG
    }
  }
  auto elapsed = progress.elapsed();
  cout << "done; Total " << absl::Trunc(elapsed, absl::Seconds(0.1))
       << "; Average " << image.size() / absl::ToDoubleSeconds(elapsed)
       << " px/s" << endl;

  cout << "Unwrapping colors..." << flush;
  for (auto& pixel : image.buffer) {
    pixel = RenderABGR(color_values[pixel].first);
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
