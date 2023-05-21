#ifndef WIDDERS_CONTAINER_KD_METRICS_H_
#define WIDDERS_CONTAINER_KD_METRICS_H_

#include <algorithm>
#include <compare>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "widders/container/kd_value_traits.h"

namespace widders {

using ::std::size_t;

// Standard distance metrics.
//
// Distance metrics should have:
//  * a Distance<int n>(a, b, ...) function, taking n as the number of
//    dimensisons for a numeric indexable vector and returning the distance
//    between them
//  * an AxialDistance(d, dim) function, taking a distance along a single
//    dimension axis and returning the representation of that distance

template <typename NumericDist>
struct EuclideanDistanceMetric {
  using distance = NumericDist;

  EuclideanDistanceMetric() = delete;

  template <typename Value>
  constexpr static NumericDist Distance(const Value& a, const Value& b) {
    NumericDist sum = 0;
    for (size_t i = 0; i < dimensions_of<Value>; ++i) {
      NumericDist diff = get_dimension(i, a) - get_dimension(i, b);
      sum += diff * diff;
    }
    return sum;
  }

  constexpr static NumericDist AxialDistance(NumericDist axial_difference,
                                             size_t dim) {
    return axial_difference * axial_difference;
  }
};

template <typename NumericDist>
struct ManhattanDistanceMetric {
  ManhattanDistanceMetric() = delete;

  template <typename Value>
  constexpr static NumericDist Distance(const Value& a, const Value& b) {
    NumericDist sum = 0;
    for (size_t i = 0; i < dimensions_of<Value>; ++i) {
      NumericDist diff = get_dimension(i, a) - get_dimension(i, b);
      sum += std::abs(diff);
    }
    return sum;
  }

  constexpr static NumericDist AxialDistance(NumericDist axial_difference,
                                             size_t dim) {
    return std::abs(axial_difference);
  }
};

template <typename NumericDist>
struct ChebyshevDistanceMetric {
  ChebyshevDistanceMetric() = delete;

  template <typename Value>
  constexpr static NumericDist Distance(const Value& a, const Value& b) {
    NumericDist max = 0;
    for (size_t i = 0; i < dimensions_of<Value>; ++i) {
      NumericDist diff = get_dimension(i, a) - get_dimension(i, b);
      max = std::max(max, std::abs(diff));
    }
    return max;
  }

  constexpr static NumericDist AxialDistance(NumericDist axial_difference,
                                             size_t dim) {
    return std::abs(axial_difference);
  }
};

}  // namespace widders

#endif  // WIDDERS_CONTAINER_KD_METRICS_H_
