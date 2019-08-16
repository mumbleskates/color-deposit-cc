#ifndef WIDDERS_CONTAINER_KD_METRICS_H_
#define WIDDERS_CONTAINER_KD_METRICS_H_

#include <algorithm>
#include <limits>
#include <utility>

namespace widders {

// Standard distance metrics.
//
// Distance metrics should have:
//  * a default constructor to a maximum distance value
//  * operator== and operator<
//  * an ImproveDistance<int n>(a, b, ...) function, taking n as the number of
//    dimensisons for a numeric indexable vector and returning a tri-value
//    Comparison describing whether the distance between the referenced points
//    is worse, better, or equal to the distance already stored. If the
//    distance is better, it is expected that the callee update its value to
//    reflect this better distance.
enum Comparison {
  BETTER = -1,
  TIED = 0,
  WORSE = 1,
};

template <typename NumericDistance>
struct EuclideanDistance {
  // Initialize with maximum distance.
  EuclideanDistance()
      : sqdistance(std::numeric_limits<NumericDistance>::max()) {}
  explicit EuclideanDistance(NumericDistance value)
      : sqdistance(std::move(value)) {}
  EuclideanDistance(const EuclideanDistance& copy_from) = default;
  EuclideanDistance(EuclideanDistance&& move_from) noexcept = default;
  EuclideanDistance& operator=(const EuclideanDistance& assn) = default;

  template <size_t dims, typename Pt, typename... ExtraArgs>
  Comparison ImproveDistance(const Pt& a, const Pt& b,
                             const ExtraArgs&... args) {
    NumericDistance sum = 0;
    for (size_t i = 0; i < dims; ++i) {
      NumericDistance diff = a[i] - b[i];
      sum += diff * diff;
    }
    if (sum < sqdistance) {
      sqdistance = sum;
      return BETTER;
    } else if (sum == sqdistance) {
      return TIED;
    } else {
      return WORSE;
    }
  }

  bool operator==(const EuclideanDistance& other) {
    return sqdistance == other.sqdistance;
  }

  bool operator<(const EuclideanDistance& other) {
    return sqdistance < other.sqdistance;
  }

  template <typename DimensionDist>
  bool IntersectsPlane(DimensionDist distance_to_plane, size_t dim) {
    return sqdistance >= distance_to_plane * distance_to_plane;
  }

  NumericDistance sqdistance;
};

template <typename NumericDistance>
struct ManhattanDistance {
  // Initialize with maximum distance.
  ManhattanDistance()
      : sumdistance(std::numeric_limits<NumericDistance>::max()) {}
  explicit ManhattanDistance(NumericDistance value)
      : sumdistance(std::move(value)) {}
  ManhattanDistance(const ManhattanDistance& copy_from) = default;
  ManhattanDistance(ManhattanDistance&& move_from) noexcept = default;
  ManhattanDistance& operator=(const ManhattanDistance& assn) = default;

  template <size_t dims, typename Pt, typename... ExtraArgs>
  Comparison ImproveDistance(const Pt& a, const Pt& b,
                             const ExtraArgs&... args) {
    NumericDistance sum = 0;
    for (size_t i = 0; i < dims; ++i) {
      NumericDistance diff = a[i] - b[i];
      sum += std::abs(diff);
    }
    if (sum < sumdistance) {
      sumdistance = sum;
      return BETTER;
    } else if (sum == sumdistance) {
      return TIED;
    } else {
      return WORSE;
    }
  }

  bool operator==(const ManhattanDistance& other) {
    return sumdistance == other.sumdistance;
  }

  bool operator<(const ManhattanDistance& other) {
    return sumdistance < other.sumdistance;
  }

  template <typename DimensionDist>
  bool IntersectsPlane(DimensionDist distance_to_plane, size_t dim) {
    return sumdistance >= std::abs(distance_to_plane);
  }

  NumericDistance sumdistance;
};

template <typename NumericDistance>
struct ChebyshevDistance {
  // Initialize with maximum distance.
  ChebyshevDistance()
      : maxdistance(std::numeric_limits<NumericDistance>::max()) {}
  explicit ChebyshevDistance(NumericDistance value)
      : maxdistance(std::move(value)) {}
  ChebyshevDistance(const ChebyshevDistance& copy_from) = default;
  ChebyshevDistance(ChebyshevDistance&& move_from) noexcept = default;
  ChebyshevDistance& operator=(const ChebyshevDistance& assn) = default;

  template <size_t dims, typename Pt, typename... ExtraArgs>
  Comparison ImproveDistance(const Pt& a, const Pt& b,
                             const ExtraArgs&... args) {
    NumericDistance max = 0;
    for (size_t i = 0; i < dims; ++i) {
      NumericDistance diff = a[i] - b[i];
      max = std::max(max, std::abs(diff));
    }
    if (max < maxdistance) {
      maxdistance = max;
      return BETTER;
    } else if (max == maxdistance) {
      return TIED;
    } else {
      return WORSE;
    }
  }

  bool operator==(const ChebyshevDistance& other) {
    return maxdistance == other.maxdistance;
  }

  bool operator<(const ChebyshevDistance& other) {
    return maxdistance < other.maxdistance;
  }

  template <typename DimensionDist>
  bool IntersectsPlane(DimensionDist distance_to_plane, size_t dim) {
    return maxdistance >= std::abs(distance_to_plane);
  }

  NumericDistance maxdistance;
};

// A wrapper type for other distance types which can integrate an arbitrary
// tiebreaking function.
//
// The Tiebreaker type provided should have operator== and operator< and will be
// constructed with the arguments passed to ImproveDistance after points a and
// b. In ScapegoatKdMap, this will be a reference to the key of the entry we are
// measuring distance to.
template <typename OtherDistanceType, typename Tiebreak>
struct TiebreakingDistance {
  TiebreakingDistance() = default;
  TiebreakingDistance(OtherDistanceType dist, Tiebreak tb)
      : distance(std::move(dist)), tiebreak(std::move(tb)) {}
  TiebreakingDistance(const TiebreakingDistance& copy_from) = default;
  TiebreakingDistance(TiebreakingDistance&& move_from) noexcept = default;
  TiebreakingDistance& operator=(const TiebreakingDistance& assn) = default;

  template <size_t dims, typename Pt, typename... TiebreakArgs>
  Comparison ImproveDistance(const Pt& a, const Pt& b,
                             const TiebreakArgs&... args) {
    Comparison inner_result =
        distance.template ImproveDistance<dims>(a, b, &args...);
    if (inner_result == BETTER) {
      tiebreak = Tiebreak(&args...);
      return BETTER;
    } else if (inner_result == TIED) {
      Tiebreak new_tiebreak = Tiebreak(&args...);
      if (new_tiebreak < tiebreak) {
        tiebreak = new_tiebreak;
        return BETTER;
      } else if (tiebreak == new_tiebreak) {
        return TIED;
      } else {
        return WORSE;
      }
    } else {
      return WORSE;
    }
  }

  bool operator==(const TiebreakingDistance& other) {
    return distance == other.distance && tiebreak == other.tiebreak;
  }

  bool operator<(const TiebreakingDistance& other) {
    return distance == other.distance ? tiebreak < other.tiebreak
                                      : distance < other.distance;
  }

  template <typename DimensionDist>
  bool IntersectsPlane(DimensionDist distance_to_plane, size_t dim) {
    return distance.IntersectsPlane(distance_to_plane, dim);
  }

  OtherDistanceType distance;
  Tiebreak tiebreak;
};

template <typename Rng>
struct RandomTiebreak {
  RandomTiebreak() : val() {}
  template <typename... Args>
  explicit RandomTiebreak(Args... args) : val(Rng::Value()) {}
  RandomTiebreak(const RandomTiebreak& copy_from) = default;
  RandomTiebreak(RandomTiebreak&& move_from) noexcept = default;
  RandomTiebreak& operator=(const RandomTiebreak& assn) = default;

  bool operator==(const RandomTiebreak& other) { return val == other.val; }
  bool operator<(const RandomTiebreak& other) { return val < other.val; }

  typename Rng::type val;
};

}  // namespace widders

#endif  // WIDDERS_CONTAINER_KD_METRICS_H_
