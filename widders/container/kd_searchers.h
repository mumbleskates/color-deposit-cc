#ifndef WIDDERS_CONTAINER_KD_SEARCHERS_H_
#define WIDDERS_CONTAINER_KD_SEARCHERS_H_

#include <algorithm>
#include <compare>
#include <limits>
#include <optional>
#include <utility>

#include "widders/container/kd_value_traits.h"

namespace widders {

enum SearchDirectionFlags {
  LOOK_LEFT_FIRST = 0b0001,
  LOOK_RIGHT_FIRST = 0b0010,
  LOOK_BOTH = 0b0100,
  STOP = 0b1000,
};

enum class SearchDirection {
  SKIP /*                 */ = 0,
  LOOK_LEFT /*            */ = LOOK_LEFT_FIRST,
  LOOK_LEFT_FIRST_BOTH /* */ = LOOK_LEFT_FIRST | LOOK_BOTH,
  LOOK_RIGHT /*           */ = LOOK_RIGHT_FIRST,
  LOOK_RIGHT_FIRST_BOTH /**/ = LOOK_RIGHT_FIRST | LOOK_BOTH,
  STOP_SEARCH /*          */ = STOP,
};

template <typename T, typename Key, typename Value, typename Statistic>
concept Searcher = requires(T& t, Key k, Value v, Statistic s, size_t dim) {
                     t.visit(k, v);
                     {
                       t.guide_search(get_dimension(0, v), dim, s)
                     } -> std::convertible_to<SearchDirection>;
                     { t.recheck() } -> std::convertible_to<bool>;
                   };

template <typename Key, typename Value, typename Statistic,
          typename DistanceMetric>
class Nearest {
 public:
  using Distance = DistanceMetric::distance;

  Nearest(Value target, Distance max_distance)
      : target_(target), best_distance_(max_distance) {}
  explicit Nearest(Value target)
      : Nearest(target, std::max(std::numeric_limits<Distance>::max(),
                                 std::numeric_limits<Distance>::infinity())) {}

  void visit(Key key, Value value) {
    Distance this_distance = DistanceMetric::Distance(target_, value);
    if (this_distance < best_distance_) {
      best_distance_ = this_distance;
      best_key_ = key;
      best_value_ = value;
      found_ = true;
    }
  }

  SearchDirection guide_search(dimension_type<Value> plane, size_t axis,
                               Statistic) const {
    using enum SearchDirection;
    auto this_pos = get_dimension(axis, target_);
    auto ord = this_pos <=> plane;
    if (DistanceMetric::AxialDistance(this_pos - plane, axis) <=
        best_distance_) {
      if (ord < 0) {
        return LOOK_LEFT_FIRST_BOTH;
      } else {
        return LOOK_RIGHT_FIRST_BOTH;
      }
    } else {
      if (ord < 0) {
        return LOOK_LEFT;
      } else {
        return LOOK_RIGHT;
      }
    }
  }

  constexpr static bool recheck() { return true; }

  std::optional<std::pair<Key, Value>> result() const {
    if (found_) {
      return {{best_key_, best_value_}};
    } else {
      return std::nullopt;
    }
  }

 protected:
  Value target_;
  Distance best_distance_;
  Key best_key_ = {};
  Value best_value_ = {};
  bool found_ = false;
};

template <typename Key, typename Value, typename Statistic,
          typename DistanceMetric, typename Rng>
class NearestRandomTiebreak : Nearest<Key, Value, Statistic, DistanceMetric> {
 private:
  using Super = Nearest<Key, Value, Statistic, DistanceMetric>;

 public:
  using Distance = Super::Distance;

  NearestRandomTiebreak(Value target, Distance max_distance)
      : Super(target, max_distance) {}
  explicit NearestRandomTiebreak(Value target) : Super(target) {}

  void visit(Key key, Value value) {
    Distance this_distance = DistanceMetric::Distance(Super::target_, value);
    if (this_distance < Super::best_distance_) {
      Super::best_distance_ = this_distance;
      Super::best_key_ = key;
      Super::best_value_ = value;
      Super::found_ = true;
      tiebreak_ = std::nullopt;
    } else if (this_distance == Super::best_distance_) {
      if (!tiebreak_) tiebreak_ = Rng()();
      auto this_tiebreak = Rng()();
      if (this_tiebreak < tiebreak_) {
        Super::best_distance_ = this_distance;
        Super::best_key_ = key;
        Super::best_value_ = value;
        Super::found_ = true;
        tiebreak_ = this_tiebreak;
      }
    }
  }

  using Super::guide_search;
  using Super::recheck;
  using Super::result;

 protected:
  std::optional<std::invoke_result_t<decltype(std::declval<Rng>())>> tiebreak_;
};

}  // namespace widders

#endif  // WIDDERS_CONTAINER_KD_SEARCHERS_H_
