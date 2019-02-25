#ifndef WIDDERS_ALGORITHM_MEDIANS_H_
#define WIDDERS_ALGORITHM_MEDIANS_H_

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

namespace widders {
namespace detail {

const ptrdiff_t kMedianCutoff = 32;

template <typename Comp, typename Iterator>
inline Iterator _median3(Comp comp, Iterator a, Iterator b, Iterator c) {
  if (comp(*b, *a)) std::swap(b, a);
  if (comp(*c, *b)) std::swap(c, b);
  return comp(*b, *a) ? a : b;
}

template <typename Comp, typename Iterator>
inline Iterator _median4(Comp comp, Iterator a, Iterator b, Iterator c,
                         Iterator d) {
  if (comp(*b, *a)) std::swap(b, a);
  if (comp(*d, *c)) std::swap(d, c);
  if (comp(*c, *a)) std::swap(c, a);
  if (comp(*d, *b)) std::swap(d, b);
  return comp(*c, *b) ? c : b;
}

template <typename Comp, typename Iterator>
inline Iterator _median5(Comp comp, Iterator a, Iterator b, Iterator c,
                         Iterator d, Iterator e) {
  if (comp(*b, *a)) std::swap(b, a);
  if (comp(*d, *c)) std::swap(d, c);
  if (comp(*c, *a)) {
    std::swap(b, d);
    c = a;
  }
  if (comp(*b, *e)) std::swap(b, e);
  if (comp(*e, *c)) {
    std::swap(b, d);
    e = c;
  }
  return comp(*d, *e) ? d : e;
}

template <typename Comp, typename Iterator>
inline Iterator _medianle5(Comp comp, Iterator start, Iterator end) {
  switch (end - start) {
    case 5:
      return _median5(comp, start, start + 1, start + 2, start + 3, start + 4);
    case 4:
      return _median4(comp, start, start + 1, start + 2, start + 3);
    case 3:
      return _median3(comp, start, start + 1, start + 2);
    case 2:
    case 1:
      return start;
    default:
      assert(false);
      return {};
  }
}

}  // namespace detail

// TODO(widders): docs
template <ptrdiff_t exact_cutoff = detail::kMedianCutoff, typename Iterator,
          typename Comp = std::less<typename Iterator::value_type>>
Iterator MedianOfMedians(Iterator start, Iterator end, Comp comp = Comp()) {
  static_assert(exact_cutoff > 0,
                "cutoff for getting exact median must be positive");
  Iterator current_end = end;
  while (current_end - start > exact_cutoff) {
    Iterator new_end = start;
    Iterator sub_start;
    Iterator body_end = current_end - 5;
    for (sub_start = start; sub_start < body_end; sub_start += 5) {
      std::swap(*new_end,
                *detail::_median5(comp, sub_start, sub_start + 1, sub_start + 2,
                                  sub_start + 3, sub_start + 4));
      ++new_end;
    }
    std::swap(*new_end,
              *detail::_medianle5(comp, sub_start,
                                  std::min(current_end, sub_start + 5)));
    ++new_end;
    current_end = new_end;
  }
  Iterator middle = start + (current_end - start) / 2;
  std::nth_element(start, middle, current_end, comp);
  return middle;
}

// TODO(widders): docs
template <typename Iterator,
          typename Comp = std::less<typename Iterator::value_type>>
Iterator Partition(Iterator start, Iterator end, Iterator pivot,
                   Comp comp = Comp()) {
  const Iterator last = end - 1;
  std::swap(*pivot, *last);
  const auto& pivot_val = *last;
  auto boundary = std::partition(
      start, last, [&comp, &pivot_val](const typename Iterator::value_type& b) {
        return comp(b, pivot_val);
      });
  std::swap(*last, *boundary);
  return boundary;
}

// TODO(widders): docs
template <ptrdiff_t exact_cutoff = detail::kMedianCutoff, typename Iterator,
          typename Comp = std::less<typename Iterator::value_type>>
Iterator PartitionAtMedianOfMedians(Iterator start, Iterator end,
                                    Comp comp = Comp()) {
  static_assert(exact_cutoff > 0,
                "cutoff for getting exact median must be positive");
  if (end - start <= exact_cutoff) {
    Iterator middle = start + (end - start) / 2;
    std::nth_element(start, middle, end, comp);
    return middle;
  } else {
    return Partition(start, end,
                     MedianOfMedians<exact_cutoff>(start, end, comp), comp);
  }
}

}  // namespace widders

#endif  // WIDDERS_ALGORITHM_MEDIANS_H_
