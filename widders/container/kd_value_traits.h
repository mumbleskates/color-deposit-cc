#ifndef WIDDERS_CONTAINER_KD_VALUE_TRAITS_H_
#define WIDDERS_CONTAINER_KD_VALUE_TRAITS_H_

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <utility>

using ::std::size_t;

namespace widders {
namespace detail {

template <typename V>
constexpr size_t dimensions_of()
  requires requires { std::tuple_size_v<V>; }
{
  return std::tuple_size_v<V>;
}

template <typename V>
constexpr size_t dimensions_of()
  requires requires {
             { V::dimensions } -> std::same_as<size_t>;
           }
{
  return V::dimensions;
}

template <size_t this_dim, typename... Ds>
constexpr std::variant<Ds...> get_dimension_tuple_impl(
    size_t dim, const std::tuple<Ds...>& v) {
  if constexpr (this_dim == sizeof...(Ds)) {
    return {};
  } else {
    if (dim == this_dim) {
      return std::variant<Ds...>(std::in_place_index<this_dim>,
                                 std::get<this_dim>(v));
    } else {
      return get_dimension_tuple_impl<this_dim + 1, Ds...>(dim, v);
    }
  }
}

}  // namespace detail

template <typename V>
inline constexpr size_t dimensions_of = detail::dimensions_of<V>();

template <typename D, size_t size>
constexpr D get_dimension(size_t dim, const std::array<D, size>& v) {
  return v[dim];
}

template <typename... Ds>
constexpr std::variant<Ds...> get_dimension(size_t dim,
                                            const std::tuple<Ds...>& v) {
  return detail::get_dimension_tuple_impl<0, Ds...>(dim, v);
}

template <typename V>
constexpr auto get_dimension(size_t dim, const V& v) {
  return v.get_dimension(dim);
}

template <typename V>
using dimension_type =
    std::remove_cv_t<decltype(get_dimension(0, std::declval<V>()))>;

static_assert(dimensions_of<std::tuple<int, long, char>> == 3);
static_assert(dimensions_of<std::array<int, 3>> == 3);

static_assert(get_dimension(0, std::array<int, 3>{1, 2, 3}) == 1);
static_assert(get_dimension(0, std::tuple<int, long>(1, 2)) ==
              std::variant<int, long>((int)1));
static_assert(get_dimension(1, std::tuple<int, long>(1, 2)) ==
              std::variant<int, long>((long)2));

}  // namespace widders
#endif  // WIDDERS_CONTAINER_KD_VALUE_TRAITS_H_
