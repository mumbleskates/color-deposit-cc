#ifndef WIDDERS_CONTAINER_SCAPEGOAT_KD_MAP_H_
#define WIDDERS_CONTAINER_SCAPEGOAT_KD_MAP_H_

#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "widders/algorithm/medians.h"

namespace widders {

template <typename NodeType>
struct MedianOfMediansPolicy {
  using VecIter = typename std::vector<std::unique_ptr<NodeType>>::iterator;
  static VecIter PartitionAtMedian(VecIter start, VecIter end, size_t dim) {
    return widders::PartitionAtMedianOfMedians(
        start, end,
        [dim](const std::unique_ptr<NodeType>& a,
              const std::unique_ptr<NodeType>& b) -> bool {
          // Compare first on the point at dim; tiebreak with pointer value
          return a->val[dim] < b->val[dim] ||
                 (a->val[dim] == b->val[dim] && a < b);
        });
  }
  constexpr static bool kBalanceGuaranteed = false;
};

template <typename NodeType>
struct ExactMedianPolicy {
  using VecIter = typename std::vector<std::unique_ptr<NodeType>>::iterator;
  static VecIter PartitionAtMedian(VecIter start, VecIter end, size_t dim) {
    auto middle = start + (end - start) / 2;
    std::nth_element(start, middle, end,
                     [dim](const std::unique_ptr<NodeType>& a,
                           const std::unique_ptr<NodeType>& b) -> bool {
                       // Compare first on the point at dim; tiebreak with
                       // pointer value
                       return a->val[dim] < b->val[dim] ||
                              (a->val[dim] == b->val[dim] && a < b);
                     });
    return middle;
  }
  constexpr static bool kBalanceGuaranteed = true;
};

// Standard distance metrics.
//
// Distance metrics should have:
//  * a default constructor to a maximum distance value
//  * operator== and operator<
//  * an ImproveDistance<int n>(a, b, ...) function, taking n as the number of
//    dimensisons for a numeric indexable vector and returning a tri-value
//    Comparison describing whether the distance between the referenced points
//    is greater, lesser, or equal to the distance already stored. If the
//    distance is lesser, it is expected that the callee update its value to
//    reflect this lesser distance.
enum Comparison {
  LESSER = -1,
  TIED = 0,
  GREATER = 1,
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
  Comparison ImproveDistance(const Pt& a, const Pt& b, ExtraArgs... args) {
    NumericDistance sum = 0;
    for (size_t i = 0; i < dims; ++i) {
      NumericDistance diff = a[i] - b[i];
      sum += diff * diff;
    }
    if (sum < sqdistance) {
      sqdistance = sum;
      return LESSER;
    } else if (sum == sqdistance) {
      return TIED;
    } else {
      return GREATER;
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
  Comparison ImproveDistance(const Pt& a, const Pt& b, ExtraArgs... args) {
    NumericDistance sum = 0;
    for (size_t i = 0; i < dims; ++i) {
      auto diff = a[i] - b[i];
      sum += std::abs(diff);
    }
    if (sum < sumdistance) {
      sumdistance = sum;
      return LESSER;
    } else if (sum == sumdistance) {
      return TIED;
    } else {
      return GREATER;
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
  Comparison ImproveDistance(const Pt& a, const Pt& b, ExtraArgs... args) {
    NumericDistance max = 0;
    for (size_t i = 0; i < dims; ++i) {
      auto diff = a[i] - b[i];
      max = std::max(max, std::abs(diff));
    }
    if (max < maxdistance) {
      maxdistance = max;
      return LESSER;
    } else if (max == maxdistance) {
      return TIED;
    } else {
      return GREATER;
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
  Comparison ImproveDistance(const Pt& a, const Pt& b, TiebreakArgs... args) {
    Comparison inner_result =
        distance.template ImproveDistance<dims>(a, b, &args...);
    if (inner_result == LESSER) {
      tiebreak = Tiebreak(&args...);
      return LESSER;
    } else if (inner_result == TIED) {
      Tiebreak new_tiebreak = Tiebreak(&args...);
      if (new_tiebreak < tiebreak) {
        tiebreak = new_tiebreak;
        return LESSER;
      } else if (tiebreak == new_tiebreak) {
        return TIED;
      } else {
        return GREATER;
      }
    } else {
      return GREATER;
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

// --------------------------------------------------------------------------

// TODO(widders): doc
template <size_t dims, typename Dimension, typename Key,
          template <typename> class MedianPolicy = MedianOfMediansPolicy>
class ScapegoatKdMap {
  static_assert(dims > 0, "k-d tree with less than 1 dimension");

  struct KdNode;

  // TODO(widders): maybe memoize
  inline bool tree_is_balanced(size_t height, size_t node_count) const {
    // 1 + (implicit) floor of (log2 of ^ tree node count multiplied by the
    //                               height balance we maintain)
    return height <= 1 + static_cast<uint32_t>(std::log2f(node_count) *
                                               max_height_factor_);
  }

 public:
  using Point = std::array<Dimension, dims>;

  // Struct for results of a tree search.
  template <typename DistanceType>
  struct NearestResult {
    NearestResult() = default;
    explicit NearestResult(const DistanceType& dist) : distance(dist) {}

    bool operator<(const NearestResult& other) const {
      return distance < other.distance;
    };

    Key key = {};
    const Point* val = nullptr;
    DistanceType distance = {};
#ifdef KD_SEARCH_STATS
    size_t nodes_searched = 0;
#endif
  };

  explicit ScapegoatKdMap(float balance_factor = 1.5)
      : max_height_factor_(balance_factor) {
    // 1.0 represents the minimum possible height of a binary tree of
    // a given size.
    // Values less than 1.0 will result in a highly unbalanced tree that
    // continually rebuilds only its tiniest ends; values less than
    // log(5)/log(3) ~= 1.465 are not guaranteed to behave well either,
    // in the worst case, as the median-of-medians that chooses the pivot
    // for rebuilding a subtree does not have guarantees better than this.
    assert(max_height_factor_ >= 1.0);
  }
  // Move constructor
  ScapegoatKdMap(ScapegoatKdMap&& move_from) noexcept = default;
  // TODO(widders): Add constructor(s) for preexisting value sets all at once
  // Disable copy construction for now
  ScapegoatKdMap(ScapegoatKdMap& copy_from) = delete;

  // Return the number of items in the tree.
  size_t size() const { return items_.size(); }
  bool empty() const { return size() == 0; }

  // TODO(widders): iterators
  //  const_iterator begin() const { return const_iterator(items_.begin()); }
  //  const_iterator end() const { return const_iterator(items_.end()); }
  //  const_iterator find(const Key& key) const {
  //    return const_iterator(items_.find(key));
  //  }

  // Return the point value of the given key.
  // Raises if the key does not exist in the tree.
  // Takes O(1) time.
  const Point& get(const Key& key) const { return items_.at(key)->val; }
  const Point& operator[](const Key& key) const { return get(key); }
  // Returns true if the key is in the tree, false otherwise.
  // Takes O(1) time.
  bool contains(const Key& key) const { return items_.count(key) > 0; }
  // Return the height of the tree: the number of nodes that must be
  // traversed to reach the deepest node in the tree.
  size_t height() const { return head_->maxdepth; }

  // Set the point value of the given key.
  // If the key already exists in the tree, its value will be changed.
  // Takes O(height) time.
  void set(const Key& key, Point val) {
    auto item_found = items_.find(key);
    if (item_found != items_.end()) {
      // The key already exists. Re-insert its node with the new value.
      std::unique_ptr<KdNode> reinsert_node = tree_pop_node(item_found->second);
      reinsert_node->val = std::move(val);
      // Reinsert the removed node with our new value.
      item_found->second = reinsert_node.get();
      tree_insert_node(std::move(reinsert_node));
    } else {
      auto new_node = std::make_unique<KdNode>(key, std::move(val));
      items_.insert({key, new_node.get()});
      tree_insert_node(std::move(new_node));
    }
  }

  // Remove the given key from the tree, returning 1 if it existed
  // (0 otherwise).
  // Takes O(height) time.
  size_t erase(const Key& key) {
    const auto found_item = items_.find(key);
    if (found_item != items_.end()) {
      // Remove the item from the hash table.
      items_.erase(found_item);
      // Pop the node out of the tree and delete it.
      std::unique_ptr<KdNode> removed_node = tree_pop_node(found_item->second);
      return 1;
    } else {
      return 0;
    }
  }

  template <typename DistanceType = EuclideanDistance<long double>>
  NearestResult<DistanceType> nearest(const Point& val,
                                      DistanceType max_distance = {}) const {
    NearestResult<DistanceType> result(max_distance);
    tree_search(val, *head_, 0, result);
    return result;
  }

  template <typename DistanceType = EuclideanDistance<long double>>
  std::vector<NearestResult<DistanceType>> nearest_n(
      const Point& val, size_t n, DistanceType max_distance = {}) const {
    std::vector<NearestResult<DistanceType>> result(n, {max_distance});
    tree_search_n(val, *head_, 0, result);
    absl::c_sort_heap(result);
    // Find out how many items we actually located, eliminating all those which
    // are still empty at the end of our found set.
    size_t i = n;
    while (!result[i - 1].val) --i;
    result.resize(i);
    return result;
  }

  void validate() const {
#ifndef NDEBUG
    size_t tree_nodes;
    if (empty()) {
      assert(!head_);
      tree_nodes = 0;
    } else {
      assert(!head_->parent);
      Point lower, upper;
      Dimension low, high;
      if (std::numeric_limits<Dimension>::has_infinity) {
        low = -std::numeric_limits<Dimension>::infinity();
        high = std::numeric_limits<Dimension>::infinity();
      } else {
        low = std::numeric_limits<Dimension>::lowest();
        high = std::numeric_limits<Dimension>::max();
      }
      for (auto& d : lower) d = low;
      for (auto& d : upper) d = high;
      tree_nodes = validate_node(*head_, 0, lower, upper);
    }
    assert(tree_nodes == items_.size());
    if (MedianPolicy<KdNode>::kBalanceGuaranteed) {
      assert(!head_ || tree_is_balanced(head_->maxdepth, size()));
    }
#endif
  }

 private:
  using VecIter = typename std::vector<std::unique_ptr<KdNode>>::iterator;

  struct KdNode {
    KdNode* parent = nullptr;
    std::unique_ptr<KdNode> left = nullptr;
    std::unique_ptr<KdNode> right = nullptr;
    Key key = {};
    Point val = {};
    Dimension mid = {};
    // Absolute depth of the deepest leaf in this subtree, including root.
    // Zero if the tree is empty. Form a max-statistic tree on this field.
    uint32_t maxdepth = 0;

    KdNode(Key k, Point v) : key(k), val(v) {}
    KdNode() = default;
    ~KdNode() = default;

    // Update the stats of this node and its ancestors.
    // TODO(widders): consider queueing and deferring these updates?
    void revise_stats() {
      KdNode* current = this;
      while (current) {
        auto new_maxdepth =
            1 + std::max(current->left ? current->left->maxdepth : 0,
                         current->right ? current->right->maxdepth : 0);
        if (current->maxdepth == new_maxdepth) {
          return;
        } else {
          current->maxdepth = new_maxdepth;
          current = current->parent;
          continue;
        }
      }
    }
  };

  size_t validate_node(const KdNode& node, const size_t dim, const Point& lower,
                       const Point& upper) const {
    // Check key reference is correct
    assert(items_.find(node.key)->second == &node);
    // Check val is inside the bounds
    for (size_t i = 0; i < dims; ++i) {
      assert(node.val[i] >= lower[i]);
      assert(node.val[i] <= upper[i]);
    }
    // Check mid is inside the bounds
    assert(node.mid >= lower[dim]);
    assert(node.mid <= upper[dim]);
    // Check maxdepth is correct
    assert(node.maxdepth ==
           1 + std::max((node.left ? node.left->maxdepth : 0),
                        (node.right ? node.right->maxdepth : 0)));
    // Check child parent pointers
    if (node.left) assert(node.left->parent == &node);
    if (node.right) assert(node.right->parent == &node);
    // Recurse
    const size_t next_dim = dim == dims - 1 ? 0 : dim + 1;
    size_t children = 0;
    if (node.left) {
      Point new_upper = upper;
      new_upper[dim] = node.mid;
      children += validate_node(*node.left, next_dim, lower, new_upper);
    }
    if (node.right) {
      Point new_lower = lower;
      new_lower[dim] = node.mid;
      children += validate_node(*node.right, next_dim, new_lower, upper);
    }
    return 1 + children;
  }

  // Check each subtree starting from the given node, all the way up
  // the tree looking for the smallest subtree that is unbalanced, then
  // rebuild it.
  void rebuild_one_ancestor(KdNode* tree, size_t dim) {
    size_t node_count = count_nodes(tree);
    while (true) {
      if (tree_is_balanced(tree->maxdepth, node_count)) {
        // Subtree is sufficiently balanced; continue checking ancestors.
        auto parent = tree->parent;
        if (!parent) return;
        // Add the nodes from the parent and sibling to node_count.
        if (parent->left.get() == tree) {
          node_count += 1 + count_nodes(parent->right.get());
        } else {
          node_count += 1 + count_nodes(parent->left.get());
        }
        tree = parent;
        // Keep dim in sync with tree's current level.
        dim = dim == 0 ? dims - 1 : dim - 1;
        continue;
      } else {
        // Subtree is too tall: rebalance.
        std::unique_ptr<KdNode>* tree_root;
        if (tree->parent) {
          KdNode* const parent = tree->parent;
          if (parent->left.get() == tree) {
            tree_root = &parent->left;
          } else {
            tree_root = &parent->right;
          }
        } else {
          tree_root = &head_;
        }
        rebuild(*tree_root, node_count, dim);
        return;
      }
    }
  }

  void rebuild(std::unique_ptr<KdNode>& tree_root, size_t node_count,
               size_t dim) {
    KdNode* const parent = tree_root->parent;
    std::vector<std::unique_ptr<KdNode>> nodes;
    nodes.reserve(node_count);
    collect_nodes(std::move(tree_root), nodes);
    tree_root = std::move(rebuild_recursive(dim, nodes.begin(), nodes.end()));
    tree_root->parent = parent;
    parent->revise_stats();
  }

  static size_t count_nodes(KdNode* tree) {
    return tree ? 1 + count_nodes(tree->left.get()) +
                      count_nodes(tree->right.get())
                : 0;
  }

  static void collect_nodes(std::unique_ptr<KdNode> node,
                            std::vector<std::unique_ptr<KdNode>>& vec) {
    KdNode& node_ref = *node;
    if (node_ref.left) collect_nodes(std::move(node_ref.left), vec);
    vec.push_back(std::move(node));
    if (node_ref.right) collect_nodes(std::move(node_ref.right), vec);
  }

  static std::unique_ptr<KdNode> rebuild_recursive(size_t dim, VecIter start,
                                                   VecIter end) {
    const size_t next_dim = dim == dims - 1 ? 0 : dim + 1;
    const VecIter pivot =
        MedianPolicy<KdNode>::PartitionAtMedian(start, end, dim);
    std::unique_ptr<KdNode> node = std::move(*pivot);
    uint32_t left_depth, right_depth;
    if (start == pivot) {
      left_depth = 0;
    } else {
      std::unique_ptr<KdNode> left_child =
          rebuild_recursive(next_dim, start, pivot);
      left_child->parent = node.get();
      left_depth = left_child->maxdepth;
      node->left = std::move(left_child);
    }
    if (pivot + 1 == end) {
      right_depth = 0;
    } else {
      std::unique_ptr<KdNode> right_child =
          rebuild_recursive(next_dim, pivot + 1, end);
      right_child->parent = node.get();
      right_depth = right_child->maxdepth;
      node->right = std::move(right_child);
    }
    // Fix node's metadata
    node->maxdepth = 1 + std::max(left_depth, right_depth);
    node->mid = node->val[dim];
    return std::move(node);
  }

  // Given a pointer to a node, removes that node's value from the tree.
  // Some node is returned, along with its ownership.
  //
  // The returned node may not be the same as the node that was passed, but
  // it will contain the key and value of the passed node. The returned node's
  // parent pointer and mid are unspecified.
  //
  // No updates are made to the hash table pointer for the key being removed
  // (the node that is passed in).
  std::unique_ptr<KdNode> tree_pop_node(KdNode* const node) {
    KdNode* current = node;
    // Traverse down the deeper branch until we reach a leaf.
    while (current->left || current->right) {
      const auto left_depth = current->left ? current->left->maxdepth : 0;
      if (current->right && current->right->maxdepth >= left_depth) {
        // Right subtree exists and is at least as deep as the left.
        current = current->right.get();
      } else {
        // Left subtree is deeper.
        current = current->left.get();
      }
    }
    // Current is now the deepest node from this subtree. Detach it from the
    // tree into 'popped'.
    // Will equal 'current' when we take back its ownership.
    std::unique_ptr<KdNode> popped;
    // This is the parent of the node object that's being popped from the tree.
    KdNode* const popped_parent = current->parent;
    if (popped_parent) {
      // Detach it from its parent and adjust tree depths above.
      if (popped_parent->left.get() == current) {
        popped = std::move(popped_parent->left);
      } else {
        popped = std::move(popped_parent->right);
      }
      popped_parent->revise_stats();
    } else {
      // We are popping the head of the tree.
      popped = std::move(head_);
    }
    // Popped is now the node that will be returned to the caller, but if it is
    // not the same node that was passed to us in the first place we need to
    // make sure that its key & value are preserved in the tree. So, before we
    // return it, we swap key/value in the nodes and update the hash table for
    // the key of the popped node to point to the node we swapped that value
    // into. The key & value of the node that were originally passed to us then
    // end up in the node that is popped off and returned.
    if (popped.get() != node) {
      // Rectify the hash table pointer to the node that is staying.
      items_.find(popped->key)->second = node;
      // Swap the popped node's key and value into that same staying node.
      std::swap(node->key, popped->key);
      std::swap(node->val, popped->val);
    }

    // Check for tree balance.
    if (head_ && !tree_is_balanced(head_->maxdepth, size())) {
      // Current is still the leaf node that we removed.
      // Traverse upwards counting through dim so we can discover what dimension
      // is the discriminant at popped_parent before rebalancing.
      size_t dim = 0;
      for (KdNode* n = popped_parent->parent; n; n = n->parent) {
        dim = dim == dims - 1 ? 0 : dim + 1;
      }
      // dim is now the discriminant dim of popped_parent.
      rebuild_one_ancestor(popped_parent, dim);
    }
    return std::move(popped);
  }

  // Insert the given node into the tree.
  void tree_insert_node(std::unique_ptr<KdNode> node) {
    // Leaf nodes always have depth 1, and this node will be a leaf.
    node->maxdepth = 1;
    if (!head_) {
      node->mid = node->val[0];
      node->parent = nullptr;
      head_ = std::move(node);
    } else {
      const Point& val = node->val;
      size_t dim = 0;
      KdNode* current = head_.get();
      std::unique_ptr<KdNode>* destination;  // This is where we'll insert.
      while (true) {
        // Order by discriminant, or pointer when discriminant is equal
        if (val[dim] == current->mid ? node.get() < current
                                     : val[dim] < current->mid) {
          // val is to the left of the splitting plane.
          if (current->left) {
            // Traverse down to the child.
            current = current->left.get();
          } else {
            // Insert node as the empty left slot here.
            destination = &current->left;
            break;
          }
        } else {
          // val is to the right of the splitting plane.
          if (current->right) {
            // Traverse down to the child.
            current = current->right.get();
          } else {
            // Insert node as the right empty slot here.
            destination = &current->right;
            break;
          }
        }
        // Each time we traverse down, cycle through the dimensions of
        // the point value.
        dim++;
        if (dim == dims) dim = 0;
      }
      // Finish inserting node under current.
      const size_t next_dim = dim == dims - 1 ? 0 : dim + 1;
      node->mid = node->val[next_dim];
      node->parent = current;
      *destination = std::move(node);
      current->revise_stats();
      // Rebuild parts of the tree if necessary
      if (!tree_is_balanced(head_->maxdepth, size()))
        rebuild_one_ancestor(current, dim);
    }
  }

  // Search the tree to find the nearest key/value to the given point.
  template <typename DistanceType>
  void tree_search(const Point& val, const KdNode& node, const size_t dim,
                   NearestResult<DistanceType>& result) const {
#ifdef KD_SEARCH_STATS
    result.nodes_searched++;
#endif
    // Update best result.
    if (result.distance.template ImproveDistance<dims>(val, node.val,
                                                       node.key) == LESSER) {
      result.key = node.key;
      result.val = &node.val;
    }
    // Traverse downwards.
    const size_t next_dim = dim == dims - 1 ? 0 : dim + 1;
    auto distance_to_plane = node.mid - val[dim];
    if (val[dim] < node.mid) {  // val is left of the splitting plane.
      if (node.left) {
        tree_search(val, *node.left, next_dim, result);
      }
      // Traverse to the other side if still needed.
      if (node.right &&
          result.distance.IntersectsPlane(distance_to_plane, dim)) {
        tree_search(val, *node.right, next_dim, result);
      }
    } else {  // val is right of the splitting plane.
      if (node.right) {
        tree_search(val, *node.right, next_dim, result);
      }
      // Traverse to the other side if still needed.
      if (node.left &&
          result.distance.IntersectsPlane(distance_to_plane, dim)) {
        tree_search(val, *node.left, next_dim, result);
      }
    }
  }

  // Search the tree to find the nearest key/value to the given point.
  template <typename DistanceType>
  void tree_search_n(const Point& val, const KdNode& node, const size_t dim,
                     std::vector<NearestResult<DistanceType>>& result) const {
    NearestResult<DistanceType>& nth_best = *result.begin();
    // Update best result.
    if (nth_best.distance.template ImproveDistance<dims>(val, node.val,
                                                         node.key) == LESSER) {
      nth_best.key = node.key;
      nth_best.val = &node.val;
      // By replacing values in the root of the heap, then popping & pushing,
      // that item is swapped to the back and then reinserted.
      absl::c_pop_heap(result);
      absl::c_push_heap(result);
    }

    // Traverse downwards.
    const size_t next_dim = dim == dims - 1 ? 0 : dim + 1;
    auto distance_to_plane = node.mid - val[dim];
    if (val[dim] < node.mid) {  // val is left of the splitting plane.
      if (node.left) {
        tree_search_n(val, *node.left, next_dim, result);
      }
      // Traverse to the other side if still needed.
      if (node.right &&
          nth_best.distance.IntersectsPlane(distance_to_plane, dim)) {
        tree_search_n(val, *node.right, next_dim, result);
      }
    } else {  // val is right of the splitting plane.
      if (node.right) {
        tree_search_n(val, *node.right, next_dim, result);
      }
      // Traverse to the other side if still needed.
      if (node.left &&
          nth_best.distance.IntersectsPlane(distance_to_plane, dim)) {
        tree_search_n(val, *node.left, next_dim, result);
      }
    }
  }

  std::unique_ptr<KdNode> head_ = nullptr;
  absl::flat_hash_map<Key, KdNode*> items_;
  const float max_height_factor_;
};

}  // namespace widders

#endif  // WIDDERS_CONTAINER_SCAPEGOAT_KD_MAP_H_
