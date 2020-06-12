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
#include "widders/container/kd_metrics.h"

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
  static constexpr bool kBalanceGuaranteed = false;
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
  static constexpr bool kBalanceGuaranteed = true;
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
  bool contains(const Key& key) const { return items_.contains(key); }
  // Return the height of the tree: the number of nodes that must be
  // traversed to reach the deepest node in the tree.
  size_t height() const { return head_->height; }

  // Set the point value of the given key.
  // If the key already exists in the tree, its value will be changed.
  // Takes O(height) time.
  void set(const Key& key, Point val) {
    KdNode*& item_node = items_[key];
    if (item_node) {
      // The key already existed. Re-insert its node with the new value.
      std::unique_ptr<KdNode> reinsert_node = tree_pop_node(item_node);
      reinsert_node->val = std::move(val);
      // Reinsert the removed node with our new value.
      item_node = reinsert_node.get();
      tree_insert_node(std::move(reinsert_node));
    } else {
      auto new_node = std::make_unique<KdNode>(key, std::move(val));
      item_node = new_node.get();
      tree_insert_node(std::move(new_node));
    }
  }

  // Remove the given key from the tree, returning 1 if it existed
  // (0 otherwise).
  // Takes O(height) time.
  size_t erase(const Key& key) {
    const auto found_item = items_.find(key);
    if (found_item != items_.end()) {
      // Pop the node out of the tree and delete it.
      std::unique_ptr<KdNode> removed_node = tree_pop_node(found_item->second);
      // Remove the item from the hash table.
      items_.erase(found_item);
      return 1;
    } else {
      return 0;
    }
  }

  template <typename DistanceType = EuclideanDistance<long double>>
  NearestResult<DistanceType> nearest(const Point& val,
                                      DistanceType max_distance = {}) const {
    NearestResult<DistanceType> result(max_distance);
    tree_search(val, *head_, 0, &result);
    return result;
  }

  template <typename DistanceType = EuclideanDistance<long double>>
  std::vector<NearestResult<DistanceType>> nearest_n(
      const Point& val, size_t n, DistanceType max_distance = {}) const {
    std::vector<NearestResult<DistanceType>> result(n, {max_distance});
    tree_search_n(val, *head_, 0, &result);
    absl::c_sort_heap(result);
    // Find out how many items we actually located, eliminating all those which
    // are still empty at the end of our found set.
    size_t i = n;
    while (!result[i - 1].val) --i;
    result.resize(i);
    return result;
  }

#ifndef NDEBUG
  void validate() const {
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
      assert(!head_ || tree_is_balanced(head_->height, size()));
    }
  }
#endif

 private:
  struct KdNode {
    // Our nodes store parent-node pointers because the keys of the structure
    // are indexed in the hash table and the values are indexed in the tree, and
    // we want to be able to remove items by key. Therefore, we usually did not
    // reach the containing node by traversing to it from the root when we want
    // to remove it, rather looking it up directly from the hash table.
    KdNode* parent = nullptr;
    std::unique_ptr<KdNode> left = nullptr;
    std::unique_ptr<KdNode> right = nullptr;
    Key key = {};
    Point val = {};
    Dimension mid = {};
    // Absolute height this subtree, including this node. (Always 1 or greater.)
    uint32_t height = 0;

    KdNode(Key k, Point v) : key(std::move(k)), val(std::move(v)) {}
    KdNode() = default;
    ~KdNode() = default;
  };

#ifndef NDEBUG
  size_t validate_node(const KdNode& node, const size_t dim, const Point& lower,
                       const Point& upper) const {
    // Check key reference is correct
    assert(items_[node.key] == &node);
    // Check val is inside the bounds
    for (size_t i = 0; i < dims; ++i) {
      assert(node.val[i] >= lower[i]);
      assert(node.val[i] <= upper[i]);
    }
    // Check mid is inside the bounds
    assert(node.mid >= lower[dim]);
    assert(node.mid <= upper[dim]);
    // Check height is correct
    assert(node.height == 1 + std::max((node.left ? node.left->height : 0),
                                       (node.right ? node.right->height : 0)));
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
#endif

  // Update the stats of the given node and its ancestors. The passed pointer
  // must never be null.
  // TODO(widders): consider queueing and deferring these updates?
  static void revise_stats(KdNode* node) {
    do {
      auto new_height = 1 + std::max(node->left ? node->left->height : 0,
                                     node->right ? node->right->height : 0);
      if (node->height == new_height) {
        return;
      } else {
        node->height = new_height;
        node = node->parent;
        continue;
      }
    } while (node);
  }

  // Check each subtree starting from the given node, all the way up
  // the tree looking for the smallest subtree that is unbalanced, then
  // rebuild it.
  void rebuild_one_ancestor(KdNode* tree, size_t dim) {
    size_t node_count = count_nodes(tree);
    while (true) {
      if (tree_is_balanced(tree->height, node_count)) {
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

  static void rebuild(std::unique_ptr<KdNode>& tree_root, size_t node_count,
                      size_t dim) {
    KdNode* const parent = tree_root->parent;
    std::vector<std::unique_ptr<KdNode>> nodes;
    nodes.reserve(node_count);
    collect_nodes(std::move(tree_root), &nodes);
    tree_root = std::move(rebuild_recursive(dim, nodes.begin(), nodes.end()));
    tree_root->parent = parent;
    if (parent) revise_stats(parent);
  }

  static size_t count_nodes(KdNode* tree) {
    return tree ? 1 + count_nodes(tree->left.get()) +
                      count_nodes(tree->right.get())
                : 0;
  }

  static void collect_nodes(std::unique_ptr<KdNode> node,
                            std::vector<std::unique_ptr<KdNode>>* vec) {
    KdNode& node_ref = *node;
    if (node_ref.left) collect_nodes(std::move(node_ref.left), vec);
    vec->push_back(std::move(node));
    if (node_ref.right) collect_nodes(std::move(node_ref.right), vec);
  }

  using VecIter = typename std::vector<std::unique_ptr<KdNode>>::iterator;

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
      left_depth = left_child->height;
      node->left = std::move(left_child);
    }
    if (pivot + 1 == end) {
      right_depth = 0;
    } else {
      std::unique_ptr<KdNode> right_child =
          rebuild_recursive(next_dim, pivot + 1, end);
      right_child->parent = node.get();
      right_depth = right_child->height;
      node->right = std::move(right_child);
    }
    // Fix node's metadata
    node->height = 1 + std::max(left_depth, right_depth);
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
      const auto left_depth = current->left ? current->left->height : 0;
      if (current->right && current->right->height >= left_depth) {
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
      revise_stats(popped_parent);
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
      items_[popped->key] = node;
      // Swap the popped node's key and value into that same staying node.
      std::swap(node->key, popped->key);
      std::swap(node->val, popped->val);
    }

    // Check for tree balance.
    if (head_ && !tree_is_balanced(head_->height, size())) {
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
    node->height = 1;
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
      revise_stats(current);
      // Rebuild parts of the tree if necessary
      if (!tree_is_balanced(head_->height, size()))
        rebuild_one_ancestor(current, dim);
    }
  }

  // Search the tree to find the nearest key/value to the given point.
  template <typename DistanceType>
  void tree_search(const Point& val, const KdNode& node, const size_t dim,
                   NearestResult<DistanceType>* result) const {
#ifdef KD_SEARCH_STATS
    result.nodes_searched++;
#endif
    // Update best result.
    if (result->distance.template ImproveDistance<dims>(val, node.val,
                                                        node.key) == BETTER) {
      result->key = node.key;
      result->val = &node.val;
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
          result->distance.IntersectsPlane(distance_to_plane, dim)) {
        tree_search(val, *node.right, next_dim, result);
      }
    } else {  // val is right of the splitting plane.
      if (node.right) {
        tree_search(val, *node.right, next_dim, result);
      }
      // Traverse to the other side if still needed.
      if (node.left &&
          result->distance.IntersectsPlane(distance_to_plane, dim)) {
        tree_search(val, *node.left, next_dim, result);
      }
    }
  }

  // Search the tree to find the nearest key/value to the given point.
  template <typename DistanceType>
  void tree_search_n(const Point& val, const KdNode& node, const size_t dim,
                     std::vector<NearestResult<DistanceType>>* result) const {
    NearestResult<DistanceType>& nth_best = *result->begin();
    // Update best result.
    if (nth_best.distance.template ImproveDistance<dims>(val, node.val,
                                                         node.key) == BETTER) {
      nth_best.key = node.key;
      nth_best.val = &node.val;
      // By replacing values in the root of the heap, then popping & pushing,
      // that item is swapped to the back and then reinserted.
      absl::c_pop_heap(*result);
      absl::c_push_heap(*result);
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
