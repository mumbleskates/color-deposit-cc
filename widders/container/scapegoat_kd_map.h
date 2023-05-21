#ifndef WIDDERS_CONTAINER_SCAPEGOAT_KD_MAP_H_
#define WIDDERS_CONTAINER_SCAPEGOAT_KD_MAP_H_

#include <array>
#include <cassert>
#include <cmath>
#include <compare>
#include <cstddef>
#include <limits>
#include <memory>
#include <ratio>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "widders/container/kd_metrics.h"
#include "widders/container/kd_searchers.h"
#include "widders/container/kd_value_traits.h"

namespace widders {

using ::std::size_t;

// --------------------------------------------------------------------------

struct NoStatistic;

// TODO(widders): doc
template <typename Key, typename Value, typename Statistic = NoStatistic,
          typename balance_ratio = std::ratio<3, 2>>
class ScapegoatKdMap {
  constexpr static size_t dims = dimensions_of<Value>;
  using Dimension = dimension_type<Value>;

  static_assert(dims > 0, "k-d tree with less than 1 dimension");
  // 1.0 represents the minimum possible height of a binary tree of a given
  // size. Values less than 1.0 will result in a highly unbalanced tree that
  // continually rebuilds only its tiniest ends.
  static_assert(static_cast<double>(balance_ratio::num) /
                        static_cast<double>(balance_ratio::den) >
                    1.0,
                "balance_ratio must be greater than 1.0");

  // TODO(widders): maybe memoize
  inline bool tree_is_balanced(uint32_t height, size_t node_count) const {
    // 2 + (implicit) floor of (log2 of tree node count multiplied by the
    //                               height balance we maintain)
    // We add 2 instead of just 1 because the tree "height" statistic we
    // maintain is 1 for a tree with no children, for simplicity; this is 1 more
    // than the "height" of a tree in the literature, but it also means an empty
    // tree has a different height than a tree with 1 node.
    return height <=
           2 + static_cast<uint32_t>(
                   std::log2f(static_cast<float>(node_count)) *
                   static_cast<float>(static_cast<double>(balance_ratio::num) /
                                      static_cast<double>(balance_ratio::den)));
  }

  struct KdNode;

 public:
  using key = Key;
  using value = Value;

  ScapegoatKdMap() = default;
  // Move constructor
  ScapegoatKdMap(ScapegoatKdMap&& move_from) noexcept = default;
  ScapegoatKdMap& operator=(ScapegoatKdMap&& move_from) noexcept = default;
  // TODO(widders): Add constructor(s) for preexisting value sets all at once
  // Disable copy construction for now
  ScapegoatKdMap(ScapegoatKdMap& copy_from) = delete;
  ScapegoatKdMap& operator=(ScapegoatKdMap& copy_from) = delete;
  ~ScapegoatKdMap() = default;

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
  const Value& get(const Key& key) const { return items_.at(key)->val; }
  const Value& operator[](const Key& key) const { return get(key); }
  // Returns true if the key is in the tree, false otherwise.
  // Takes O(1) time.
  bool contains(const Key& key) const { return items_.contains(key); }
  // Return the height of the tree: the number of nodes that must be
  // traversed to reach the deepest node in the tree.
  size_t height() const { return head_->height; }

  // Set the point value of the given key.
  // If the key already exists in the tree, its value will be changed.
  // Takes O(height) time.
  void set(const Key& key, Value val) {
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

  template <template <typename...> typename ST, typename... Apply,
            typename... SearcherArgs>
  auto search(SearcherArgs... args) const {
    return search<ST<Key, Value, Statistic, Apply...>>(
        std::forward<SearcherArgs>(args)...);
  }

  template <Searcher<Key, Value, Statistic> S, typename... SearcherArgs>
  S search(SearcherArgs... args) const {
    auto searcher = S(std::forward<SearcherArgs>(args)...);
    searchWith(searcher);
    return searcher;
  }

  template <Searcher<Key, Value, Statistic> S>
  S& searchWith(S& searcher) const {
    if (head_) {
      tree_search(*head_, 0, &searcher);
    }
    return searcher;
  }

#ifndef NDEBUG
  void validate() const {
    if (!head_) {
      assert(empty());
    } else {
#ifndef KD_TREE_DEBUG_ONLY_BALANCE
      assert(!head_->parent);
      Value lower, upper;
      Dimension low, high;
      if constexpr (std::numeric_limits<Dimension>::has_infinity) {
        low = -std::numeric_limits<Dimension>::infinity();
        high = std::numeric_limits<Dimension>::infinity();
      } else {
        low = std::numeric_limits<Dimension>::lowest();
        high = std::numeric_limits<Dimension>::max();
      }
      for (auto& d : lower) d = low;
      for (auto& d : upper) d = high;
      assert(validate_node(*head_, 0, lower, upper) == items_.size());
#endif  // !KD_TREE_DEBUG_ONLY_BALANCE
      // We are never more than +1 out of total height balance.
      //
      // We cannot guarantee *complete* satisfaction of the tree balance at all
      // times unless the tree is only modified by insertion. An insert-only
      // tree only becomes unbalanced on insertion when the inserted node is the
      // new deepest node in the tree, and then its subtree will be rebuilt
      // shorter, bringing the whole tree back into balance.
      //
      // However, when a tree becomes unbalanced on deletion we only rebuild one
      // of its deepest leaves' subtrees. Because there may be multiple
      // imbalanced subtrees with leaves at the same depth over the maximum, the
      // whole tree will only be shortened after one deletion for each of these
      // dangling sub-trees; however, the number of nodes between steps of the
      // tolerated maximum height is always greater than the number of
      // individually unbalanced subtrees that can be over-height, so repeated
      // deletions will get the balance back under control before the threshold
      // shrinks again. Intermixed insertions cannot make the balance worse,
      // since even if they end up at a depth 2 greater than the threshold, and
      // after rebalance that subtree won't be taller than it was before.
      assert(tree_is_balanced(head_->height - 1, size()));
    }
  }
#endif  // !NDEBUG

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
    Value val = {};
    Dimension mid = {};
    // Absolute height this subtree, including this node. (Always 1 or greater.)
    uint32_t height = 1;
    [[no_unique_address]] Statistic stat = {};

    KdNode(Key k, Value v) : key(std::move(k)), val(std::move(v)) {}
    KdNode() = default;
    KdNode(KdNode&&) = default;
    KdNode& operator=(KdNode&&) noexcept = default;
    KdNode(const KdNode&) = delete;
    KdNode& operator=(const KdNode&) = delete;
    ~KdNode() = default;

    Statistic make_stat() const {
      if (left) {
        if (right) {
          return Statistic::combine(std::array{left->val, right->val},
                                    std::array{left->stat, right->stat});
        } else {
          return Statistic::combine(std::array{left->val},
                                    std::array{left->stat});
        }
      } else {
        if (right) {
          return Statistic::combine(std::array{right->val},
                                    std::array{right->stat});
        }
      }
      return {};
    }

    bool update_stats() {
      auto new_height =
          1 + std::max(left ? left->height : 0, right ? right->height : 0);
      auto new_stat = make_stat();
      if (std::tie(new_height, new_stat) == std::tie(height, stat)) {
        return false;
      } else {
        height = new_height;
        stat = new_stat;
        return true;
      }
    }
  };

#ifndef NDEBUG
  size_t validate_node(const KdNode& node, const size_t dim, const Value& lower,
                       const Value& upper) const {
    // Check key reference is correct
    assert(items_.find(node.key)->second == &node);
    // Check val is inside the bounds
    for (size_t i = 0; i < dims; ++i) {
      assert(get_dimension(i, node.val) >= get_dimension(i, lower));
      assert(get_dimension(i, node.val) <= get_dimension(i, upper));
    }
    // Check mid is inside the bounds
    assert(node.mid >= get_dimension(dim, lower));
    assert(node.mid <= get_dimension(dim, upper));
    // Check height is correct
    assert(node.height == 1 + std::max((node.left ? node.left->height : 0),
                                       (node.right ? node.right->height : 0)));
    assert(node.stat == node.make_stat());
    // Check child parent pointers
    if (node.left) assert(node.left->parent == &node);
    if (node.right) assert(node.right->parent == &node);
    // Recurse
    const size_t next_dim = dim + 1 == dims ? 0 : dim + 1;
    size_t children = 0;
    if (node.left) {
      Value new_upper = upper;
      new_upper[dim] = node.mid;
      children += validate_node(*node.left, next_dim, lower, new_upper);
    }
    if (node.right) {
      Value new_lower = lower;
      new_lower[dim] = node.mid;
      children += validate_node(*node.right, next_dim, new_lower, upper);
    }
    return 1 + children;
  }
#endif  // !NDEBUG

  // Update the stats of the given node and its ancestors. The passed pointer
  // must never be null.
  // TODO(widders): consider queueing and deferring these updates?
  static void revise_stats(KdNode* node, KdNode* until = nullptr) {
    while (node != until && node->update_stats()) {
      node = node->parent;
    }
  }

  // Check each subtree starting from the given node, all the way up
  // the tree looking for the smallest subtree that is unbalanced, then
  // rebuild it.
  void rebuild_one_ancestor(KdNode* tree, size_t dim) {
    assert(tree);
    std::vector<std::unique_ptr<KdNode>> collection;
    KdNode* current = tree;
    while (true) {
      KdNode* parent = current->parent;
      std::unique_ptr<KdNode>* current_tree;
      if (parent == nullptr) {
        current_tree = &head_;
      } else if (parent->left.get() == current) {
        current_tree = &parent->left;
      } else {
        current_tree = &parent->right;
      }
      collect_nodes(std::move(*current_tree), &collection);
      if (tree_is_balanced(current->height, collection.size()) &&
          parent != nullptr) {
        // Subtree is sufficiently balanced; continue checking ancestors.
        current = parent;
        // Keep dim in sync with tree's current level.
        dim = dim == 0 ? dims - 1 : dim - 1;
        continue;
      } else {
        *current_tree =
            rebuild_recursive(dim, collection.begin(), collection.end());
        (*current_tree)->parent = parent;
        revise_stats(parent);
        return;
      }
    }
  }

  static void collect_nodes(std::unique_ptr<KdNode> node,
                            std::vector<std::unique_ptr<KdNode>>* vec) {
    KdNode& node_ref = *node;
    if (node_ref.left) collect_nodes(std::move(node_ref.left), vec);
    vec->push_back(std::move(node));
    if (node_ref.right) collect_nodes(std::move(node_ref.right), vec);
  }

  using VecIter = typename std::vector<std::unique_ptr<KdNode>>::iterator;

  template <typename Iter>
  static Iter partition_at_median(Iter start, Iter end, int dim) {
    auto middle = start + (end - start) / 2;
    std::nth_element(
        start, middle, end,
        [dim](const Iter::reference a, const Iter::reference b) -> bool {
          // Compare first on the point at dim; tiebreak with pointer value
          return (std::tuple(get_dimension(dim, a->val), a.get()) <=>
                  std::tuple(get_dimension(dim, b->val), b.get())) < 0;
        });
    return middle;
  }

  static std::unique_ptr<KdNode> rebuild_recursive(size_t dim, VecIter start,
                                                   VecIter end) {
    const size_t next_dim = dim + 1 == dims ? 0 : dim + 1;
    const VecIter pivot = partition_at_median(start, end, dim);
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
    node->stat = node->make_stat();
    node->mid = get_dimension(dim, node->val);
    return node;
  }

  static KdNode* deepest_leaf_of(KdNode* const node) {
    KdNode* current = node;
    // Traverse down the deeper branch until we reach a leaf.
    while (current->height > 1) {
      // Prefer to pick the rightmost deep leaf.
      if (current->right && current->right->height == current->height - 1) {
        // Right subtree exists and is at least as deep as the left.
        current = current->right.get();
      } else {
        // Left subtree is deeper.
        current = current->left.get();
      }
    }
    return current;
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
    KdNode* current = deepest_leaf_of(node);
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
      // Statistics always need to be updated for nodes whose descendents have
      // changed in any way. This includes the parents of the removed node and
      // the node its value was swapped into.
      // The removed node "popped" is below the updated node "node"; update the
      // statistics on the path between them, and also from "node"'s parent up
      // to the root.
      revise_stats(popped_parent, node->parent);
      revise_stats(node->parent);
    } else {
      // "poppped" and "node" are both the node that was popped off and removed.
      revise_stats(popped_parent);
    }
    // Check for tree balance.
    if (head_ && !tree_is_balanced(head_->height, size())) {
      // If the tree is unbalanced after a removal, rebuild some ancestor of
      // the deepest leaf in the tree.
      rebuild_one_ancestor(deepest_leaf_of(head_.get()),
                           (head_->height - 1) % dims);
    }

    return popped;
  }

  // Insert the given node into the tree.
  void tree_insert_node(std::unique_ptr<KdNode> node) {
    // We never need to set the height or stat of a node, because the node is
    // guaranteed to either be a freshly constructed node or a former leaf.
    if (!head_) {
      node->mid = get_dimension(0, node->val);
      node->parent = nullptr;
      head_ = std::move(node);
    } else {
      const Value& val = node->val;
      size_t dim = 0;
      KdNode* current = head_.get();
      size_t insert_depth = 2;  // We will at least insert as a child of head_
      std::unique_ptr<KdNode>* destination;  // This is where we'll insert.
      while (true) {
        // Order by discriminant, or pointer when discriminant is equal
        auto sorted = std::tuple(get_dimension(dim, val), node.get()) <=>
                      std::tuple(current->mid, current);
        if (sorted < 0) {
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
        insert_depth++;
      }
      // Finish inserting node under current.
      const size_t next_dim = dim + 1 == dims ? 0 : dim + 1;
      node->mid = get_dimension(next_dim, node->val);
      node->parent = current;
      *destination = std::move(node);
      // Rebuild the subtree we inserted to if it is out of balance with the
      // whole tree. This is not 100% guaranteed to put the whole tree into
      // balance after one time if the tree is pathologically unbalanced, but it
      // eventually will.
      if (!tree_is_balanced(insert_depth, size())) {
        rebuild_one_ancestor(current, dim);
      } else {
        revise_stats(current);
      }
    }
  }

  // Search the tree to find the nearest key/value to the given point.
  template <Searcher<Key, Value, Statistic> S>
  bool tree_search(const KdNode& node, const size_t dim, S* searcher) const {
    searcher->visit(node.key, node.val);
    if (!node.left && !node.right) return true;
    using enum SearchDirection;
    bool right_first = false;
    bool looking_left = false;
    bool looking_right = false;
    switch (searcher->guide_search(node.mid, dim, Statistic{})) {
      case SKIP:
        return true;
      case LOOK_LEFT: {
        looking_left = true;
        break;
      }
      case LOOK_RIGHT: {
        right_first = true;
        looking_right = true;
        break;
      }
      case LOOK_LEFT_FIRST_BOTH: {
        looking_left = true;
        looking_right = true;
        break;
      }
      case LOOK_RIGHT_FIRST_BOTH: {
        right_first = true;
        looking_left = true;
        looking_right = true;
        break;
      }
      case STOP_SEARCH:
        return false;
    }
    // Traverse downwards.
    const size_t next_dim = dim + 1 == dims ? 0 : dim + 1;
    if (right_first) {
      if (!node.right) {
        // Only have left child
        if (looking_left) {
          return tree_search(*node.left, next_dim, searcher);
        }
      } else {
        // Have right child
        if (looking_right && !tree_search(*node.right, next_dim, searcher)) {
          return false;
        }
        if (looking_left && node.left) {
          // Have both children
          if (looking_right && searcher->recheck()) {
            // Recompute guidance if we already looked at some right children
            switch (searcher->guide_search(node.mid, dim, Statistic{})) {
              case LOOK_LEFT:
              case LOOK_LEFT_FIRST_BOTH:
              case LOOK_RIGHT_FIRST_BOTH:
                return tree_search(*node.left, next_dim, searcher);
              case STOP_SEARCH:
                return false;
              default:
                return true;
            }
          } else {
            return tree_search(*node.left, next_dim, searcher);
          }
        }
      }
    } else {
      // Left first
      if (!node.left) {
        // Only have right child
        if (looking_right) {
          return tree_search(*node.right, next_dim, searcher);
        }
      } else {
        // Have left child
        if (looking_left && !tree_search(*node.left, next_dim, searcher)) {
          return false;
        }
        if (looking_right && node.right) {
          // Have both children
          if (looking_left && searcher->recheck()) {
            // Recompute guidance if we already looked at some left children
            switch (searcher->guide_search(node.mid, dim, Statistic{})) {
              case LOOK_RIGHT:
              case LOOK_LEFT_FIRST_BOTH:
              case LOOK_RIGHT_FIRST_BOTH:
                return tree_search(*node.right, next_dim, searcher);
              case STOP_SEARCH:
                return false;
              default:
                return true;
            }
          } else {
            return tree_search(*node.right, next_dim, searcher);
          }
        }
      }
    }
    return true;
  }

  std::unique_ptr<KdNode> head_ = nullptr;
  absl::flat_hash_map<Key, KdNode*> items_;
};

struct NoStatistic {
  NoStatistic() = default;
  NoStatistic(const NoStatistic&) = default;

  template <typename Iter>
  static NoStatistic combine(Iter stats) {
    return {};
  }

  template <typename ValIter, typename StatIter>
  static NoStatistic combine(ValIter values, StatIter stats) {
    return {};
  }

  bool operator==(const NoStatistic& other) const { return true; }

  char _[0];
};

struct OrderStatistic {
  bool operator==(const OrderStatistic& other) const = default;

  template <typename Iter>
  static OrderStatistic combine(Iter stats) {
    int sum = 0;
    for (OrderStatistic os : stats) {
      sum += os.value;
    }
    return OrderStatistic{sum};
  }

  template <typename ValIter, typename StatsIter>
  static OrderStatistic combine(ValIter values, StatsIter stats) {
    int sum = 0;
    sum += values.size();
    for (OrderStatistic os : stats) {
      sum += os.value;
    }
    return OrderStatistic{sum};
  }

  int value = 0;
};

}  // namespace widders

#endif  // WIDDERS_CONTAINER_SCAPEGOAT_KD_MAP_H_
