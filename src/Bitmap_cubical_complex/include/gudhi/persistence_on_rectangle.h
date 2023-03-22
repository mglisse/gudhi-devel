/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s): Marc Glisse
 *
 *    Copyright (C) 2023 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

//  ds_find_set_ is inspired from code in Boost.Graph that is
//
//  (C) Copyright Jeremy Siek 2004
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#ifndef PERSISTENCE_ON_RECTANGLE_H
#define PERSISTENCE_ON_RECTANGLE_H

#include <gudhi/Debug_utils.h>
#ifdef GUDHI_DETAILED_TIMES
 #include <gudhi/Clock.h>
#endif

#include <boost/range/adaptor/reversed.hpp>

#ifdef GUDHI_USE_TBB
 #include <tbb/parallel_sort.h>
#endif

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <utility>
#include <stdexcept>
#include <cstddef>
#include <numeric>
#include <functional>

namespace Gudhi {

// TODO: specify in the name that the values are for top cells
// TODO: maybe check if it only works for dimensions[i] >= 3
// TODO: make it possible to choose if we want to output a value or an index into input
template <class Filtration_value, class Index = std::size_t, bool output_index = false>
struct Persistence_on_rectangle {
  // If we want to save space, we don't have to store the redundant 'first'
  // field in T_with_index. However, it would slow down the primal/dual passes.
  struct T_with_index {
    Filtration_value first; Index second;
    T_with_index() = default;
    T_with_index(Filtration_value f, Index i) : first(f), second(i) {}
    bool operator<(T_with_index const& other) const { return std::tie(first, second) < std::tie(other.first, other.second); }
    Index out() const { return second; }
  };
  struct T_no_index {
    Filtration_value first;
    T_no_index() = default;
    T_no_index(Filtration_value f, Index) : first(f) {}
    bool operator<(T_no_index const& other) const { return first < other.first; }
    Filtration_value out() const { return first; }
  };
  typedef std::conditional_t<output_index, T_with_index, T_no_index> T;
  std::vector<Filtration_value> const* input_p;
  // size_* counts the number of vertices in each direction.
  Index size_x, size_y;
  // The square i + dy is right above i.
  Index dy;

  // Squares keep their index from the input.
  // Vertices have the index of the square at their bottom left (smaller x and y)
  // Store the filtration value of vertices / squares that could be critical.
  std::unique_ptr<T[]> data_v_;
  std::unique_ptr<Filtration_value[]> data_s_;
  T& data_vertex(Index i){ return data_v_[i]; }
  T data_vertex(Index i) const { return data_v_[i]; }
  Filtration_value& data_square(Index i){ return data_s_[i]; }
  Filtration_value data_square(Index i) const { return data_s_[i]; }

  // Information on a cluster
  // We do not use the rank/size heuristics, they do not go well with the pre-pairing and end up slowing things down.
  // We thus use the same representative for disjoint-sets and persistence (the minimum).
  std::unique_ptr<Index[]> ds_parent_v_;
  std::vector<Index> ds_parent_s_;
  Index& ds_parent_vertex(Index n) { return ds_parent_v_[n]; }
  Index& ds_parent_square(Index n) { return ds_parent_s_[n]; }

  template<class Parent>
  Index ds_find_set_(Index v, Parent&&ds_parent) {
    // Experimentally, path halving is currently the fastest. Note that with a
    // different algorithm, full compression was faster, so make sure to check
    // again if the algorithm changes.
    // (the setting is unusual because we start from a forest with broken ranks)
#if 0
    // Full compression
    Index old = v;
    Index ancestor = ds_parent(v);
    while (ancestor != v)
    {
      v = ancestor;
      ancestor = ds_parent(v);
    }
    v = ds_parent(old);
    while (ancestor != v)
    {
      ds_parent(old) = ancestor;
      old = v;
      v = ds_parent(old);
    }
    return ancestor;
#elif 1
    // Path halving
    Index parent = ds_parent(v);
    Index grandparent = ds_parent(parent);
    while (parent != grandparent)
    {
      ds_parent(v) = grandparent;
      v = grandparent;
      parent = ds_parent(v);
      grandparent = ds_parent(parent);
    }
    return parent;
#elif 1
    // Path splitting
    Index parent = ds_parent(v);
    Index grandparent = ds_parent(parent);
    while (parent != grandparent)
    {
      ds_parent(v) = grandparent;
      v = parent;
      parent = grandparent;
      grandparent = ds_parent(parent);
    }
    return parent;
#elif 1
    // No compression (just for reference)
    Index parent;
    while (v != (parent = ds_parent(v)))
      v = parent;
    return v;
#endif
  }
  Index ds_find_set_vertex(Index v) {
    return ds_find_set_(v, [this](Index i) -> Index& { return ds_parent_vertex(i); });
  }
  Index ds_find_set_square(Index v) {
    return ds_find_set_(v, [this](Index i) -> Index& { return ds_parent_square(i); });
  }

  void init(const std::vector<unsigned>& dimensions, const std::vector<Filtration_value>& input_) {
#ifdef DEBUG_TRACES
    std::clog << "Input\n";
    for(Index i = 0; i < input_.size(); ++i) {
      std::clog << i << '\t' << input_[i] << '\n';
    }
#endif
    GUDHI_CHECK(dimensions.size() == 2, std::logic_error("persistence_2d_dual() only works on 2-dimensional complexes"));
    GUDHI_CHECK(dimensions[0] * dimensions[1] == input_.size(),
        std::invalid_argument("Number of cells inconsistent with dimensions"));
    input_p = &input_;
    dy = dimensions[0];
    size_x = dy - 1;
    size_y = dimensions[1] - 1;
    // The unique_ptr could be std::vector, but the initialization is useless.
    data_v_.reset(new T[input_.size() - dy - 1]); // 1 row/column less for vertices than squares
    data_s_.reset(new Filtration_value[input_.size()]);
    ds_parent_v_.reset(new Index[input_.size() - dy - 1]);
    ds_parent_s_.resize(input_.size()); // Initializing the boundary squares to 0 is important
    // Everything, and in particular the boundary squares, has cell 0 (representing the infinite exterior cell) as representative by default.
    edges.reserve(input_.size() / 2); // TODO: what is a good estimate here? For a random 1000x1000 input, we get ~311k edges. For a checkerboard, ~498k.
  }
  struct Edge {
    T f;
    Index v1, v2; // v1 < v2
    Edge() = default;
    Edge(T f, Index v1, Index v2) : f(f), v1(v1), v2(v2) {}
    Filtration_value filt() const { return f.first; }
    bool operator<(Edge const& other) const { return filt() < other.filt(); }
  };
  void dualize_edge(Edge& e) const {
    Index new_v2 = e.v1 + (dy + 1);
    e.v1 = e.v2;
    e.v2 = new_v2;
  };
  std::vector<Edge> edges;

  Filtration_value input(Index i) const { return (*input_p)[i]; }

  bool has_larger_input(Index a, Index b, Filtration_value fb) const {
    // Is passing fb useful, or would the compiler notice that it already has it available?
    GUDHI_CHECK(a != b, std::logic_error("Bug in Gudhi"));
    Filtration_value fa = input(a);
    if (fa > fb) return true;
    if (fa < fb) return false;
    return a > b;
  }
  void set_parent_vertex(Index child, Index parent) {
    GUDHI_CHECK(child != parent, std::logic_error("Bug in Gudhi: use mark_*_critical instead of set_parent"));
    ds_parent_vertex(child) = parent;
  }
  void set_parent_square(Index child, Index parent) {
    GUDHI_CHECK(child != parent, std::logic_error("Bug in Gudhi: use mark_*_critical instead of set_parent"));
    ds_parent_square(child) = parent;
  }

  // Locally pair simplices around each square.
  // Work implicitly from input, only store the filtration value of critical vertices and squares.
  // Store critical edges for later processing.
  void fill_and_pair() {
    Index i; // Index of the current square
    Filtration_value f; // input(i)
    auto mark_vertex_critical = [&](Index c) {
      ds_parent_vertex(c) = c;
      data_vertex(c) = T(f, i);
    };
    auto mark_square_critical = [&]() {
      ds_parent_square(i) = i;
      data_square(i) = f;
    };
    auto mark_edge_critical = [&](Index v1, Index v2) {
      edges.emplace_back(T(f, i), v1, v2);
    };
    auto  v_ul = [&](){ return i - 1; };
    auto  v_ur = [&](){ return i; };
    auto  v_dl = [&](){ return i - dy - 1; };
    auto  v_dr = [&](){ return i - dy; };
    auto pair_square_u = [&](){ set_parent_square(i, i + dy); };
    auto pair_square_d = [&](){ set_parent_square(i, i - dy); };
    auto pair_square_l = [&](){ set_parent_square(i, i - 1); };
    auto pair_square_r = [&](){ set_parent_square(i, i + 1); };

    // Mark the corners as critical, it will be overwritten if not
    i = 0; f = input(i);
    mark_vertex_critical(v_ur());
    i = size_x; f = input(i);
    mark_vertex_critical(v_ul());
    i = dy * size_y; f = input(i);
    mark_vertex_critical(v_dr());
    i = size_x + dy * size_y; f = input(i);
    mark_vertex_critical(v_dl());

    // Boundary nodes, 1st row
    for(Index x = 1; x < size_x; ++x) {
      i = x;
      f = input(x);
      if (has_larger_input(i + dy, i, f)) {
        auto ul = [&](){ return has_larger_input(i - 1, i, f) && has_larger_input(i + dy - 1, i, f); };
        auto ur = [&](){ return has_larger_input(i + 1, i, f) && has_larger_input(i + dy + 1, i, f); };
        if (ul()) {
          set_parent_vertex(v_ul(), v_ur());
          if (ur()) mark_vertex_critical(v_ur());
        } else if (ur()) {
          set_parent_vertex(v_ur(), v_ul());
        } else {
          mark_edge_critical(v_ul(), v_ur());
        }
      }
    }
    // Internal rows
    for(Index y = 1; y < size_y; ++y) {
      // First column
      {
        i = y * dy;
        f = input(i);
        if (has_larger_input(i + 1, i, f)) {
          auto dr = [&](){ return has_larger_input(i - dy, i, f) && has_larger_input(i + 1 - dy, i, f); };
          auto ur = [&](){ return has_larger_input(i + dy, i, f) && has_larger_input(i + 1 + dy, i, f); };
          if (dr()) {
            set_parent_vertex(v_dr(), v_ur());
            if (ur()) mark_vertex_critical(v_ur());
          } else if (ur()) {
            set_parent_vertex(v_ur(), v_dr());
          } else {
            mark_edge_critical(v_dr(), v_ur());
          }
        }
      }
      // Internal squares
      for(Index x = 1; x < size_x; ++x) {
        i = x + dy * y;
        f = input(i);
        // See what part of the boundary shares f
        auto l = [&]() { return has_larger_input(i - 1, i, f); };
        auto r = [&]() { return has_larger_input(i + 1, i, f); };
        auto d = [&]() { return has_larger_input(i - dy, i, f); };
        auto u = [&]() { return has_larger_input(i + dy, i, f); };
        auto dl = [&]() { return has_larger_input(i - dy - 1, i, f); };
        auto ul = [&]() { return has_larger_input(i + dy - 1, i, f); };
        auto dr = [&]() { return has_larger_input(i - dy + 1, i, f); };
        auto ur = [&]() { return has_larger_input(i + dy + 1, i, f); };
        if (u()) { // u
          if (l()) { // u l
            if (ul()) { // u l ul
              set_parent_vertex(v_ul(), v_ur());
              if (d()) { // U l UL d
                if (dl()) { // U l UL d dl
                  set_parent_vertex(v_dl(), v_ul());
                  if (r()) { // U L UL d DL r
                    if (dr()) { // U L UL d DL r dr
                      set_parent_vertex(v_dr(), v_dl());
                      pair_square_r();
                      if (ur()) { // U L UL D DL R DR ur - cr
                        mark_vertex_critical(v_ur());
                      }
                    } else { // U L UL d DL r !dr
                      pair_square_d();
                      if (ur()) { // U L UL D DL r !dr ur - cd
                        set_parent_vertex(v_ur(), v_dr());
                      } else { // U L UL D DL r !dr !ur - cd
                        mark_edge_critical(v_dr(), v_ur());
                      }
                    }
                  } else { // U L UL d DL !r
                    pair_square_d();
                  }
                } else { // U l UL d !dl
                  pair_square_l();
                  if (r()) { // U L UL d !dl r - cl
                    if (dr()) { // U L UL d !dl r dr - cl
                      set_parent_vertex(v_dr(), v_dl());
                    } else { // U L UL d !dl r !dr - cl
                      mark_edge_critical(v_dl(), v_dr());
                    }
                    if (ur()) { // U L UL D !dl r ur - cl
                      set_parent_vertex(v_ur(), v_dr());
                    } else { // U L UL D !dl r !ur - cl
                      mark_edge_critical(v_dr(), v_ur());
                    }
                  } else { // U L UL d !dl !r - cl
                    mark_edge_critical(v_dl(), v_dr());
                  }
                }
              } else { // U l UL !d
                pair_square_l();
                if (r()) { // U L UL !d r - cl
                  if (ur()) { // U L UL !d r ur - cl
                    set_parent_vertex(v_ur(), v_dr());
                  } else { // U L UL !d r !ur - cl
                    mark_edge_critical(v_dr(), v_ur());
                  }
                } else {} // U L UL !d !r - cl
              }
            } else { // u l !ul
              pair_square_u();
              if (d()) { // U l !ul d - cu
                if (dl()) { // U l !ul d dl - cu
                  set_parent_vertex(v_dl(), v_ul());
                } else { // U l !ul d !dl - cu
                  mark_edge_critical(v_dl(), v_ul());
                }
                if (r()) { // U L !ul d r - cu
                  if (dr()) { // U L !ul d r dr - cu
                    set_parent_vertex(v_dr(), v_dl());
                  } else { // U L !ul d r !dr - cu
                    mark_edge_critical(v_dl(), v_dr());
                  }
                  if (ur()) { // U L !ul D r ur - cu
                    set_parent_vertex(v_ur(), v_dr());
                  } else { // U L !ul D r !ur - cu
                    mark_edge_critical(v_dr(), v_ur());
                  }
                } else { // U L !ul d !r - cu
                  mark_edge_critical(v_dl(), v_dr());
                }
              } else { // U l !ul !d - cu
                mark_edge_critical(v_dl(), v_ul());
                if (r()) { // U L !ul !d r - cu
                  if (ur()) { // U L !ul !d r ur - cu
                    set_parent_vertex(v_ur(), v_dr());
                  } else { // U L !ul !d r !ur - cu
                    mark_edge_critical(v_dr(), v_ur());
                  }
                } else {} // U L !ul !d !r - cu
              }
            }
          } else { // u !l
            pair_square_u();
            if (d()) { // U !l d - cu
              if (r()) { // U !l d r - cu
                if (dr()) { // U !l d r dr - cu
                  set_parent_vertex(v_dr(), v_dl());
                } else { // U !l d r !dr - cu
                  mark_edge_critical(v_dl(), v_dr());
                }
                if (ur()) { // U !l D r ur - cu
                  set_parent_vertex(v_ur(), v_dr());
                } else { // U !l D r !ur - cu
                  mark_edge_critical(v_dr(), v_ur());
                }
              } else { // U !l d !r - cu
                mark_edge_critical(v_dl(), v_dr());
              }
            } else { // U !l !d - cu
              if (r()) { // U !l !d r - cu
                if (ur()) { // U !l !d r ur - cu
                  set_parent_vertex(v_ur(), v_dr());
                } else { // U !l !d r !ur - cu
                  mark_edge_critical(v_dr(), v_ur());
                }
              } else {} // U !l !d !r - cu
            }
          }
        } else { // !u
          if (l()) { // !u l
            if (d()) { // !u l d
              if (dl()) { // !u l d dl
                set_parent_vertex(v_dl(), v_ul());
                if (r()) { // !u L d DL r
                  if (dr()) { // !u L d DL r dr
                    set_parent_vertex(v_dr(), v_dl());
                  } else { // !u L d DL r !dr
                    mark_edge_critical(v_dl(), v_dr());
                  }
                  pair_square_r();
                } else { // !u L d DL !r
                  pair_square_d();
                }
              } else { // !u l d !dl
                pair_square_l();
                if (r()) { // !u L d !dl r - cl
                  if (dr()) { // !u L d !dl r dr - cl
                    set_parent_vertex(v_dr(), v_dl());
                  } else { // !u L d !dl r !dr - cl
                    mark_edge_critical(v_dl(), v_dr());
                  }
                  mark_edge_critical(v_dr(), v_ur());
                } else { // !u L d !dl !r - cl
                  mark_edge_critical(v_dl(), v_dr());
                }
              }
            } else { // !u l !d
              pair_square_l();
              if (r()) { // !u L !d r - cl
                mark_edge_critical(v_dr(), v_ur());
              } else {} // !u L !d !r - cl
            }
          } else { // !u !l
            if (d()) { // !u !l d
              pair_square_d();
              if (r()) { // !u !l D r - cd
                if (dr()) { // !u !l D r dr - cd
                  set_parent_vertex(v_dr(), v_ur());
                } else { // !u !l D r !dr - cd
                  mark_edge_critical(v_dr(), v_ur());
                }
              } else {} // !u !l D !r - cd
            } else { // !u !l !d
              if (r()) { // !u !l !d r
                pair_square_r();
              } else { // !u !l !d !r
                mark_square_critical();
              }
            }
          }
        }
      }
      // Last column
      {
        i = size_x + dy * y;
        f = input(i);
        if (has_larger_input(i - 1, i, f)) {
          auto dl = [&](){ return has_larger_input(i - dy, i, f) && has_larger_input(i - 1 - dy, i, f); };
          auto ul = [&](){ return has_larger_input(i + dy, i, f) && has_larger_input(i - 1 + dy, i, f); };
          if (dl()) {
            set_parent_vertex(v_dl(), v_ul());
            if (ul()) mark_vertex_critical(v_ul());
          } else if (ul()) {
            set_parent_vertex(v_ul(), v_dl());
          } else {
            mark_edge_critical(v_dl(), v_ul());
          }
        }
      }
    }
    // Boundary nodes, last row
    for(Index x = 1; x < size_x; ++x) {
      i = size_y * dy + x;
      f = input(i);
      if (has_larger_input(i - dy, i, f)) {
        auto dl = [&](){ return has_larger_input(i - 1, i, f) && has_larger_input(i - dy - 1, i, f); };
        auto dr = [&](){ return has_larger_input(i + 1, i, f) && has_larger_input(i - dy + 1, i, f); };
        if (dl()) {
          set_parent_vertex(v_dl(), v_dr());
          if (dr()) mark_vertex_critical(v_dr());
        } else if (dr()) {
          set_parent_vertex(v_dr(), v_dl());
        } else {
          mark_edge_critical(v_dl(), v_dr());
        }
      }
    }
  }

  void sort_edges(){
#ifdef GUDHI_USE_TBB
    // Parallelizing just this part is a joke. It would be possible to
    // parallelize the pairing (one edge list per thread) and run the dual in
    // parallel with the primal if we were motivated...
    tbb::parallel_sort(edges.begin(), edges.end());
#else
    std::sort(edges.begin(), edges.end());
#endif
#ifdef DEBUG_TRACES
    std::clog << "edges\n";
    for(auto&e : edges){ std::clog << e.v1 << '\t' << e.v2 << '\t' << e.filt() << '\n'; }
#endif
  }

  template<class Out>
  void primal(Out&&out){
    auto it = std::remove_if(edges.begin(), edges.end(), [&](Edge& e) {
        assert(e.v1 < e.v2);
        Index a = ds_find_set_vertex(e.v1);
        Index b = ds_find_set_vertex(e.v2);
        if (a == b) return false;
        if (data_vertex(b) < data_vertex(a)) std::swap(a, b);
        ds_parent_vertex(b) = a;
        out(data_vertex(b).out(), e.f.out());
        return true;
    });
    edges.erase(it, edges.end());
  }

  template<class Out>
  void dual(Out&&out){
    for (auto e : boost::adaptors::reverse(edges)) {
      dualize_edge(e);
      Index a = ds_find_set_square(e.v1);
      Index b = ds_find_set_square(e.v2);
      GUDHI_CHECK(a != b, std::logic_error("Bug in Gudhi"));
      // This is more robust in case the input contains inf? I used to set the filtration of 0 to inf.
      if (b == 0 || (a != 0 && data_square(a) < data_square(b))) std::swap(a, b);
      ds_parent_square(b) = a;
      if constexpr (output_index)
        out(e.f.out(), b);
      else
        out(e.f.out(), data_square(b));
    }
  }
  auto global_min(){
    return data_vertex(ds_find_set_vertex(0)).out();
  }
};

// TODO: pass dimensions as a pair or array<,2>
template <typename U, typename Out0, typename Out1>
auto persistence_on_rectangle(const std::vector<unsigned>& dimensions, const std::vector<U>& input, Out0&&out0, Out1&&out1){
#ifdef GUDHI_DETAILED_TIMES
  Gudhi::Clock clock;
#endif
  Persistence_on_rectangle<U, unsigned> X;
  X.init(dimensions, input);
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "init: " << clock; clock.begin();
#endif
  X.fill_and_pair();
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "fill and pair: " << clock; clock.begin();
#endif
  X.sort_edges();
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "sort: " << clock; clock.begin();
#endif
  X.primal(out0);
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "primal pass: " << clock; clock.begin();
#endif
  X.dual(out1);
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "dual pass: " << clock;
#endif
  return X.global_min();
}

// Undocumented, exists to ensure that we do not break the possibility to get indices.
template <typename U, typename Out0, typename Out1>
auto persistence_on_rectangle_index(const std::vector<unsigned>& dimensions, const std::vector<U>& input, Out0&&out0, Out1&&out1){
#ifdef GUDHI_DETAILED_TIMES
  Gudhi::Clock clock;
#endif
  Persistence_on_rectangle<U, unsigned, true> X;
  X.init(dimensions, input);
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "init: " << clock; clock.begin();
#endif
  X.fill_and_pair();
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "fill and pair: " << clock; clock.begin();
#endif
  X.sort_edges();
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "sort: " << clock; clock.begin();
#endif
  X.primal(out0);
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "primal pass: " << clock; clock.begin();
#endif
  X.dual(out1);
#ifdef GUDHI_DETAILED_TIMES
    std::clog << "dual pass: " << clock;
#endif
  return X.global_min();
}

}  // namespace Gudhi

#endif  // PERSISTENCE_ON_RECTANGLE_H
