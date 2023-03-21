/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s): Marc Glisse
 *
 *    Copyright (C) 2023 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef PERSISTENCE_ON_RECTANGLE_H
#define PERSISTENCE_ON_RECTANGLE_H

#include <gudhi/Debug_utils.h>
#include <gudhi/Clock.h>

#include <boost/pending/disjoint_sets.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/property_map/transform_value_property_map.hpp>
#include <boost/property_map/function_property_map.hpp>
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
#include <limits>
#include <utility>
#include <stdexcept>
#include <cstddef>
#include <numeric>
#include <functional>

namespace Gudhi {

// TODO: specify in the name that the values are for top cells
// TODO: split out into out0 and out1, or pass the dimension to it.
// TODO: maybe check if it only works for dimensions[i] >= 3
// TODO: make it possible to choose if we want to output a value or an index into input
template <class Filtration_value>
struct Persistence_on_rectangle {
  // If we want to save space, we don't have to store the redundant 'first'
  // field in T. Removing it even speeds up the pairing. However, it slows down
  // filling and the primal/dual passes, resulting in a global slow down (but
  // not horrible).

  // std::pair has a bad implementation (constrained by compatibility) in some libraries.
  // typedef std::pair<Filtration_value,std::size_t> T;
  struct T {
    Filtration_value first; std::size_t second;
    bool operator<(T const& other) const { return std::tie(first, second) < std::tie(other.first, other.second); }
  };
  std::vector<Filtration_value> const* input_p;
  std::size_t size_x, size_y, dy, data_size;
  std::unique_ptr<T[]> data;

  // Information on a cluster
  // We only need this for vertices and squares. Since edges and non-edges alternate, we can use n/2 as index.
  // We do not use the rank/size heuristics, they do not go well with the pre-pairing and end up slowing things down.
  // We thus use the same representative for disjoint-sets and persistence (the minimum).
  std::vector<std::size_t> ds_parent_;
  std::size_t& ds_parent(std::size_t n) { return ds_parent_[n / 2]; }
  Gudhi::Clock clock;

  std::size_t ds_find_set(std::size_t v) {
    // Experimentally, path halving is currently the fastest. Note that with a
    // different algorithm, full compression was faster, so make sure to check
    // again if the algorithm changes.
    // (the setting is unusual because we start from a forest with broken ranks)
#if 0
    // Full compression
    std::size_t old = v;
    std::size_t ancestor = ds_parent(v);
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
    std::size_t parent = ds_parent(v);
    std::size_t grandparent = ds_parent(parent);
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
    std::size_t parent = ds_parent(v);
    std::size_t grandparent = ds_parent(parent);
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
    std::size_t parent;
    while (v != (parent = ds_parent(v)))
      v = parent;
    return v;
#endif
  }

  void init(const std::vector<unsigned>& dimensions, const std::vector<Filtration_value>& input_) {
#ifdef DEBUG_TRACES
    std::clog << "Input\n";
    for(std::size_t i = 0; i < input_.size(); ++i) {
      std::clog << i << '\t' << input_[i] << '\n';
    }
#endif
    GUDHI_CHECK(dimensions.size() == 2, std::logic_error("persistence_2d_dual() only works on 2-dimensional complexes"));
    GUDHI_CHECK(dimensions[0] * dimensions[1] == input_.size(),
        std::invalid_argument("Number of cells inconsistent with dimensions"));
    input_p = &input_;
    size_x = dimensions[0] - 1;
    size_y = dimensions[1] - 1;
    dy = 2 * size_x + 1;
    data_size = dy * (2 * size_y + 1);
    data.reset(new T[data_size]); // Could be std::vector, but the initialization is useless
    ds_parent_.resize((data_size + 1) / 2); // TODO: tighten this number a bit
    // Everything has cell 0 (the infinite exterior cell) as representative by default.
    // Real vertices/squares should be their own cluster at the beginning.
    edges.reserve(data_size / 2); // TODO: tighten this number a bit
    // FIXME: is that robust enough? At least make it an argument to the function.
    //T save_data_0 = data[0]; data[0] = std::numeric_limits<T>::infinity();
    std::clog << "init: " << clock; clock.begin();
  }
  struct Edge {
    T f;
    std::size_t v1, v2; // v1 < v2
    Edge(T f, std::size_t v1, std::size_t v2) : f(f), v1(v1), v2(v2) {}
    Filtration_value filt() const { return f.first; }
    bool operator<(Edge const& other) const { return filt() < other.filt(); }
  };
  void dualize_edge(Edge& e) const {
    std::size_t new_v2 = e.v1 + (dy + 1);
    e.v1 = e.v2 - (dy + 1);
    e.v2 = new_v2;
  };
  std::vector<Edge> edges;

  Filtration_value input(std::size_t i) const { return (*input_p)[i]; }
  void set_data(std::size_t cell, std::size_t i) {
    data[cell] = T{ input(i), i };
  }

  void fill_data_from_input() {
    // TODO: see if copying input in a separate loop (for-y for-x) has better memory efficiency, or if we can even swap the 2 'for' in this loop (merging with the next loop may be too much).
    for(std::size_t x = 0; x < size_x + 1; ++x) {
      set_data(2 * x, x);
      for(std::size_t y = 0; y < size_y; ++y) {
        std::size_t cub1 = 2 * x + dy * 2 * y;
        std::size_t e = cub1 + dy;
        std::size_t cub2 = e + dy;
        std::size_t i = x + (size_x + 1) * (y + 1);
        set_data(cub2, i);
        data[e] = std::min(data[cub1], data[cub2]);
      }
    }
    std::clog << "fill 1: " << clock; clock.begin();
    for(std::size_t y = 1; y < 2 * size_y; ++y) {
      for(std::size_t x = 0; x < size_x; ++x) {
        std::size_t ref = dy * y + 2 * x;
        std::size_t i = ref + 1;
        data[i] = std::min(data[ref], data[ref + 2]);
      }
    }
    // FIXME: is that robust enough?
    data[0].first = std::numeric_limits<Filtration_value>::infinity();
    std::clog << "fill 2: " << clock; clock.begin();
#ifdef DEBUG_TRACES
    std::clog << "data\n";
    for(std::size_t i = 0; i < data_size; ++i) {
      std::clog << data[i].first << '|' << data[i].second << '\t';
      if ((i+1)%dy == 0)
        std::clog << '\n';
    }
#endif
  }
  // We don't need the filtration value here, only compare ids.
  auto id(std::size_t i){return data[i].second;};
  void pair_internal(){
    // Internal cubes
    for(std::size_t y = 1; y < size_y; ++y) {
      for(std::size_t x = 1; x < size_x; ++x) {
        // TODO: see if testing edges first helps
        std::size_t cub = 2 * x + dy * 2 * y;
        auto ff = data[cub];
        auto f = ff.second;
        bool edge_used[4] = { false, false, false, false };
        bool cub_paired = false;
        int nv = 0;
        if (f == id(cub+dy-1)) {
          ds_parent(cub+dy-1) = cub+dy+1;
          edge_used[0] = true;
          ++nv;
        }
        if (f == id(cub+dy+1)) {
          ds_parent(cub+dy+1) = cub-dy+1;
          edge_used[1] = true;
          ++nv;
        }
        if (f == id(cub-dy+1)) {
          ds_parent(cub-dy+1) = cub-dy-1;
          edge_used[2] = true;
          ++nv;
        }
        if (f == id(cub-dy-1)) {
          if (nv < 3) {
            ds_parent(cub-dy-1) = cub+dy-1;
            edge_used[3] = true;
            ++nv;
          } else {
            // the whole neighborhood has value f
            ds_parent(cub-dy-1) = cub-dy-1;
            ds_parent(cub) = cub - 2;
            continue;
          }
        }
        if (!edge_used[0] && f == id(cub+dy)) {
          if (!cub_paired) {
            ds_parent(cub) = cub + 2 * dy;
            cub_paired = true;
          } else {
            edges.emplace_back(ff, cub + dy - 1, cub + dy + 1);
          }
        }
        if (!edge_used[1] && f == id(cub+1)) {
          if (!cub_paired) {
            ds_parent(cub) = cub + 2;
            cub_paired = true;
          } else {
            edges.emplace_back(ff, cub - dy + 1, cub + dy + 1);
          }
        }
        if (!edge_used[2] && f == id(cub-dy)) {
          if (!cub_paired) {
            ds_parent(cub) = cub - 2 * dy;
            cub_paired = true;
          } else {
            edges.emplace_back(ff, cub - dy - 1, cub - dy + 1);
          }
        }
        if (!edge_used[3] && f == id(cub-1)) {
          if (!cub_paired) {
            ds_parent(cub) = cub - 2;
            cub_paired = true;
          } else {
            edges.emplace_back(ff, cub - dy - 1, cub + dy - 1);
          }
        }
        if (!cub_paired) {
          ds_parent(cub) = cub;
        }
      }
    }
    std::clog << "pair internal: " << clock; clock.begin();
  }
  void pair_boundary(){
    // Boundary nodes
    for(std::size_t x = 1; x < size_x; ++x) {
      std::size_t cub = 2 * x;
      auto ff = data[cub];
      auto f = ff.second;
      if (f == id(cub+dy-1)) {
        ds_parent(cub+dy-1) = cub+dy+1;
        if (f == id(cub+dy+1)) {
          ds_parent(cub+dy+1) = cub+dy+1;
        }
      } else if (f == id(cub+dy+1)) {
        ds_parent(cub+dy+1) = cub+dy-1;
      } else if (f == id(cub+dy)) {
        edges.emplace_back(ff, cub+dy-1, cub+dy+1);
      }
    }
    for(std::size_t x = 1; x < size_x; ++x) {
      std::size_t cub = 2 * x + 2 * dy * size_y;
      auto ff = data[cub];
      auto f = ff.second;
      if (f == id(cub-dy-1)) {
        ds_parent(cub-dy-1) = cub-dy+1;
        if (f == id(cub-dy+1)) {
          ds_parent(cub-dy+1) = cub-dy+1;
        }
      } else if (f == id(cub-dy+1)) {
        ds_parent(cub-dy+1) = cub-dy-1;
      } else if (f == id(cub-dy)) {
        edges.emplace_back(ff, cub-dy-1, cub-dy+1);
      }
    }
    for(std::size_t y = 1; y < size_y; ++y) {
      std::size_t cub = 2 * dy * y;
      auto ff = data[cub];
      auto f = ff.second;
      if (f == id(cub+dy+1)) {
        ds_parent(cub+dy+1) = cub-dy+1;
        if (f == id(cub-dy+1)) {
          ds_parent(cub-dy+1) = cub-dy+1;
        }
      } else if (f == id(cub-dy+1)) {
        ds_parent(cub-dy+1) = cub+dy+1;
      } else if (f == id(cub+1)) {
        edges.emplace_back(ff, cub-dy+1, cub+dy+1);
      }
    }
    for(std::size_t y = 1; y < size_y; ++y) {
      std::size_t cub = 2 * size_x + 2 * dy * y;
      auto ff = data[cub];
      auto f = ff.second;
      if (f == id(cub+dy-1)) {
        ds_parent(cub+dy-1) = cub-dy-1;
        if (f == id(cub-dy-1)) {
          ds_parent(cub-dy-1) = cub-dy-1;
        }
      } else if (f == id(cub-dy-1)) {
        ds_parent(cub-dy-1) = cub+dy-1;
      } else if (f == id(cub-1)) {
        edges.emplace_back(ff, cub-dy-1, cub+dy-1);
      }
    }
    // Corners
    std::size_t vc = 0;
    std::size_t vi = dy + 1;
    if (id(vc) == id(vi)) {
      ds_parent(vi) = vi;
    }
    vc = 2 * size_x;
    vi = vc + dy - 1;
    if (id(vc) == id(vi)) {
      ds_parent(vi) = vi;
    }
    vc = 2 * dy * size_y;
    vi = vc - dy + 1;
    if (id(vc) == id(vi)) {
      ds_parent(vi) = vi;
    }
    vc = 2 * size_x + 2 * dy * size_y;
    vi = vc - dy - 1;
    if (id(vc) == id(vi)) {
      ds_parent(vi) = vi;
    }
    std::clog << "pair boundary: " << clock; clock.begin();
#ifdef DEBUG_TRACES
    std::clog << "ds_parent after pairing\n";
    for(std::size_t i = 0; i < ds_parent_.size(); ++i) {
      std::clog << (2 * i) << '\t' << ds_parent_[i] << '\n';
    }
#endif
  }
  bool has_larger_input(std::size_t a, std::size_t b, Filtration_value fb) const {
    // Is passing fb useful, or would the compiler notice that it already has it available?
    GUDHI_CHECK(a != b, std::logic_error("Bug in Gudhi"));
    Filtration_value fa = input(a);
    if (fa > fb) return true;
    if (fa < fb) return false;
    return a > b;
  }

  // TODO: split numbering of vertices / squares (no need for edges), split data and ds_parent
  // Locally pair simplices around each square.
  // Work implicitly from input, only store the filtration value of critical vertices and squares.
  // Store critical edges for later processing.
  void fill_and_pair() {
    data[0] = T{std::numeric_limits<Filtration_value>::infinity(), 0};
    const std::size_t dy_input = size_x + 1;
    std::size_t cub; // Index of the current square in the full complex
    std::size_t i;   // Index of the current square in the input
    Filtration_value f; // input(i)
    auto mark_vertex_critical = [&](std::size_t c) {
      // Also set data?
      ds_parent(c) = c;
      data[c] = T{f, i};
    };
    auto mark_square_critical = [&]() {
      ds_parent(cub) = cub;
      data[cub] = T{f, i};
    };
    auto mark_edge_critical = [&](std::size_t v1, std::size_t v2) {
      edges.emplace_back(T{f, i}, v1, v2);
    };
    auto set_parent = [&](std::size_t child, std::size_t parent) {
      GUDHI_CHECK(child != parent, std::logic_error("Bug in Gudhi: use mark_*_critical instead of set_parent"));
      ds_parent(child) = parent;
    };
    // Mark the corners as critical, it will be overwritten if not
    cub = 0; i = 0; f = input(i);
    mark_vertex_critical(dy + 1);
    cub = 2 * size_x; i = size_x; f = input(i);
    mark_vertex_critical(cub + dy - 1);
    cub = 2 * dy * size_y; i = dy_input * size_y; f = input(i);
    mark_vertex_critical(cub - dy + 1);
    cub = 2 * size_x + 2 * dy * size_y; i = size_x + dy_input * size_y; f = input(i);
    mark_vertex_critical(cub - dy - 1);

    // Boundary nodes, 1st row
    for(std::size_t x = 1; x < size_x; ++x) {
      cub = 2 * x;
      i = x;
      f = input(x);
      if (has_larger_input(i + dy_input, i, f)) {
        bool ul = has_larger_input(i - 1, i, f) && has_larger_input(i + dy_input - 1, i, f);
        bool ur = has_larger_input(i + 1, i, f) && has_larger_input(i + dy_input + 1, i, f);
        if (ul) {
          set_parent(cub + dy - 1, cub + dy + 1);
          if (ur) mark_vertex_critical(cub + dy + 1);
        } else if (ur) {
          set_parent(cub + dy + 1, cub + dy - 1);
        } else {
          mark_edge_critical(cub + dy - 1, cub + dy + 1);
        }
      }
    }
    // Internal rows
    for(std::size_t y = 1; y < size_y; ++y) {
      // First column
      {
        cub = 2 * dy * y;
        i = y * dy_input;
        f = input(i);
        if (has_larger_input(i + 1, i, f)) {
          bool dr = has_larger_input(i - dy_input, i, f) && has_larger_input(i + 1 - dy_input, i, f);
          bool ur = has_larger_input(i + dy_input, i, f) && has_larger_input(i + 1 + dy_input, i, f);
          if (dr) {
            set_parent(cub - dy + 1, cub + dy + 1);
            if (ur) mark_vertex_critical(cub + dy + 1);
          } else if (ur) {
            set_parent(cub + dy + 1, cub - dy + 1);
          } else {
            mark_edge_critical(cub - dy + 1, cub + dy + 1);
          }
        }
      }
      // Internal cubes
      for(std::size_t x = 1; x < size_x; ++x) {
        cub = 2 * x + 2 * dy * y;
        i = x + dy_input * y;
        f = input(i);
        // See what part of the boundary shares f
        bool l = has_larger_input(i - 1, i, f);
        bool r = has_larger_input(i + 1, i, f);
        bool d = has_larger_input(i - dy_input, i, f);
        bool u = has_larger_input(i + dy_input, i, f);
        // TODO: achieve the same laziness with lambdas
#define dl has_larger_input(i - dy_input - 1, i, f)
#define ul has_larger_input(i + dy_input - 1, i, f)
#define dr has_larger_input(i - dy_input + 1, i, f)
#define ur has_larger_input(i + dy_input + 1, i, f)
        if (u) { // u
          if (l) { // u l
            if (ul) { // u l ul
              set_parent(cub + dy - 1, cub + dy + 1);
              if (d) { // U l UL d
                if (dl) { // U l UL d dl
                  set_parent(cub - dy - 1, cub + dy - 1);
                  if (r) { // U L UL d DL r
                    if (dr) { // U L UL d DL r dr
                      set_parent(cub - dy + 1, cub - dy - 1);
                      set_parent(cub, cub + 2);
                      if (ur) { // U L UL D DL R DR ur - cr
                        mark_vertex_critical(cub + dy + 1);
                      }
                    } else { // U L UL d DL r !dr
                      set_parent(cub, cub - 2 * dy);
                      if (ur) { // U L UL D DL r !dr ur - cd
                        set_parent(cub + dy + 1, cub - dy + 1);
                      } else { // U L UL D DL r !dr !ur - cd
                        mark_edge_critical(cub - dy + 1, cub + dy + 1);
                      }
                    }
                  } else { // U L UL d DL !r
                    set_parent(cub, cub - 2 * dy); // cd
                  }
                } else { // U l UL d !dl
                  set_parent(cub, cub - 2);
                  if (r) { // U L UL d !dl r - cl
                    if (dr) { // U L UL d !dl r dr - cl
                      set_parent(cub - dy + 1, cub - dy - 1);
                    } else { // U L UL d !dl r !dr - cl
                      mark_edge_critical(cub - dy - 1, cub - dy + 1);
                    }
                    if (ur) { // U L UL D !dl r ur - cl
                      set_parent(cub + dy + 1, cub - dy + 1);
                    } else { // U L UL D !dl r !ur - cl
                      mark_edge_critical(cub - dy + 1, cub + dy + 1);
                    }
                  } else { // U L UL d !dl !r - cl
                    mark_edge_critical(cub - dy - 1, cub - dy + 1);
                  }
                }
              } else { // U l UL !d
                set_parent(cub, cub - 2);
                if (r) { // U L UL !d r - cl
                  if (ur) { // U L UL !d r ur - cl
                    set_parent(cub + dy + 1, cub - dy + 1);
                  } else { // U L UL !d r !ur - cl
                    mark_edge_critical(cub - dy + 1, cub + dy + 1);
                  }
                } else {} // U L UL !d !r - cl
              }
            } else { // u l !ul
              set_parent(cub, cub + 2 * dy);
              if (d) { // U l !ul d - cu
                if (dl) { // U l !ul d dl - cu
                  set_parent(cub - dy - 1, cub + dy - 1);
                } else { // U l !ul d !dl - cu
                  mark_edge_critical(cub - dy - 1, cub + dy - 1);
                }
                if (r) { // U L !ul d r - cu
                  if (dr) { // U L !ul d r dr - cu
                    set_parent(cub - dy + 1, cub - dy - 1);
                  } else { // U L !ul d r !dr - cu
                    mark_edge_critical(cub - dy - 1, cub - dy + 1);
                  }
                  if (ur) { // U L !ul D r ur - cu
                    set_parent(cub + dy + 1, cub - dy + 1);
                  } else { // U L !ul D r !ur - cu
                    mark_edge_critical(cub - dy + 1, cub + dy + 1);
                  }
                } else { // U L !ul d !r - cu
                  mark_edge_critical(cub - dy - 1, cub - dy + 1);
                }
              } else { // U l !ul !d - cu
                mark_edge_critical(cub - dy - 1, cub + dy - 1);
                if (r) { // U L !ul !d r - cu
                  if (ur) { // U L !ul !d r ur - cu
                    set_parent(cub + dy + 1, cub - dy + 1);
                  } else { // U L !ul !d r !ur - cu
                    mark_edge_critical(cub - dy + 1, cub + dy + 1);
                  }
                } else {} // U L !ul !d !r - cu
              }
            }
          } else { // u !l
            set_parent(cub, cub + 2 * dy);
            if (d) { // U !l d - cu
              if (r) { // U !l d r - cu
                if (dr) { // U !l d r dr - cu
                  set_parent(cub - dy + 1, cub - dy - 1);
                } else { // U !l d r !dr - cu
                  mark_edge_critical(cub - dy - 1, cub - dy + 1);
                }
                if (ur) { // U !l D r ur - cu
                  set_parent(cub + dy + 1, cub - dy + 1);
                } else { // U !l D r !ur - cu
                  mark_edge_critical(cub - dy + 1, cub + dy + 1);
                }
              } else { // U !l d !r - cu
                mark_edge_critical(cub - dy - 1, cub - dy + 1);
              }
            } else { // U !l !d - cu
              if (r) { // U !l !d r - cu
                if (ur) { // U !l !d r ur - cu
                  set_parent(cub + dy + 1, cub - dy + 1);
                } else { // U !l !d r !ur - cu
                  mark_edge_critical(cub - dy + 1, cub + dy + 1);
                }
              } else {} // U !l !d !r - cu
            }
          }
        } else { // !u
          if (l) { // !u l
            if (d) { // !u l d
              if (dl) { // !u l d dl
                set_parent(cub - dy - 1, cub + dy - 1);
                if (r) { // !u L d DL r
                  if (dr) { // !u L d DL r dr
                    set_parent(cub - dy + 1, cub - dy - 1);
                  } else { // !u L d DL r !dr
                    mark_edge_critical(cub - dy - 1, cub - dy + 1);
                  }
                  set_parent(cub, cub + 2); // cr
                } else { // !u L d DL !r
                  set_parent(cub, cub - 2 * dy); // cd
                }
              } else { // !u l d !dl
                set_parent(cub, cub - 2);
                if (r) { // !u L d !dl r - cl
                  if (dr) { // !u L d !dl r dr - cl
                    set_parent(cub - dy + 1, cub - dy - 1);
                  } else { // !u L d !dl r !dr - cl
                    mark_edge_critical(cub - dy - 1, cub - dy + 1);
                  }
                  mark_edge_critical(cub - dy + 1, cub + dy + 1);
                } else { // !u L d !dl !r - cl
                  mark_edge_critical(cub - dy - 1, cub - dy + 1);
                }
              }
            } else { // !u l !d
              set_parent(cub, cub - 2);
              if (r) { // !u L !d r - cl
                mark_edge_critical(cub - dy + 1, cub + dy + 1);
              } else {} // !u L !d !r - cl
            }
          } else { // !u !l
            if (d) { // !u !l d
              set_parent(cub, cub - 2 * dy);
              if (r) { // !u !l D r - cd
                if (dr) { // !u !l D r dr - cd
                  set_parent(cub - dy + 1, cub + dy + 1);
                } else { // !u !l D r !dr - cd
                  mark_edge_critical(cub - dy + 1, cub + dy + 1);
                }
              } else {} // !u !l D !r - cd
            } else { // !u !l !d
              if (r) { // !u !l !d r
                set_parent(cub, cub + 2); // cr
              } else { // !u !l !d !r
                mark_square_critical();
              }
            }
          }
        }
      }
#undef dl
#undef ul
#undef dr
#undef ur
      // Last column
      {
        cub = 2 * size_x + 2 * dy * y;
        i = size_x + dy_input * y;
        f = input(i);
        if (has_larger_input(i - 1, i, f)) {
          bool dl = has_larger_input(i - dy_input, i, f) && has_larger_input(i - 1 - dy_input, i, f);
          bool ul = has_larger_input(i + dy_input, i, f) && has_larger_input(i - 1 + dy_input, i, f);
          if (dl) {
            set_parent(cub - dy - 1, cub + dy - 1);
            if (ul) mark_vertex_critical(cub + dy - 1);
          } else if (ul) {
            set_parent(cub + dy - 1, cub - dy - 1);
          } else {
            mark_edge_critical(cub - dy - 1, cub + dy - 1);
          }
        }
      }
    }
    // Boundary nodes, last row
    for(std::size_t x = 1; x < size_x; ++x) {
      cub = 2 * x + 2 * dy * size_y;
      i = size_y * dy_input + x;
      f = input(i);
      if (has_larger_input(i - dy_input, i, f)) {
        bool dl = has_larger_input(i - 1, i, f) && has_larger_input(i - dy_input - 1, i, f);
        bool dr = has_larger_input(i + 1, i, f) && has_larger_input(i - dy_input + 1, i, f);
        if (dl) {
          set_parent(cub - dy - 1, cub - dy + 1);
          if (dr) mark_vertex_critical(cub - dy + 1);
        } else if (dr) {
          set_parent(cub - dy + 1, cub - dy - 1);
        } else {
          mark_edge_critical(cub - dy - 1, cub - dy + 1);
        }
      }
    }

#ifdef DEBUG_TRACES
    std::clog << "data\n";
    for(std::size_t i = 0; i < data_size; ++i) {
      std::clog << data[i].first << '|' << data[i].second << '\t';
      if ((i+1)%dy == 0)
        std::clog << '\n';
    }
#endif
#ifdef DEBUG_TRACES
    std::clog << "ds_parent after pairing\n";
    for(std::size_t i = 0; i < ds_parent_.size(); ++i) {
      std::clog << (2 * i) << '\t' << ds_parent_[i] << '\n';
    }
#endif
    std::clog << "fill and pair: " << clock; clock.begin();
  }
  void sort_edges(){
#ifdef GUDHI_USE_TBB
    tbb::parallel_sort(edges.begin(), edges.end());
#else
    std::sort(edges.begin(), edges.end());
#endif
    std::clog << "sort: " << clock; clock.begin();
#ifdef DEBUG_TRACES
    std::clog << "edges\n";
    for(auto&e : edges){ std::clog << e.v1 << '\t' << e.v2 << '\t' << e.filt() << '\t' << e.f.second << '\n'; }
#endif
  }
  template<class Out>
  void primal(Out&&out){
    auto it = std::remove_if(edges.begin(), edges.end(), [&](Edge& e) {
        assert(e.v1 < e.v2);
        std::size_t a = ds_find_set(e.v1);
        std::size_t b = ds_find_set(e.v2);
#ifdef DEBUG_TRACES
        std::clog << "processing edge " << e.v1 << '-' << e.v2 << " : " << a << '-' << b << '\n';
#endif
        if (a == b) return false;
        if (data[b] < data[a]) std::swap(a, b);
        ds_parent(b) = a;
        out(data[b].first, e.filt());
        return true;
    });
    edges.erase(it, edges.end());
    std::clog << "primal pass: " << clock; clock.begin();
  }
  template<class Out>
  void dual(Out&&out){
    for (auto e : boost::adaptors::reverse(edges)) {
#ifdef DEBUG_TRACES
      std::clog << "reprocessing edge " << e.v1 << '-' << e.v2 << '\n';
#endif
      dualize_edge(e);
      std::size_t a = ds_find_set(e.v1);
      std::size_t b = ds_find_set(e.v2);
#ifdef DEBUG_TRACES
      std::clog << "i.e. dual edge " << e.v1 << '-' << e.v2 << " : " << a << '-' << b << '\n';
#endif
      GUDHI_CHECK(a != b, std::logic_error("Bug in Gudhi"));
      // We could check here if a or b is 0, it would be more robust in case the input contains inf.
      // if (b == 0 || (a != 0 && lt_data()(data[a], data[b]))) std::swap(a, b);
      if (data[a] < data[b]) std::swap(a, b);
      ds_parent(b) = a;
      out(e.filt(), data[b].first);
    }
    std::clog << "dual pass: " << clock;
  }
  auto global_min(){
    return data[ds_find_set(dy + 1)].first;
  }
};

template <typename U, typename Out>
U persistence_on_rectangle(const std::vector<unsigned>& dimensions, const std::vector<U>& input, Out&&out){
  Persistence_on_rectangle<U> X;
  X.init(dimensions, input);
  //X.fill_data_from_input();
  //X.pair_internal();
  //X.pair_boundary();
  X.fill_and_pair(); // bogus
  X.sort_edges();
  X.primal(out);
  X.dual(out);
  //data[0] = save_data_0;
  return X.global_min();
}

}  // namespace Gudhi

#endif  // PERSISTENCE_ON_RECTANGLE_H
