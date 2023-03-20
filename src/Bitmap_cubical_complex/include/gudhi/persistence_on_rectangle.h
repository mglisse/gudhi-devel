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
template <typename U, typename Lt, typename Out>
U persistence_on_rectangle(const std::vector<unsigned>& dimensions, const std::vector<U>& input, Lt&&lt_filt, Out&&out){
#ifdef DEBUG_TRACES
  std::clog << "input\n";
  for(std::size_t i=0;i<input.size();++i) {
    std::clog << i << '\t' << input[i] << '\n';
  }
#endif
  Gudhi::Clock clock;

  GUDHI_CHECK(dimensions.size() == 2, std::logic_error("persistence_2d_dual() only works on 2-dimensional complexes"));
  GUDHI_CHECK(dimensions[0] * dimensions[1] == input.size(),
      std::invalid_argument("Number of cells inconsistent with dimensions"));
  const std::vector<std::size_t> sizes { dimensions[0] - 1, dimensions[1] - 1 };
  const std::size_t dy = 2 * sizes[0] + 1;
  const std::size_t data_size = dy * (2 * sizes[1] + 1);
  typedef std::pair<U,std::size_t> T;
  std::unique_ptr<T[]> data(new T[data_size]); // std::make_unique_for_overwrite
  //std::vector<T> data(data_size, std::pair(std::numeric_limits<U>::infinity(), -1)); // TMP for debugging

  // We only need this for vertices and squares. Since edges and non-edges alternate, we can use n/2 as index.
  // Only parent is needed on all nodes, rank and birth are only meaningful on cluster representatives. We could
  // store them in an unordered_map, but this would only save memory if we interleave vertex and edge insertion,
  // which is more complicated and probably slower.
  struct Pers2d_cluster_data {
    std::size_t parent;
    std::size_t rank;
    // The rank heuristic in union-find means that the representative may not be the same as defined by persistent
    // homology, so we store this one as well.
    T birth;
  };  // information on a cluster
  std::vector<Pers2d_cluster_data> ds_base((data_size + 1) / 2); // TODO: tighten this number a bit
  // boost::vector_property_map does resize(size+1) for every new element, don't use it
  auto ds_data =
      boost::make_function_property_map<std::size_t>([&ds_base](std::size_t n) -> Pers2d_cluster_data&
          { return ds_base[n/2]; }); // n is already even
  auto ds_parent =
      boost::make_transform_value_property_map([](auto& p) -> std::size_t& { return p.parent; }, ds_data);
  auto ds_rank = boost::make_transform_value_property_map([](auto& p) -> std::size_t& { return p.rank; }, ds_data);
  auto ds_birth = boost::make_transform_value_property_map([](auto& p) -> T& { return p.birth; }, ds_data);
  boost::disjoint_sets<decltype(ds_rank), decltype(ds_parent)> ds(ds_rank, ds_parent);
  // Everything has rank 0 and has cell 0 (the infinite exterior cell) as representative by default.
  // Real vertices/squares should be their own cluster at the beginning.
  ds_rank[0] = 1;

  std::clog << "debut: " << clock; clock.begin();

  struct Edge {
    T f;
    std::size_t v1, v2; // v1 < v2
    Edge(T f, std::size_t v1, std::size_t v2) : f(f), v1(v1), v2(v2) {}
    bool operator<(Edge const& other) const { return f.first < other.f.first; }
  };
  auto dualize_edge = [diag = dy + 1](Edge& e) {
    std::size_t new_v2 = e.v1 + diag;
    e.v1 = e.v2 - diag;
    e.v2 = new_v2;
  };
  std::vector<Edge> edges; edges.reserve(data_size / 2); // TODO: tighten this number a bit

  // TODO: build many local pairs and omit them from the list of edges.

  for(std::size_t x = 0; x < sizes[0] + 1; ++x) {
    data[2 * x] = std::pair(input[x], 2 * x); // FIXME: use index in input instead?
    for(std::size_t y = 0; y < sizes[1]; ++y) {
      std::size_t cub1 = 2 * x + dy * 2 * y;
      std::size_t e = cub1 + dy;
      std::size_t cub2 = e + dy;
      data[cub2] = std::pair(input[x + (sizes[0] + 1) * (y + 1)], cub2);
      data[e] = std::min(data[cub1], data[cub2]);
    }
  }
  std::clog << "fill 1: " << clock; clock.begin();
  for(std::size_t y = 1; y < 2 * sizes[1]; ++y) {
    for(std::size_t x = 0; x < sizes[0]; ++x) {
      std::size_t ref = dy * y + 2 * x;
      std::size_t i = ref + 1;
      data[i] = std::min(data[ref], data[ref + 2]);
    }
  }
  std::clog << "fill 2: " << clock; clock.begin();
#ifdef DEBUG_TRACES
  std::clog << "data\n";
  for(std::size_t i = 0; i < data_size; ++i) {
    std::clog << data[i].first << '|' << data[i].second << '\t';
    if ((i+1)%dy == 0)
      std::clog << '\n';
  }
#endif

  // We don't need the filtration value here, only compare ids.
  auto id = [&data](std::size_t i){return data[i].second;};
  // Internal cubes
  for(std::size_t x = 1; x < sizes[0]; ++x) {
    for(std::size_t y = 1; y < sizes[1]; ++y) {
      std::size_t cub = 2 * x + dy * 2 * y;
      auto ff = data[cub];
      auto f = ff.second;
      bool edge_used[4] = { false, false, false, false };
      bool cub_paired = false;
      int nv = 0;
      if (f == id(cub+dy-1)) {
        ds_parent[cub+dy-1] = cub+dy+1;
        edge_used[0] = true;
        ++nv;
      }
      if (f == id(cub+dy+1)) {
        ds_parent[cub+dy+1] = cub-dy+1;
        edge_used[1] = true;
        ++nv;
      }
      if (f == id(cub-dy+1)) {
        ds_parent[cub-dy+1] = cub-dy-1;
        edge_used[2] = true;
        ++nv;
      }
      if (f == id(cub-dy-1)) {
        if (nv < 3) {
          ds_parent[cub-dy-1] = cub+dy-1;
          edge_used[3] = true;
          ++nv;
        } else {
          // the whole neighborhood has value f
          ds_parent[cub-dy-1] = cub-dy-1;
          ds_birth[cub-dy-1] = data[cub-dy-1];
          ds_parent[cub] = cub - 2;
          continue;
        }
      }
      if (!edge_used[0] && f == id(cub+dy)) {
        if (!cub_paired) {
          ds_parent[cub] = cub + 2 * dy;
          cub_paired = true;
        } else {
          edges.emplace_back(ff, cub + dy - 1, cub + dy + 1);
        }
      }
      if (!edge_used[1] && f == id(cub+1)) {
        if (!cub_paired) {
          ds_parent[cub] = cub + 2;
          cub_paired = true;
        } else {
          edges.emplace_back(ff, cub - dy + 1, cub + dy + 1);
        }
      }
      if (!edge_used[2] && f == id(cub-dy)) {
        if (!cub_paired) {
          ds_parent[cub] = cub - 2 * dy;
          cub_paired = true;
        } else {
          edges.emplace_back(ff, cub - dy - 1, cub - dy + 1);
        }
      }
      if (!edge_used[3] && f == id(cub-1)) {
        if (!cub_paired) {
          ds_parent[cub] = cub - 2;
          cub_paired = true;
        } else {
          edges.emplace_back(ff, cub - dy - 1, cub + dy - 1);
        }
      }
      if (!cub_paired) {
        ds_parent[cub] = cub;
        ds_birth[cub] = data[cub];
      }
    }
  }
  // Boundary nodes
  for(std::size_t x = 1; x < sizes[0]; ++x) {
    std::size_t cub = 2 * x;
    auto ff = data[cub];
    auto f = ff.second;
    if (f == id(cub+dy-1)) {
      ds_parent[cub+dy-1] = cub+dy+1;
      if (f == id(cub+dy+1)) {
        ds_parent[cub+dy+1] = cub+dy+1;
        ds_birth[cub+dy+1] = data[cub+dy+1];
      }
    } else if (f == id(cub+dy+1)) {
      ds_parent[cub+dy+1] = cub+dy-1;
    } else if (f == id(cub+dy)) {
      edges.emplace_back(ff, cub+dy-1, cub+dy+1);
    }
  }
  for(std::size_t x = 1; x < sizes[0]; ++x) {
    std::size_t cub = 2 * x + 2 * dy * sizes[1];
    auto ff = data[cub];
    auto f = ff.second;
    if (f == id(cub-dy-1)) {
      ds_parent[cub-dy-1] = cub-dy+1;
      if (f == id(cub-dy+1)) {
        ds_parent[cub-dy+1] = cub-dy+1;
        ds_birth[cub-dy+1] = data[cub-dy+1];
      }
    } else if (f == id(cub-dy+1)) {
      ds_parent[cub-dy+1] = cub-dy-1;
    } else if (f == id(cub-dy)) {
      edges.emplace_back(ff, cub-dy-1, cub-dy+1);
    }
  }
  for(std::size_t y = 1; y < sizes[1]; ++y) {
    std::size_t cub = 2 * dy * y;
    auto ff = data[cub];
    auto f = ff.second;
    if (f == id(cub+dy+1)) {
      ds_parent[cub+dy+1] = cub-dy+1;
      if (f == id(cub-dy+1)) {
        ds_parent[cub-dy+1] = cub-dy+1;
        ds_birth[cub-dy+1] = data[cub-dy+1];
      }
    } else if (f == id(cub-dy+1)) {
      ds_parent[cub-dy+1] = cub+dy+1;
    } else if (f == id(cub+1)) {
      edges.emplace_back(ff, cub-dy+1, cub+dy+1);
    }
  }
  for(std::size_t y = 1; y < sizes[1]; ++y) {
    std::size_t cub = 2 * sizes[0] + 2 * dy * y;
    auto ff = data[cub];
    auto f = ff.second;
    if (f == id(cub+dy-1)) {
      ds_parent[cub+dy-1] = cub-dy-1;
      if (f == id(cub-dy-1)) {
        ds_parent[cub-dy-1] = cub-dy-1;
        ds_birth[cub-dy-1] = data[cub-dy-1];
      }
    } else if (f == id(cub-dy-1)) {
      ds_parent[cub-dy-1] = cub+dy-1;
    } else if (f == id(cub-1)) {
      edges.emplace_back(ff, cub-dy-1, cub+dy-1);
    }
  }
  // Corners
  std::size_t vc = 0;
  std::size_t vi = dy + 1;
  if (id(vc) == id(vi)) {
    ds_parent[vi] = vi;
    ds_birth[vi] = data[vi];
  }
  vc = 2 * sizes[0];
  vi = vc + dy - 1;
  if (id(vc) == id(vi)) {
    ds_parent[vi] = vi;
    ds_birth[vi] = data[vi];
  }
  vc = 2 * dy * sizes[1];
  vi = vc - dy + 1;
  if (id(vc) == id(vi)) {
    ds_parent[vi] = vi;
    ds_birth[vi] = data[vi];
  }
  vc = 2 * sizes[0] + 2 * dy * sizes[1];
  vi = vc - dy - 1;
  if (id(vc) == id(vi)) {
    ds_parent[vi] = vi;
    ds_birth[vi] = data[vi];
  }
  // FIXME: the ranks are completely bogus :-( But using link instead of setting parent would likely be more expensive.

  std::clog << "pairing: " << clock; clock.begin();
#ifdef GUDHI_USE_TBB
  tbb::parallel_sort(edges.begin(), edges.end());
#else
  std::sort(edges.begin(), edges.end());
#endif
  std::clog << "sort: " << clock; clock.begin();
#ifdef DEBUG_TRACES
  std::clog << "edges\n";
  for(auto&e : edges){ std::clog << e.v1 << '\t' << e.v2 << '\t' << e.f.first << '\t' << e.f.second << '\n'; }
#endif

  // FIXME: is that robust enough? At least make it an argument to the function.
  ds_birth[0].first = std::numeric_limits<U>::infinity();

#ifdef DEBUG_TRACES
  std::clog << "ds_data after pairing\n";
  for(std::size_t i=0;i<ds_base.size();++i) {
    auto&dat=ds_base[i];
    std::clog << i << '\t' << dat.parent << '\t' << dat.birth.first << '\t' << dat.birth.second << '\n';
  }
#endif

  //T save_data_0 = data[0]; data[0] = std::numeric_limits<T>::infinity();
  auto it = std::remove_if(edges.begin(), edges.end(), [&](Edge& e) {
      std::size_t a = ds.find_set(e.v1);
      std::size_t b = ds.find_set(e.v2);
#ifdef DEBUG_TRACES
      std::clog << "processing edge " << e.v1 << '-' << e.v2 << " : " << a << '-' << b << '\n';
#endif
      if (a == b) return false;
      if (ds_birth[b] < ds_birth[a]) std::swap(a, b);
      // ds.link(a, b); std::size_t newrep = ds.find_set(a);
      std::size_t rank_a = ds_rank[a];
      std::size_t& rank_b = ds_rank[b];
      std::size_t newrep = a;
      if (rank_a > rank_b) ds_parent[b] = a;
      else {
        ds_parent[a] = b;
        newrep = b;
        if (rank_a == rank_b) ++rank_b;
      }
      out(ds_birth[b].first, e.f.first);
      ds_birth[newrep] = ds_birth[a];
      return true;
      });
  edges.erase(it, edges.end());
  std::clog << "primal pass: " << clock; clock.begin();
  for (auto e : boost::adaptors::reverse(edges)) {
#ifdef DEBUG_TRACES
    std::clog << "reprocessing edge " << e.v1 << '-' << e.v2 << '\n';
#endif
    dualize_edge(e);
    std::size_t a = ds.find_set(e.v1);
    std::size_t b = ds.find_set(e.v2);
#ifdef DEBUG_TRACES
    std::clog << "i.e. dual edge " << e.v1 << '-' << e.v2 << " : " << a << '-' << b << '\n';
#endif
    GUDHI_CHECK(a != b, std::logic_error("Bug in Gudhi"));
    // We could check here if a or b is 0.
    if (ds_birth[a] < ds_birth[b]) std::swap(a, b);
    // ds.link(a, b); std::size_t newrep = ds.find_set(a);
    std::size_t rank_a = ds_rank[a];
    std::size_t& rank_b = ds_rank[b];
    std::size_t newrep = a;
    if (rank_a > rank_b) ds_parent[b] = a;
    else {
      ds_parent[a] = b;
      newrep = b;
      if (rank_a == rank_b) ++rank_b;
    }
    out(e.f.first, ds_birth[b].first);
    ds_birth[newrep] = ds_birth[a];
  }
  std::clog << "dual pass: " << clock;

  //data[0] = save_data_0;
  return ds_birth[ds.find_set(dy + 1)].first;
}

}  // namespace Gudhi

#endif  // PERSISTENCE_ON_RECTANGLE_H
