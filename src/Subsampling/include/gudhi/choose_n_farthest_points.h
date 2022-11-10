/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Siargey Kachanovich
 *
 *    Copyright (C) 2016 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef CHOOSE_N_FARTHEST_POINTS_H_
#define CHOOSE_N_FARTHEST_POINTS_H_

#include <boost/range.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include <boost/unordered_set.hpp> // preferably with boost 1.79+ for Fibonacci hashing

#include <gudhi/Null_output_iterator.h>

#include <iterator>
#include <vector>
#include <set>
#include <random>
#include <limits>  // for numeric_limits<>

namespace Gudhi {

namespace subsampling {

/**
 *  \ingroup subsampling
 */
enum : std::size_t {
/**
 *  Argument for `choose_n_farthest_points` to indicate that the starting point should be picked randomly.
 */
  random_starting_point = std::size_t(-1)
};

/** 
 *  \ingroup subsampling
 *  \brief Subsample by a greedy strategy of iteratively adding the farthest point from the
 *  current chosen point set to the subsampling. 
 *  \details
 *  The iteration starts with the landmark `starting point` or, if `starting point==random_starting_point`,
 *  with a random landmark.
 *  It chooses `final_size` points from a random access range
 *  `input_pts` (or the number of input points if `final_size` is larger)
 *  and outputs them in the output iterator `output_it`. It also
 *  outputs the distance from each of those points to the set of previous
 *  points in `dist_it`.
 *  \tparam Distance must provide an operator() that takes 2 points (value type of the range)
 *  and returns their distance as a `double`. It must be a true metric, the algorithm relies on the triangle inequality.
 *  \tparam Point_range Random access range of points.
 *  \tparam PointOutputIterator Output iterator whose value type is the point type.
 *  \tparam DistanceOutputIterator Output iterator for distances.
 * @param[in] dist A distance function.
 * @param[in] input_pts The input points.
 * @param[in] final_size The size of the subsample to compute.
 * @param[in] starting_point The seed in the farthest point algorithm.
 * @param[out] output_it The output iterator for points.
 * @param[out] dist_it The optional output iterator for distances.
 *
 * \warning Older versions of this function took a CGAL kernel as argument. Users need to replace `k` with
 * `k.squared_distance_d_object()` in the first argument of every call to `choose_n_farthest_points`.
 *  
 */
template < typename Distance,
typename Point_range,
typename PointOutputIterator,
typename DistanceOutputIterator = Null_output_iterator>
void choose_n_farthest_points1(Distance dist,
                              Point_range const &input_pts,
                              std::size_t final_size,
                              std::size_t starting_point,
                              PointOutputIterator output_it,
                              DistanceOutputIterator dist_it = {}) {
  std::size_t nb_points = boost::size(input_pts);
  if (final_size > nb_points)
    final_size = nb_points;

  // Tests to the limit
  if (final_size < 1)
    return;

  if (starting_point == random_starting_point) {
    // Choose randomly the first landmark
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::size_t> dis(0, nb_points - 1);
    starting_point = dis(gen);
  }

  // FIXME: don't hard-code the type as double. For Epeck_d, we also want to handle types that do not have an infinity.
  typedef double FT;
  static_assert(std::numeric_limits<double>::has_infinity, "the number type needs to support infinity()");

  *output_it++ = input_pts[starting_point];
  *dist_it++ = std::numeric_limits<FT>::infinity();
  if (final_size == 1) return;

  std::vector<std::size_t> points(nb_points);  // map from remaining points to indexes in input_pts
  std::vector< FT > dist_to_L(nb_points);  // vector of current distances to L from points
  for(std::size_t i = 0; i < nb_points; ++i) {
    points[i] = i;
    dist_to_L[i] = dist(input_pts[i], input_pts[starting_point]);
  }
  // The indirection through points makes the program a bit slower. Some alternatives:
  // - the original code never removed points and counted on them not
  //   reappearing because of a self-distance of 0. This causes unnecessary
  //   computations when final_size is large. It also causes trouble if there are
  //   input points at distance 0 from each other.
  // - copy input_pts and update the local copy when removing points.

  std::size_t curr_max_w = starting_point;

  for (std::size_t current_number_of_landmarks = 1; current_number_of_landmarks != final_size; current_number_of_landmarks++) {
    std::size_t latest_landmark = points[curr_max_w];
    // To remove the latest landmark at index curr_max_w, replace it
    // with the last point and reduce the length of the vector.
    std::size_t last = points.size() - 1;
    if (curr_max_w != last) {
      points[curr_max_w] = points[last];
      dist_to_L[curr_max_w] = dist_to_L[last];
    }
    points.pop_back();

    // Update distances to L.
    std::size_t i = 0;
    for (auto p : points) {
      FT curr_dist = dist(input_pts[p], input_pts[latest_landmark]);
      if (curr_dist < dist_to_L[i])
        dist_to_L[i] = curr_dist;
      ++i;
    }
    // choose the next landmark
    curr_max_w = 0;
    FT curr_max_dist = dist_to_L[curr_max_w];  // used for defining the furthest point from L
    for (i = 1; i < points.size(); i++)
      if (dist_to_L[i] > curr_max_dist) {
        curr_max_dist = dist_to_L[i];
        curr_max_w = i;
      }
    *output_it++ = input_pts[points[curr_max_w]];
    *dist_it++ = dist_to_L[curr_max_w];
  }
}


template<class FT> struct Point_info;
template<class FT>
struct Compare_landmark_radius {
  std::vector<Point_info<FT>>* landmarks_p;
  Compare_landmark_radius(std::vector<Point_info<FT>>* p): landmarks_p(p) {}
  bool operator()(std::size_t, std::size_t) const;
};
// I compared all the heaps in boost. Fibonacci is not bad, but d_ary is by far the fastest. Arity of 3 is clearly faster than 2, and doesn't change much afterwards.
template<class FT>
using radius_priority_ds = boost::heap::d_ary_heap<std::size_t, boost::heap::arity<7>, boost::heap::compare<Compare_landmark_radius<FT>>, boost::heap::mutable_<true>, boost::heap::constant_time_size<false>>;
template<class FT>
struct Point_info {
  // As a landmark
  std::size_t far; FT radius;
  FT date; // radius when it became a landmark
  // The points that are closer to this landmark than to other landmarks
  //struct Point {
  //  std::size_t idx; FT d; std::size_t old; FT since;
  //  Point(std::size_t i, FT d_, std::size_t o, FT s): idx(i), d(d_), old(o), since(s) {}
  //};
  std::vector<std::size_t> voronoi;
  // For a landmark A, the list of landmarks B such that picking a Voronoi
  // point of A as a new landmark might steal a Voronoi point from B.
  std::vector<std::pair<std::size_t, FT>> neighbors;
  // Note that above we cache the distances. This is always good for Voronoi, and for neighbors it is neutral in 2D and helps in 4D.
  typename radius_priority_ds<FT>::handle_type position_in_queue;
  // As a witness
  FT d; std::size_t old; FT since; // TODO: reuse far/radius for those
};
template<class FT>
bool Compare_landmark_radius<FT>::operator()(std::size_t a, std::size_t b)const{ return (*landmarks_p)[a].radius < (*landmarks_p)[b].radius; }

template < typename Distance,
typename Point_range,
typename PointOutputIterator,
typename DistanceOutputIterator = Null_output_iterator>
void choose_n_farthest_points(Distance dist_,
                              Point_range const &input_pts,
                              std::size_t final_size,
                              std::size_t starting_point,
                              PointOutputIterator output_it,
                              DistanceOutputIterator dist_it = {}) {
  std::size_t nb_points = boost::size(input_pts);
  if (final_size > nb_points)
    final_size = nb_points;

  // Tests to the limit
  if (final_size < 1)
    return;

  if (starting_point == random_starting_point) {
    // Choose randomly the first landmark
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::size_t> dis(0, nb_points - 1);
    starting_point = dis(gen);
  }

  // FIXME: don't hard-code the type as double. For Epeck_d, we also want to handle types that do not have an infinity.
  typedef double FT;
  static_assert(std::numeric_limits<FT>::has_infinity, "the number type needs to support infinity()");

  *output_it++ = input_pts[starting_point];
  *dist_it++ = std::numeric_limits<FT>::infinity();
  if (final_size == 1) return;

  auto dist = [&](std::size_t a, std::size_t b){ return dist_(input_pts[a], input_pts[b]); };

  // TODO: rename
  std::vector<Point_info<FT>> landmarks(nb_points);
  radius_priority_ds<FT> radius_priority(&landmarks);

  auto compute_radius = [&](std::size_t i)
  {
    FT r = -std::numeric_limits<FT>::infinity(); // -2 * diameter should suffice
    std::size_t jmax = -1;
    for(auto p : landmarks[i].voronoi) {
      FT d = landmarks[p].d;
      if (d > r) {
        r = d;
        jmax = p;
      }
    }
    landmarks[i].radius = r;
    landmarks[i].far = jmax;
  };
  auto update_radius = [&](std::size_t i)
  {
    compute_radius(i);
    radius_priority.decrease(landmarks[i].position_in_queue);
  };

  {
    // Initialize everything with starting_point
    auto& ini = landmarks[starting_point];
    ini.voronoi.reserve(nb_points - 1);
    for (std::size_t i = 0; i < nb_points; ++i)
      if (i != starting_point) {
        FT d = dist(starting_point, i);
        ini.voronoi.emplace_back(i);
        landmarks[i].d = d;
        landmarks[i].old = -1;
        landmarks[i].since = std::numeric_limits<FT>::infinity();
      }
    compute_radius(starting_point);
    ini.position_in_queue = radius_priority.push(starting_point);
    ini.date = std::numeric_limits<FT>::infinity();
  }
  // outside the loop to recycle the allocation
  for (std::size_t current_number_of_landmarks = 1; current_number_of_landmarks != final_size; current_number_of_landmarks++) {
    std::size_t l_parent = radius_priority.top();
    auto& parent_info = landmarks[l_parent];
    std::size_t l = parent_info.far;
    assert(l != -1);
    FT radius = parent_info.radius;
    auto& info = landmarks[l];
    *output_it++ = input_pts[l];
    *dist_it++ = radius;
    // If a Voronoi point X of A can steal a Voronoi point Y from B, then
    // BY > XY >= AB - AX - BY, so AB <= AX + 2 * BY.
    auto max_dist = [](FT a, FT b){ return a + 2 * b; }; // tighter than 3 * radius
    // Check if any Voronoi points from ngb need to move to l
    auto handle_neighbor_voronoi = [&](std::size_t ngb)
    {
      auto& ngb_info = landmarks[ngb];
      auto it = std::remove_if(ngb_info.voronoi.begin(), ngb_info.voronoi.end(), [&](std::size_t w)
          {
          auto& w_info = landmarks[w];
            FT d = w_info.d;
            FT newd = dist(l, w);
            if (newd < d) {
              if (w != l) { // w==l can only happen for ngb==l_parent
                info.voronoi.emplace_back(w);
                w_info.d = newd;
                if (w_info.since > radius * std::sqrt(3)) {
                  w_info.old = ngb;
                  w_info.since = radius;
                }
              }
              return true;
            }
            return false;
          });
      if (it != ngb_info.voronoi.end()) { // modified, always true for ngb==l_parent
        ngb_info.voronoi.erase(it, ngb_info.voronoi.end());
        // We only need to recompute the radius if far was removed, which we can test here with
        //   if (dist(l, ngb_info.far) < ngb_info.radius)
        // to avoid a costly test for each w in the loop above, but it does not seem to help.
        update_radius(ngb);
        // if (ngb_info.voronoi.empty()) radius_priority.erase(ngb_info.position_in_queue);
        return true;
      } else {
        return false;
      }
    };

    // BUG: guardian used before being set...

    // First update the Voronoi diagram, so we can compute all the updated
    // radii before pruning neighbor lists. The main drawback is that we have
    // to store modified_neighbors, and we don't have access to the old radii
    // in handle_neighbor_neighbors.
    std::size_t guardian = (info.since > std::sqrt(3) * radius) ? l_parent : info.old;
    assert(guardian != -1);
    auto& guardian_info = landmarks[guardian];
    handle_neighbor_voronoi(guardian);
    // Should we make this loop a remove_if? We already remove in the next loop.
    //for (auto ngb_ : parent_info.neighbors) {
    //  std::size_t ngb = ngb_.first;
    //  if(ngb_.second <= max_dist(radius, landmarks[ngb].radius)) // radius from before update_radius(l_parent)
    //    handle_neighbor_voronoi(ngb);
    //}
    // should we try to use the next radius already?
    FT d_to_guardian = dist(l, guardian);
    std::remove_if(guardian_info.neighbors.begin(), guardian_info.neighbors.end(), [&](auto& ngb){
        assert(ngb.first != -1);
        if (ngb.second > 6 * radius) return true;
        //if(ngb.second <= max_dist(d_to_guardian, landmarks[ngb.first].radius)) // radius from before update_radius(guardian)
        if(dist(l, ngb.first) <= 2 * landmarks[ngb.first].radius)
          handle_neighbor_voronoi(ngb.first);
        return false;
    });
    for (auto& ngb : guardian_info.neighbors) {
        FT d = dist(l, ngb.first);
        if (d <= std::min(6 * radius, std::sqrt(12) * landmarks[ngb.first].date)) {
          landmarks[ngb.first].neighbors.emplace_back(l, d);
        }
        if (d <= std::sqrt(12) * radius) {
          info.neighbors.emplace_back(ngb.first, d);
        }
    }
    guardian_info.neighbors.emplace_back(l, d_to_guardian);
    info.neighbors.emplace_back(guardian, d_to_guardian);
    compute_radius(l);
    info.position_in_queue = radius_priority.push(l);
    info.date = radius;
  }
}

}  // namespace subsampling

}  // namespace Gudhi

#endif  // CHOOSE_N_FARTHEST_POINTS_H_

/* THIS IS COMPLETELY BROKEN
 * The right constants are 2/4/8 as in Har-Peled's paper, none of that sqrt(3) business (strange that it seems to work in practice...)
 * And even then, it isn't clear how we are supposed to add l to the neighbor lists of other points, the relation is not symmetric.
 */
