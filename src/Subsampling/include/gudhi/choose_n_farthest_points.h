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
#include <boost/heap/skew_heap.hpp>

#include <gudhi/Null_output_iterator.h>

#include <iterator>
#include <vector>
#include <set>
#include <unordered_set>
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
  static_assert(std::numeric_limits<double>::has_infinity, "the number type needs to support infinity()");

  *output_it++ = input_pts[starting_point];
  *dist_it++ = std::numeric_limits<double>::infinity();
  if (final_size == 1) return;

  std::vector<std::size_t> points(nb_points);  // map from remaining points to indexes in input_pts
  std::vector< double > dist_to_L(nb_points);  // vector of current distances to L from points
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
      double curr_dist = dist(input_pts[p], input_pts[latest_landmark]);
      if (curr_dist < dist_to_L[i])
        dist_to_L[i] = curr_dist;
      ++i;
    }
    // choose the next landmark
    curr_max_w = 0;
    double curr_max_dist = dist_to_L[curr_max_w];  // used for defining the furthest point from L
    for (i = 1; i < points.size(); i++)
      if (dist_to_L[i] > curr_max_dist) {
        curr_max_dist = dist_to_L[i];
        curr_max_w = i;
      }
    *output_it++ = input_pts[points[curr_max_w]];
    *dist_it++ = dist_to_L[curr_max_w];
  }
}
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

//  // FIXME: don't hard-code the type as double. For Epeck_d, we also want to handle types that do not have an infinity.
//  static_assert(std::numeric_limits<double>::has_infinity, "the number type needs to support infinity()");

  *output_it++ = input_pts[starting_point];
  *dist_it++ = std::numeric_limits<double>::infinity();
  if (final_size == 1) return;

  auto dist = [&](std::size_t a, std::size_t b){ return dist_(input_pts[a], input_pts[b]); };

  //struct Landmark_info { double radius; std::vector<std::size_t> voronoi; std::vector<std::size_t> neighbors; typename radius_priority_ds::handle_type position_in_queue};
  //std::vector<Landmark_info> landmarks(nb_points);
  //auto compare_landmark_radius = [&](std::size_t a, std::size_t b){ return landmarks[a].radius < landmarks[b].radius; };
  //typedef boost::heap::skew_heap<std::size_t, boost::heap::compare<decltype(compare_landmark_radius)>, boost::heap::mutable_<true>, boost::heap::constant_time_size<false>> radius_priority_ds;
  //radius_priority_ds radius_priority(compare_landmark_radius);

  struct Compare_landmark_radius;
  typedef boost::heap::skew_heap<std::size_t, boost::heap::compare<Compare_landmark_radius>, boost::heap::mutable_<true>, boost::heap::constant_time_size<false>> radius_priority_ds;
  struct Landmark_info {
    std::size_t far; double radius;
    std::vector<std::size_t> voronoi;
    std::set<std::size_t> neighbors;
    typename radius_priority_ds::handle_type position_in_queue;
  };
  std::vector<Landmark_info> landmarks(nb_points);
  struct Compare_landmark_radius {
    std::vector<Landmark_info>* landmarks_p;
    Compare_landmark_radius(std::vector<Landmark_info>* p): landmarks_p(p) {}
    bool operator()(std::size_t a, std::size_t b)const{ return (*landmarks_p)[a].radius < (*landmarks_p)[b].radius; }
  };
  radius_priority_ds radius_priority(&landmarks);
  auto compute_radius = [&](std::size_t i)
  {
    double r = -std::numeric_limits<double>::infinity();
    std::size_t jmax = -1;
    for(std::size_t j : landmarks[i].voronoi) {
      double d = dist(i,j);
      if (d > r) {
        r = d;
        jmax = j;
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
      if (i != starting_point)
        ini.voronoi.push_back(i);
    compute_radius(starting_point);
    ini.position_in_queue = radius_priority.push(starting_point);
  }
  for (std::size_t current_number_of_landmarks = 1; current_number_of_landmarks != final_size; current_number_of_landmarks++) {
    std::size_t l_parent = radius_priority.top();
    auto& parent_info = landmarks[l_parent];
    std::size_t l = parent_info.far;
    double radius = parent_info.radius;
    auto& info = landmarks[l];
    *output_it++ = input_pts[l];
    *dist_it++ = radius;
    std::unordered_set<std::size_t> l_neighbors;
    auto handle_neighbor = [&](std::size_t ngb)
    {
      auto& ngb_info = landmarks[ngb];
      auto it = std::remove_if(ngb_info.voronoi.begin(), ngb_info.voronoi.end(), [&](std::size_t w)
          {
            if (dist(l, w) < dist(ngb, w)) {
              if (w != l)
                info.voronoi.push_back(w);
              return true;
            }
            return false;
          });
      bool modified = (it != ngb_info.voronoi.end());
      ngb_info.voronoi.erase(it, ngb_info.voronoi.end());
      if (modified) {
        if (ngb_info.voronoi.empty()) { // Special-casing this does not seem very useful
          radius_priority.erase(ngb_info.position_in_queue);
          for (std::size_t near : ngb_info.neighbors) {
            // Clean-up, neighbors is lazily updated when we visit it
            if (landmarks[near].voronoi.size() != 0 && dist(ngb, near) <= 3 * radius) // could be tighter using the radius of ngb and near?
              l_neighbors.insert(near);
            // don't prune ngb_info.neighbors, if this is l_parent we still need it, and this is the last time we look at it.
            // err, we shouldn't clear() it, but removing the neighbors that are too far is still useful for l_parent
          }
          // can be done directly in the loop above, once the exact datastructure is settled
          //ngb_info.neighbors.clear(); // also remove the symmetric?
        } else {
          // What if ngb_info.voronoi is now empty?
          update_radius(ngb); // check if 'far' was removed, otherwise unnecessary
          std::vector<std::size_t> to_remove;
          for (std::size_t near : ngb_info.neighbors) {
            // Clean-up, neighbors is lazily updated when we visit it
            if (landmarks[near].voronoi.size() != 0 && dist(ngb, near) <= 3 * radius) // could be tighter using the radius of ngb and near?
              l_neighbors.insert(near);
            else
              to_remove.push_back(near);
          }
          // can be done directly in the loop above, once the exact datastructure is settled
          for (std::size_t x : to_remove) {
            ngb_info.neighbors.erase(x); // also remove the symmetric?
          }
        }
      }
      return modified;
    };
    handle_neighbor(l_parent);
    for (std::size_t ngb : parent_info.neighbors)
      handle_neighbor(ngb);
    //if (parent_info.voronoi.empty()) parent_info.neighbors.clear(); // useless, we will never look at it?
    compute_radius(l);
    info.position_in_queue = radius_priority.push(l);
    // ???
    l_neighbors.insert(l_parent);
    // if neighbors is already a set, we don't need an intermediate l_neighbors, or we could make neighbors a vector
    for (std::size_t ngb : l_neighbors) {
      if (landmarks[ngb].voronoi.size() != 0 && dist(l, ngb) <= 3 * radius) {
        info.neighbors.insert(ngb);
        landmarks[ngb].neighbors.insert(l);
      }
    }
  }
}

}  // namespace subsampling

}  // namespace Gudhi

#endif  // CHOOSE_N_FARTHEST_POINTS_H_
