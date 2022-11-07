#include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <gudhi/choose_n_farthest_points.h>

#include <CGAL/Random.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <chrono>

int main(int argc, char**argv) {
  //typedef CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt K;
  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef typename K::Point_2 Point_d;

  CGAL::Random rd(atoi(argv[2]));
  //CGAL::Random rd(994);

  const int n2 = atoi(argv[1]);
  //const int n2 = 4;
  const int n1 = n2;
  const int j = 0;
  std::vector<Point_d> points;
  for (int i = 0; i < n2; ++i) {
    points.emplace_back(rd.get_double(-1., 1), rd.get_double(-1., 1));
#if PRINT
    std::cerr << "Point " << i << ": " << points.back().x() << '\t' << points.back().y() << '\n';
#endif
  }

  K k;

  auto dis = [&](auto&p, auto&q){return sqrt(k.compute_squared_distance_2_object()(p,q));};

#if PRINT
  std::cerr.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cerr.precision(2);
  std::cerr << "Matrice de distances" << '\n';
  for(int i=0; i<n2; ++i) {
    for(int j=0; j<n2; ++j) {
      std::cerr << CGAL::to_double(dis(points[i], points[j])) << '\t';
    }
    std::cerr << '\n';
  }
#endif

  std::vector<Point_d> results;
  std::vector<K::FT> dists;
#ifndef PROFIL
  auto time_start1 = std::chrono::system_clock::now();
  Gudhi::subsampling::choose_n_farthest_points1(dis, points, n1,
                                               j,
                                               std::back_inserter(results),
                                               std::back_inserter(dists)
                                               );
  auto time_stop1 = std::chrono::system_clock::now();
  std::ofstream o("O");
  for(int i=0;i<n1;++i) o << CGAL::to_double(dists[i]) << " -- " << results[i] << '\n';
  results.clear(); dists.clear();
  auto time_start2 = std::chrono::system_clock::now();
#endif
  Gudhi::subsampling::choose_n_farthest_points(dis, points, n1,
                                               j,
                                               std::back_inserter(results),
                                               std::back_inserter(dists)
                                               );
#ifndef PROFIL
  auto time_stop2 = std::chrono::system_clock::now();
  std::ofstream n("N");
  for(int i=0;i<n1;++i) n << CGAL::to_double(dists[i]) << " -- " << results[i] << '\n';
  std::cerr << "Time old " << std::chrono::duration_cast<std::chrono::milliseconds>((time_stop1 - time_start1)).count()
             << " vs new " << std::chrono::duration_cast<std::chrono::milliseconds>((time_stop2 - time_start2)).count() << '\n';
#endif
  return 0;
}
