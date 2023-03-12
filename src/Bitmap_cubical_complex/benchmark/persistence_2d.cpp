/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s): Marc Glisse
 *
 *    Copyright (C) 2023 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#include <gudhi/Clock.h>
#include <gudhi/Bitmap_cubical_complex.h>
#include <gudhi/Persistent_cohomology.h>

#include <vector>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>

std::random_device rd;
std::mt19937 gen(rd());

double get_random()
{
    std::uniform_real_distribution<double> dist(0., 1.);
    return dist(gen);
}

int main() {
  typedef Gudhi::cubical_complex::Bitmap_cubical_complex_base<double> Base;
  typedef Gudhi::cubical_complex::Bitmap_cubical_complex<Base> Cubical;
  using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;

  std::vector<unsigned> sizes {1000, 999};
  std::vector<double> data(sizes[0] * sizes[1]);
  std::generate(data.begin(), data.end(), get_random);

  Gudhi::Clock clock;
#ifndef ONLY_DUAL
  Cubical complex_from_top_cells(sizes, data, true);
  std::clog << "Construction from top cells: " << clock;

  clock.begin();
  complex_from_top_cells.initialize_filtration();
  std::clog << "initialize_filtration: " << clock;

  clock.begin();
  Gudhi::persistent_cohomology::Persistent_cohomology<Cubical, Field_Zp> pers(complex_from_top_cells);
  pers.init_coefficients(2);
  pers.compute_persistent_cohomology();
  std::clog << "Compute persistent homology: " << clock << std::endl;
  std::vector<std::pair<double, double>> res1;
  for (auto p: pers.get_persistent_pairs()){
    res1.emplace_back(complex_from_top_cells.filtration(std::get<0>(p)), complex_from_top_cells.filtration(std::get<1>(p)));
  }
  std::sort(res1.begin(), res1.end());
#endif


  clock.begin();
  Cubical complex_dual(sizes, data, Gudhi::cubical_complex::Dual_from_vertices());
  std::clog << "Dual construction from top cells: " << clock;

  clock.begin();
  std::vector<std::pair<double, double>> res2;
  std::size_t global_min = complex_dual.persistence_2d_dual([&complex_dual, &res2](std::size_t b, std::size_t d)
      {
      double bf = complex_dual.filtration(b);
      double df = complex_dual.filtration(d);
      if (bf < df) res2.emplace_back(bf, df);
      });
  res2.emplace_back(complex_dual.filtration(global_min), std::numeric_limits<double>::infinity());
  std::clog << "Compute persistent homology: " << clock << std::endl;
  std::sort(res2.begin(), res2.end());

#ifndef ONLY_DUAL
  if(res1 != res2) std::exit(-1);
#endif

  return 0;
}
