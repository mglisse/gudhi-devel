/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Marc Glisse
 *
 *    Copyright (C) 2023 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#include <vector>
#include <array>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <gudhi/ripser.h>

namespace py = pybind11;
typedef std::vector<std::array< float, 2>> Vf;
typedef std::vector<std::array<double, 2>> Vd;
PYBIND11_MAKE_OPAQUE(Vf);
PYBIND11_MAKE_OPAQUE(Vd);

template<class T>struct Numpy_euclidean {
  typedef T value_t;

  decltype(std::declval<py::array_t<T>&>().unchecked()) data;

  int size() const { return data.shape(0); }

  T operator()(int i, int j) const {
    T dist = 0;
    for (int k=0; k<data.shape(1); ++k) {
      T diff = data(i, k) - data(j, k);
      dist += diff * diff;
    }
    return std::sqrt(dist);
  }
};

template<class value_t_>struct DParams {
  typedef int vertex_t;
  typedef value_t_ value_t;
};

template<class T>
py::list euclidean(py::array_t<T> points, int max_dimension, T max_edge_length, unsigned homology_coeff_field) {
  Numpy_euclidean<T> dist_{points.unchecked()};
  if(dist_.data.ndim() != 2)
    throw std::runtime_error("points must be a 2-dimensional array");
  std::vector<std::vector<std::array<T, 2>>> dgms;
  {
    py::gil_scoped_release release;
    compressed_distance_matrix<DParams<T>, LOWER_TRIANGULAR> dist(dist_);
    auto output = [&](T birth, T death){ dgms.back().push_back({birth, death}); };
    auto switch_dim = [&](int new_dim){
      dgms.emplace_back();
    };
    ripser_auto(std::move(dist), max_dimension, max_edge_length, homology_coeff_field, switch_dim, output);
  }
  py::list ret;
  for (auto&& dgm : dgms)
    ret.append(py::array(py::cast(std::move(dgm))));
  return ret;
}

PYBIND11_MODULE(_ripser, m) {
  py::bind_vector<Vf>(m, "VectorPairFloat" , py::buffer_protocol());
  py::bind_vector<Vd>(m, "VectorPairDouble", py::buffer_protocol());
  m.def("_euclidean", euclidean<float>, py::arg("points").noconvert(), py::arg("max_dimension") = std::numeric_limits<int>::max(), py::arg("max_edge_length") = std::numeric_limits<float>::infinity(), py::arg("homology_coeff_field") = 2);
  m.def("_euclidean", euclidean<double>, py::arg("points"), py::arg("max_dimension") = std::numeric_limits<int>::max(), py::arg("max_edge_length") = std::numeric_limits<double>::infinity(), py::arg("homology_coeff_field") = 2);
}

// TODO:
// * input matrice de distances "low" (et aussi "full"? ou que "full" et on convertit côté python?)
// * input matrice de distances sparse "coo"
// * transformer euclidean en dense
// * réutiliser le code dans utility qui calcule maxmin et transforme parfois dense en sparse
//
// - sparse input -> sparse matrix
// - euclidean input & threshold -> sparse matrix (don't build dense matrix)
// - euclidean input & !threshold -> dense matrix
// - dense matrix & threshold -> sparse matrix
// - dense matrix & !threshold -> compute minmax, keep dense
