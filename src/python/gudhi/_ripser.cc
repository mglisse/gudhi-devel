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

template<class value_t_>struct DParams {
  typedef int vertex_t;
  typedef value_t_ value_t;
};

template<class T>
py::list euclidean(py::array_t<T> points, int max_dimension, T max_edge_length, unsigned homology_coeff_field) {
  py::buffer_info buf = points.request();
  if(buf.ndim!=2)
    throw std::runtime_error("points must be a 2-dimensional array");
  py::ssize_t n = buf.shape[0];
  py::ssize_t d = buf.shape[1];
  auto&& data = points.unchecked();
  std::vector<std::vector<std::array<T, 2>>> dgms;
  {
    py::gil_scoped_release release;
    // TODO: for now we copy to euclidean_distance_matrix_, but it would be nicer to create a different class with the same interface that just wraps the numpy array.
    std::vector<std::vector<T>> pts;
    pts.reserve(n);
    for(py::ssize_t i = 0; i < n; ++i) {
      std::vector<T> pt;
      pt.reserve(d);
      for(py::ssize_t j = 0; j < d; ++j) {
        pt.push_back(data(i,j));
      }
      pts.push_back(std::move(pt));
    }
    euclidean_distance_matrix_<DParams<T>> dist(std::move(pts));
    auto output = [&](T birth, T death){ dgms.back().push_back({birth, death}); };
    auto switch_dim = [&](int new_dim){
      dgms.emplace_back();
    };
    ripser(std::move(dist), max_dimension, max_edge_length, homology_coeff_field, switch_dim, output);
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
