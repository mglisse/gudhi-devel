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
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <gudhi/persistence_on_rectangle.h>

namespace py = pybind11;
typedef std::vector<std::array<double, 2>> Vd;
PYBIND11_MAKE_OPAQUE(Vd);

py::list fun(py::array_t<double, py::array::c_style | py::array::forcecast> data, double min_persistence) {
  py::buffer_info buf = data.request();
  if(buf.ndim!=2)
    throw std::runtime_error("Data must be a 2-dimensional array");
  std::vector<std::array<double, 2>> dgm0, dgm1;
  {
    py::gil_scoped_release release;
    double mini = Gudhi::cubical_complex::persistence_on_rectangle_from_top_cells(
        static_cast<double const*>(buf.ptr), buf.shape[0], buf.shape[1],
        [&](double b, double d){ if (d - b > min_persistence) dgm0.push_back({b, d}); },
        [&](double b, double d){ if (d - b > min_persistence) dgm1.push_back({b, d}); });
    dgm0.push_back({mini, std::numeric_limits<double>::infinity()});
  }
  py::list ret;
  ret.append(py::array(py::cast(std::move(dgm0))));
  ret.append(py::array(py::cast(std::move(dgm1))));
  return ret;
}

PYBIND11_MODULE(_persrect, m) {
  py::bind_vector<Vd>(m, "VectorPairDouble", py::buffer_protocol());
  m.def("persistence_on_rectangle_from_top_cells", fun);
}
