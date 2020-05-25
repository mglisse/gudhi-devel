#include <gudhi/Flag_complex_sparse_matrix.h>

#include <iostream>
#include <vector>

int main() {
  // Type definitions
  using Filtration_value = float;
  using Vertex_handle = short;
  using Flag_complex_sparse_matrix = Gudhi::collapse::Flag_complex_sparse_matrix<Vertex_handle, Filtration_value>;
  using Filtered_edge = Flag_complex_sparse_matrix::Filtered_edge;
  using Filtered_edge_list = std::vector<Filtered_edge>;
  using Edge = Flag_complex_sparse_matrix::Edge;

  //  1   2
  //  o---o
  //  |\ /|
  //  | x |
  //  |/ \|
  //  o---o
  //  0   3
  Filtered_edge_list graph = {{{0, 1}, 1.},
                              {{1, 2}, 1.},
                              {{2, 3}, 1.},
                              {{3, 0}, 1.},
                              {{0, 2}, 2.},
                              {{1, 3}, 2.}};

  Flag_complex_sparse_matrix flag_complex_sparse_matrix(graph);

  Filtered_edge_list collapse_edges;
  // Retrieve collapse edges from the output iterator
  flag_complex_sparse_matrix.filtered_edge_collapse(
    [&collapse_edges](std::pair<Vertex_handle, Vertex_handle> edge, Filtration_value filtration) {
        collapse_edges.push_back({edge, filtration});
      });

  for (Filtered_edge filtered_edge_from_collapse : collapse_edges) {
    Edge edge_from_collapse = std::get<0>(filtered_edge_from_collapse);
    std::cout << "fn[" << std::get<0>(edge_from_collapse) << ", " << std::get<1>(edge_from_collapse) << "] = "
              << std::get<1>(filtered_edge_from_collapse) << std::endl;
  }

  return 0;
}