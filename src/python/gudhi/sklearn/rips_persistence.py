# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
# See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
# Author(s):       Vincent Rouvreau
#
# Copyright (C) 2022 Inria
#
# Modification(s):
#   - YYYY/MM Author: Description of the modification

from .._ripser import _lower, _full, _sparse, _lower_to_coo
from ..flag_filtration.edge_collapse import reduce_graph
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# joblib is required by scikit-learn
from joblib import Parallel, delayed

# Mermaid sequence diagram - https://mermaid-js.github.io/mermaid-live-editor/
# sequenceDiagram
#   participant USER
#   participant R as RipsPersistence
#   USER->>R: fit_transform(X)
#   Note right of R: homology_dimensions=[i,j]
#   R->>thread1: _tranform(X[0])
#   R->>thread2: _tranform(X[1])
#   Note right of R: ...
#   thread1->>R: [array( Hi(X[0]) ), array( Hj(X[0]) )]
#   thread2->>R: [array( Hi(X[1]) ), array( Hj(X[1]) )]
#   Note right of R: ...
#   R->>USER: [[array( Hi(X[0]) ), array( Hj(X[0]) )],<br/> [array( Hi(X[1]) ), array( Hj(X[1]) )],<br/>...]


class RipsPersistence(BaseEstimator, TransformerMixin):
    """
    This is a class for constructing Vietoris-Rips complexes and computing the persistence diagrams from them.
    """

    def __init__(
        self,
        homology_dimensions,
        threshold=float('inf'),
        input_type='point cloud',
        num_collapses='auto',
        homology_coeff_field=11,
        n_jobs=None,
    ):
        """
        Constructor for the RipsPersistence class.

        Parameters:
            homology_dimensions (int or list of int): The returned persistence diagrams dimension(s).
                Short circuit the use of :class:`~gudhi.representations.preprocessing.DimensionSelector` when only one
                dimension matters (in other words, when `homology_dimensions` is an int).
            threshold (float): Rips maximal edge length value. Default is +Inf.
            input_type (str): Can be 'point cloud' when inputs are point clouds, 'full distance matrix',
                'lower distance matrix' when inputs are lower triangular distance matrix (can be full square,
                but the upper part of the distance matrix will not be considered), or 'coo_matrix' for a distance
                matrix in SciPy's sparse format. Default is 'point cloud'.
            num_collapses (int|str): Specify the number of :func:`~gudhi.SimplexTree.collapse_edges` iterations to perform
                on the SimplexTree. Default value is 'auto'.
            homology_coeff_field (int): The homology coefficient field. Must be a prime number. Default value is 11.
            n_jobs (int): Number of jobs to run in parallel. `None` (default value) means `n_jobs = 1` unless in a
                joblib.parallel_backend context. `-1` means using all processors. cf.
                https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html for more details.
        """
        self.homology_dimensions = homology_dimensions
        self.threshold = threshold
        self.input_type = input_type
        self.num_collapses = num_collapses
        self.homology_coeff_field = homology_coeff_field
        self.n_jobs = n_jobs

    def fit(self, X, Y=None):
        """
        Nothing to be done, but useful when included in a scikit-learn Pipeline.
        """
        return self

    def __transform(self, inp):
        # TODO: give the user the option to force the use of one particular strategy (including SimplexTree)
        max_dimension = max(self.dim_list_)
        input_type = self.input_type

        if input_type == 'point cloud':
            if self.threshold < float('inf'):
                # Hope that the user gave a useful threshold
                from scipy.spatial import cKDTree
                from scipy.sparse import coo_matrix
                tree = cKDTree(inp)
                # Or tree.query_pairs(r=self.threshold, output_type='ndarray') and recompute the distances?
                inp = tree.sparse_distance_matrix(tree, max_distance=self.threshold, output_type="coo_matrix")
                input_type = 'coo_matrix' # call it 'sparse distance matrix'? 'distance coo_matrix'?
            else:
                from scipy.spatial.distance import pdist, squareform
                inp = squareform(pdist(inp))
                input_type = 'full distance matrix'

        # Edge collapse always goes through the sparse format
        num_collapses = self.num_collapses
        if num_collapses == 'auto':
            num_collapses = 1 if max_dimension > 1 else 0
            # or num_collapses=max_dimension-1 maybe?
        elif max_dimension == 0:
            num_collapses = 0
        if num_collapses > 0:
            if input_type in ('full distance matrix', 'lower distance matrix'):
                # For 'full' we could use i, j = np.triu_indices_from(inp, k=1), etc
                i, j, f = _lower_to_coo(inp, self.threshold)
                # TODO move this, or use directly _collapse_edges
                from scipy.sparse import coo_matrix
                inp = coo_matrix((f, (i, j)), shape=(len(inp),) * 2)
                input_type = 'coo_matrix'
            inp = reduce_graph(inp, num_collapses)

        if input_type == 'full distance matrix':
            #TODO: compute cone_radius before edge collapse
            #TODO: possibly transpose for performance
            inp = np.asarray(inp)
            cone_radius = inp.max(-1).min()
            r = min(self.threshold, cone_radius)
            dgm = _full(inp, max_dimension=max_dimension, max_edge_length=self.threshold, homology_coeff_field=self.homology_coeff_field)
        elif input_type == 'lower distance matrix':
            # minmax threshold is computed inside _lower
            dgm = _lower(inp, max_dimension=max_dimension, max_edge_length=self.threshold, homology_coeff_field=self.homology_coeff_field)
        elif input_type == 'coo_matrix':
            #TODO: switch to coo_array (the interface is different)?
            dgm = _sparse(inp.row, inp.col, inp.data, inp.shape[0], max_dimension=max_dimension, max_edge_length=self.threshold, homology_coeff_field=self.homology_coeff_field)
        else:
            raise ValueError("Only 'point cloud', 'lower distance matrix', 'full distance matrix' and 'coo_matrix' are valid input_type") # move to __init__?
        
        return [dgm[dim] for dim in self.dim_list_]

    def transform(self, X, Y=None):
        """Compute all the Vietoris-Rips complexes and their associated persistence diagrams.

        :param X: list of point clouds as Euclidean coordinates or distance matrices.
        :type X: list of list of float OR list of numpy.ndarray

        :return: Persistence diagrams in the format:

              - If `homology_dimensions` was set to `n`: `[array( Hn(X[0]) ), array( Hn(X[1]) ), ...]` 
              - If `homology_dimensions` was set to `[i, j]`:
                `[[array( Hi(X[0]) ), array( Hj(X[0]) )], [array( Hi(X[1]) ), array( Hj(X[1]) )], ...]`
        :rtype: list of numpy ndarray of shape (,2) or list of list of numpy ndarray of shape (,2)
        """
        # Depends on homology_dimensions is an integer or a list of integer (else case)
        if isinstance(self.homology_dimensions, int):
            unwrap = True
            self.dim_list_ = [ self.homology_dimensions ]
        else:
            unwrap = False
            self.dim_list_ = self.homology_dimensions

        # threads is preferred as Rips construction and persistence computation releases the GIL
        res = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self.__transform)(inputs) for inputs in X)

        if unwrap:
            res = [d[0] for d in res]
        return res

    def get_feature_names_out(self):
        """Provide column names for implementing sklearn's set_output API."""
        return [f"H{i}" for i in self.dim_list_]
