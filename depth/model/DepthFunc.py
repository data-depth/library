from multivariate import *

# from depth.multivariate import *


def int_functional_depth(data, query_point, type_of_depth='halfspace', solver='neldermead', NRandom=100):
    """
    Compute the integrated functional depth (IFD) of a query function with respect to a sample of functional data.

    The function computes the average (integrated) multivariate depth of the query function
    across all discretization points. At each point, the multivariate depth is evaluated
    using the specified depth type and solver.

    Parameters
    ----------
    data : np.ndarray
        A 3D NumPy array of shape (N_data, L, D)
        representing the sample of functional data:
        - N_data: number of functional observations (samples)
        - L: number of discretization points per function
        - D: dimension of the data at each discretization point

        Each element data[j, i, :] corresponds to the D-dimensional
        observation of the j-th function at discretization point i.

    query_point : np.ndarray
        A 2D NumPy array of shape (L, D)
        representing the function for which the integrated depth is to be computed.
        It has the same structure as a single element of `data`.

    type_of_depth : str, optional, default='halfspace'
        The name of the multivariate depth function to use for each time slice.
        Typical choices include:
        - 'halfspace'
        - 'projection'
        - 'simplicial', etc.

    solver : str, optional, default='neldermead'
        The numerical method used to approximate the multivariate depth.
        For example, 'neldermead' uses the Nelder-Mead optimization algorithm.

    NRandom : int, optional, default=100
        The number of random projections (or random directions)
        
    Returns
    -------
    functional_depth_val : float
        The integrated functional depth of `query_point` with respect to `data`,
        computed as the average of the multivariate depths across all L discretization points.

    Notes
    -----
    For each discretization point i = 1, ..., L:
        - Extract the data slice `data[:, i, :]` (shape: N_data x D)
        - Extract the query vector `query_point[i, :]` (shape: D)
        - Compute the multivariate depth of the query vector relative to the data slice
        - Average the results over all L time points

    """
    total_depth_sum = 0
    l_points, d = query_point.shape

    for i in range(l_points):
        # data_component_slice: N_data x D matrix (all functions at time i)
        data_component_slice = data[:, i, :]

        # query_component: D-dimensional vector (query function at time i)
        query_component = query_point[i, :]

        # Compute depth at time i
        time_component_depth = depth_approximation(
            query_component, data_component_slice,
            type_of_depth, solver, NRandom, option=1
        )

        total_depth_sum += time_component_depth

    # Average depth over all L time points
    functional_depth_val = total_depth_sum / l_points

    return functional_depth_val



# Example
# --------
# import numpy as np

# # Suppose we have 3 functions (N_data = 3),
# # each discretized at 4 time points (L = 4),
# # and each observation is 2-dimensional (D = 2)
# data = np.array([
#     [[0.1, 0.2],
#      [0.2, 0.4],
#      [0.3, 0.1],
#      [0.5, 0.3]],

#     [[0.0, 0.1],
#      [0.3, 0.2],
#      [0.2, 0.3],
#      [0.6, 0.2]],

#     [[0.2, 0.0],
#      [0.4, 0.3],
#      [0.5, 0.4],
#      [0.7, 0.5]]
# ])  # Shape: (3, 4, 2)

# # Define one query function of the same structure (L = 4, D = 2)
# query_point = np.array([
#     [0.15, 0.15],
#     [0.25, 0.35],
#     [0.35, 0.25],
#     [0.55, 0.30]
# ])  # Shape: (4, 2)
# print(4)

# # Compute integrated functional depth
# depth_value = int_functional_depth(data, query_point, type_of_depth='halfspace', solver='neldermead', NRandom=50)
# print("Integrated functional depth:", depth_value)


