from . import multivariate as mvt
# from DepthEucl import DepthEucl
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Literal

# from depth.multivariate import *

class DepthFunc():
    """
    Functional Data-Depth
    
    Computes the depth of functional data

    """
    def __init__(self):
        self.data=None
    
    def load_dataset(self, data:np.ndarray=None, y:np.ndarray=None, 
                     timestamp_col:str|int='timestamp',value_cols:str|list='value', case_id:str|int="case_id", 
                     interpolate_grid:bool=True,N_grid:int=10,
                     interpolation_type:str="linear"):
        """
        Load the dataset X for reference calculations. Depth is computed with respect to this dataset.

        Parameters
        ----------
        data : array_like.
            Dataset that will be used as base for depth computation
            
        y : Ignored, default=None
            Not used, present for API consistency by convention.
        
        timestamp_col : str|int, default = timestamp
            Column used for discretization of the dataset.
            timestamp_col can be a string indicating the name of the column or an integer for the position of the column.
        
        value_cols: str|list, default='value'
            Columns used for the multivariate depth computation.
            If value_cols is a string, columns with such word in the name are used.
            If value_cols is a list, it is considered the list of columns to be used.
        
        interpolate_grid : bool, default = True   

        Returns
        ---------
        DepthFunc model
        
        Notes
        -----
        For each discretization point i = 1, ..., L:
            - Extract the data slice `data[:, i, :]` (shape: N_data x D)
            - Extract the query vector `query_point[i, :]` (shape: D)
            - Compute the multivariate depth of the query vector relative to the data slice
            - Average the results over all L time points
        
        """ 
        if type(data) == type(None):
            raise Exception("You must load a dataset")
        if type(timestamp_col)==str:
            if timestamp_col in data.columns:    
                self.timestamp_col=timestamp_col
            else:
                raise NameError(f"timestamp_col is not a column in the data.")
        if type(timestamp_col)==int:
            if timestamp_col>=len(data.columns):
                raise IndexError(f"timestamp_col is greater than the number of columns.")
            else:
                self.timestamp_col = data.columns[timestamp_col]
        print(f"timestamp_col is set to {self.timestamp_col}")
        
        if type(value_cols) == str: 
            self.value_cols = data.filter(like = value_cols).columns
            if self.value_cols==[]: raise ValueError("value_cols is an empty list, value_cols should be a suffix for different columns")
        if type(value_cols) == list:
            self.value_cols = value_cols
        print(f"value_cols is set to {self.value_cols}")
        
        if type(case_id)==str:
            if case_id in data.columns:
                self.case_id=case_id
            else:
                raise ValueError("case_id is not a column of the data")
        if type(case_id)==int:
            if case_id>=len(data.columns):
                raise IndexError(f"case_id is greater than the number of columns.")
            else:
                self.case_id = data.columns[case_id]
        print(f"case_id is set to {self.case_id}")
        
        # assert(type(data) == np.ndarray), "The dataset must be a numpy array"
        # self._nSamples=data.shape[0] # define dataset size - n
        # self._spaceDim=data.shape[2] # define space dimension - d
        if interpolate_grid==True:
            self.N_grid = N_grid
            self.t_min = data[self.timestamp_col].min()
            self.t_max = data[self.timestamp_col].max()
            
            if np.issubdtype(data[self.timestamp_col].dtype, np.datetime64):
                new_domain = np.linspace(0, (self.t_max - self.t_min).total_seconds(), self.N_grid)
            else:
                new_domain = np.linspace(0, self.t_max - self.t_min, self.N_grid)
            self.new_domain = new_domain
        elif interpolate_grid==False:
            self.t_min = data[self.timestamp_col].min()
            if np.issubdtype(data[self.timestamp_col].dtype, np.datetime64):
                self.new_domain=np.sort(np.unique((data[self.timestamp_col]-data[self.timestamp_col].min()).dt.total_seconds()))
            else:
                self.new_domain=np.sort(np.unique((data[self.timestamp_col]-data[self.timestamp_col].min())))
        
        self.data_array = self.syncronise_over_time(data, self.new_domain,True,interpolation_type)
        self.data = data
        
        return self  
    
    def syncronise_over_time(self,df, new_domain, interpolation=True, interpolation_type="linear"):
        M = []
        if np.issubdtype(df[self.timestamp_col].dtype, np.datetime64):
            for case, group in df.groupby(self.case_id):
                group = group.loc[~group[self.timestamp_col].duplicated(keep='first')]

                if interpolation:
                    interp_values = self.interpolate(group[self.value_cols], 
                                                        (group[self.timestamp_col]-self.t_min).dt.total_seconds().to_numpy(), new_domain,
                                                        interpolation_type,axis = 0)
                else:
                    interp_values = np.asarray(group[self.value_cols])
                M.append(interp_values)
        else:
            for case, group in df.groupby(self.case_id):
                group = group.loc[~group[self.timestamp_col].duplicated(keep='first')]
                
                if interpolation:
                    interp_values = self.interpolate(group[self.value_cols], (group[self.timestamp_col]-self.t_min).to_numpy(), new_domain, axis = 0)
                else:
                    interp_values = np.asarray(group[self.value_cols])
                M.append(interp_values)
        M = np.stack(M, axis = 0)
        return M
    
    def interpolate(self,value_matrix, original_domain, new_domain, method='linear', axis=0):
        """
        Interpolate functional data (1D or ND) onto a new domain.

        It supports interpolation along any specified axis and handles missing (`NaN`) values by skipping them 
        during interpolation. 

        Parameters
        ----------
        value_matrix : array-like of shape (n_points, ...) or (n_points,)
            Array containing the functional data to be interpolated.
            - For 1D input: a single function defined on `original_domain`.
            - For ND input: multi-dimensional observations, where
            interpolation is performed along the specified `axis`.

        original_domain : array-like of shape (n_points,)
            The domain (e.g., time or spatial coordinates) corresponding to the data 
            in `value_matrix`. Must be strictly increasing and match the first dimension 
            of `value_matrix` along the interpolation axis.

        new_domain : array-like of shape (n_new_points,)
            The new domain points where the interpolation will be evaluated. 

        method : str, default='linear'
            Interpolation method to use. Supported options are those provided by
            `scipy.interpolate.interp1d`, including:
            - `'linear'` : linear interpolation (default)
            - `'nearest'` : nearest-neighbor interpolation
            - `'cubic'` : cubic spline interpolation
            - `'quadratic'` : quadratic spline interpolation
            - `'previous'`, `'next'` : stepwise interpolation

        axis : int, default=0
            Axis along which to perform the interpolation. By default, interpolation 
            is performed along the first axis (rows). 
        Returns
        -------
        interp_value_matrix : np.ndarray
            Interpolated data array evaluated at the `new_domain`.
            - If input was 1D, returns a 1D NumPy array of shape `(len(new_domain),)`.
            - If input was ND, returns an array of shape similar to `value_matrix`, 
            except that the length along the interpolation axis becomes `len(new_domain)`.
        """

        
        value_matrix = np.asarray(value_matrix)
        original_domain = np.asarray(original_domain)
        new_domain = np.asarray(new_domain)

        # 1D case
        if value_matrix.ndim == 1:
            non_nan_idx = ~np.isnan(value_matrix)
            f = interp1d(original_domain[non_nan_idx], value_matrix[non_nan_idx], kind=method, bounds_error=False, fill_value="extrapolate")
            return f(new_domain)

        # Move interpolation axis to 0
        if axis != 0:
            value_matrix = np.moveaxis(value_matrix, axis, 0)

        original_shape = value_matrix.shape
        flat_value_matrix = value_matrix.reshape(original_shape[0], -1)
        interp_flat_value_matrix = np.empty((len(new_domain), flat_value_matrix.shape[1]), dtype=value_matrix.dtype)

        # Interpolate each flattened column
        for j in range(flat_value_matrix.shape[1]):
            non_nan_idx = ~np.isnan(flat_value_matrix[:, j])
            f = interp1d(original_domain[non_nan_idx], flat_value_matrix[:, j][non_nan_idx], kind=method, bounds_error=False, fill_value="extrapolate")
            interp_flat_value_matrix[:, j] = f(new_domain)

        # Reshape back to original dimensions
        interp_value_matrix = interp_flat_value_matrix.reshape((len(new_domain),) + original_shape[1:])

        # Move axis back to original position
        if axis != 0:
            interp_value_matrix = np.moveaxis(interp_value_matrix, 0, axis)

        return interp_value_matrix
    

    def int_depth(self, query_point, type_of_depth='halfspace', solver='neldermead', NRandom=100):
        """
        Compute the integrated functional depth (IFD) of a query function with respect to a sample of functional data.

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
            For example, 'neldermead' uses the Nelder–Mead optimization algorithm.

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
            data_component_slice = self.data_array[:, i, :]

            # query_component: D-dimensional vector (query function at time i)
            query_component = query_point[i, :]

            # Compute depth at time i
            time_component_depth = mvt.depth_approximation(
                query_component, data_component_slice,
                type_of_depth, solver, NRandom, option=1
            )

            total_depth_sum += time_component_depth

        # Average depth over all L time points
        functional_depth_val = total_depth_sum / l_points

        return functional_depth_val
    
    def projection_based_func_depth(self, query, interpolation = True, 
                                    interpolation_type = "linear",
                                    type_of_depth='halfspace', solver='neldermead', NRandom=100,
                                    output_option:Literal["lowest_depth","final_depht_dir"]="lowest_depth"):
        """
        Compute projection-based functional depth for query functional data with respect to a reference dataset.

        This function computes depth values of functional observations (in `query`) relative to a 
        reference dataset (`df`) using projection-based methods such as halfspace depth.
        Each function (trajectory) is represented by a sequence of multivariate values over time.
        If `interpolation=True`, the function first synchronizes all cases over a common 
        time grid via interpolation before computing depths.

        Parameters
        ----------
        df : pandas.DataFrame
            Reference dataset containing functional observations. Must include:
            - 'case_id': identifier for each functional observation.
            - 'timestamp': time or domain variable (can be numeric or datetime64).
            - One or more value columns (e.g., 'value_1', 'value_2', ...).

        query : pandas.DataFrame
            Query dataset containing functional observations whose depth will be computed
            relative to `df`. Must have the same column structure as `df`.

        N_grid : int, default=10
            Number of equally spaced grid points for interpolation. Used only if
            `interpolation=True`.

        interpolation : bool, default=True
            If True, each function is interpolated to a common grid across all cases 
            using the `interpolate()` function. If False, raw data are used as-is.

        type_of_depth : {'halfspace', ...}, default='halfspace'
            Type of functional depth to compute. Currently supports 'halfspace', but 
            can be extended to other projection-based depths.

        solver : str, default='neldermead'
            Optimization solver used within the internal depth computation (`int_depth` function).

        NRandom : int, default=100
            Number of random projections or optimization restarts used in computing 
            projection-based depth.

        Returns
        -------
        depth_array : np.ndarray of shape (n_query,)
            Array of depth values, where `n_query` is the number of functional observations 
            (unique `case_id`s) in the `query` dataset.

        Notes
        -----
        - If `timestamp` is of type `datetime64`, it is converted internally to seconds
        relative to the global minimum timestamp (`t_min`).
        - Duplicate timestamps within each `case_id` group are automatically dropped.
        - Interpolation uses linear extrapolation outside the observed time range."""


        
        # if interpolation:
        #     t_min = min(df['timestamp'].min(), query['timestamp'].min())
        #     t_max = max(df['timestamp'].max(), query['timestamp'].max())
            
        #     if np.issubdtype(df["timestamp"].dtype, np.datetime64):
        #         new_domain = np.linspace(0, (t_max - t_min).total_seconds(), N_grid)
        #     else:
        #         new_domain = np.linspace(0, t_max - t_min, N_grid)
            
        

        
        query_array = self.syncronise_over_time(query, self.new_domain, interpolation_type=interpolation_type)
        
        depth_array = np.empty((query_array.shape[0],), dtype = float)

        for i in range(query_array.shape[0]):
            #depth_array.append(int_depth(data_array, query_array[i, :, :], type_of_depth='halfspace', solver='neldermead', NRandom=100))
            depth_array[i] = self.int_depth(query_array[i, :, :], type_of_depth=type_of_depth, solver=solver, 
                                            NRandom=NRandom)[0]
       
        return depth_array

    
    