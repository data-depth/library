from . import multivariate as mvt
# from DepthEucl import DepthEucl
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Literal


class DepthFunc():
    """
    Functional Data-Depth
    
    Return the depth of each sample w.r.t. a dataset, D(x,data) in a functional space, using a chosen depth notion.

    Data depth computes the centrality (similarity, belongness) of a sample 'x' given a dataset 'data'.
    
    Notes
    -----
    Possible depth notions are : `mahalanobis`, `halfspace`, `zonoid`, `projection`, `aprojection`, `cexpchullstar`, `cexpchull`, `geometrical`.
    
    For each discretization point i = 1, ..., L:
        - Extract the data slice `data[:, i, :]` (shape: N_data x D)
        - Extract the query vector `x[i, :]` (shape: D)
        - Compute the multivariate depth of the query vector relative to the data slice
        - Average the results over all L time points
    
    """
    def __init__(self):
        self.data=None
    
    def load_dataset(self, data:pd.DataFrame=None, y:np.ndarray=None, 
                     timestamp_col:str|int='timestamp',value_cols:str|list|int='value', case_id:str|int="case_id", 
                     interpolate_grid:bool=True, N_grid:int=10,
                     interpolation_type:str="linear"):
        """
        Load the dataset X for reference calculations. Depth is computed with respect to this dataset.

        Parameters
        ----------
        data : dataframe or array_like.
            Dataset that will be used as base for depth computation
            
        y : Ignored, default=None
            Not used, present for API consistency by convention.
        
        timestamp_col : str|int, default = timestamp
            Column used for discretization of the dataset.
            timestamp_col can be a string indicating the name of the column or an integer for the position of the column.
            If the dataset is an array, timestamp_col must be an integer or it will not be considered and a new timestamp_col is created. 

        case_id : str | int, default= "case_id"
            Column used to separate different functions.
            For pandas dataframe, it must be either the column name or its position.
            For numpy array, it must be eather an integer for the position or it wil be considered as the dimension separation.

        value_cols: str|list, default = 'value'
            Columns used for the multivariate depth computation.
            If value_cols is a string, columns with such word in the name are used.
            If value_cols is a list, it is considered the list of columns to be used.
            If data is an array, value_cols must be an integer or a list of integers, if not all columns that are not timestamp or case_id will be considered.
        
        interpolate_grid : bool, default = True   
            Interpolates the timestamp grid using an equaly spaced array from the minimum to the maximum timestamp value.  

        N_grid : int, default = 10
            Determines the number of grid points in the interpolation process.


        interpolation_type : str, default='linear'
            Interpolation method to use. Supported options are those provided by
            `scipy.interpolate.interp1d`, including:
            - `'linear'` : linear interpolation (default)
            - `'nearest'` : nearest-neighbor interpolation
            - `'cubic'` : cubic spline interpolation
            - `'quadratic'` : quadratic spline interpolation
            - `'previous'`, `'next'` : stepwise interpolation

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
        assert type(data)==np.ndarray or type(data)==pd.DataFrame, 'Data must be a 3D numpy array of a pandas DataFrame'
        
        if type(data)==np.ndarray:
            self.TSnp=timestamp_col
            self.CInp=case_id
            data=self._3Dnp_tp_pd(data,self.TSnp, self.CInp)
            timestamp_col="timestamp"
            value_cols="value"
            case_id="case_id"

        if type(timestamp_col)==str:
            if timestamp_col in data.columns:    
                self.timestamp_col=timestamp_col
            else:
                raise NameError(f"timestamp_col is not a column in the data.")
        elif type(timestamp_col)==int:
            if timestamp_col>=len(data.columns):
                raise IndexError(f"timestamp_col is greater than the number of columns.")
            else:
                self.timestamp_col = data.columns[timestamp_col]
        print(f"timestamp_col is set to {self.timestamp_col}")
         
        if type(value_cols) == str: 
            self.value_cols = data.filter(like = value_cols).columns
            if len(self.value_cols)==0: raise ValueError("value_cols is an empty list, value_cols should be a suffix for different columns")
        elif type(value_cols) == list:
            self.value_cols = value_cols
        print(f"value_cols is set to {self.value_cols}")
        
        if type(case_id)==str:
            if case_id in data.columns:
                self.case_id=case_id
            else:
                raise ValueError("case_id is not a column of the data")
        elif type(case_id)==int:
            if case_id>=len(data.columns):
                raise IndexError(f"case_id is greater than the number of columns.")
            else:
                self.case_id = data.columns[case_id]
        print(f"case_id is set to {self.case_id}")
        
        self.t_min = data[self.timestamp_col].min()
        self.t_max = data[self.timestamp_col].max()
        data=data.sort_values([self.case_id,self.timestamp_col])
        data=self._MinMax(data)



        if interpolate_grid==True:
            self.N_grid = N_grid            
            if np.issubdtype(data[self.timestamp_col].dtype, np.datetime64):
                new_domain = np.linspace(0, (self.t_max - self.t_min).total_seconds(), self.N_grid)
            else:
                new_domain = np.linspace(0, self.t_max - self.t_min, self.N_grid)
            self.new_domain = new_domain
        # for i in np.unique(data.case_id)
        elif interpolate_grid==False:
            if np.issubdtype(data[self.timestamp_col].dtype, np.datetime64):
                self.new_domain=np.sort(np.unique((data[self.timestamp_col]-data[self.timestamp_col].min()).dt.total_seconds()))
            else:
                self.new_domain=np.sort(np.unique((data[self.timestamp_col]-data[self.timestamp_col].min())))
        
        self.interpolation_type=interpolation_type
        self.data_array = self._syncronise_over_time(data,True)
        self.data = data
        
        return self  
    
    def _3Dnp_tp_pd(self,data,timestamp_col, case_id,):
        assert len(data.shape)==3, "data must be a 3D numpy array or pandas dataframe"

        if type(timestamp_col)==str and timestamp_col!="timestamp":
            print(f"timestamp_col is set to {timestamp_col}, strings are only accepted with pandas dataframe.")
            print("timestamp_col is set to a equaly spaced grid")
            TS=np.arange(0,data.shape[-1])
            timestamp_col=-1
        elif type(timestamp_col)==int:
            if timestamp_col>data.shape[-1]:
                raise IndexError("timestamp_col is set to a value greater than the dataset")
        else: 
            print("timestamp_col must be an integer, it will not be considered for further computations")
            timestamp_col=-1
        if type(case_id)==str and case_id!="case_id":
            print(f"case_id is set to {case_id}, strings are only accepted with pandas dataframe.")
            print("case_id is set to the first dimension of the query.")
            case_id=-1
        elif type(case_id)==int:
            if case_id>data.shape[-1]:
                raise IndexError("case_id is set to a value greater than the dataset")
        else: 
            print("case_id must be an integer, it will not be considered for further computations")
            case_id=-1
        nCols=data.shape[-1]
        if timestamp_col==-1:
            timestamp_col=nCols
            nCols+=1
        if case_id==-1:
            case_id=nCols
            nCols+=1
        valCols=[f"value_{i}" for i in range(nCols-2)]
        df=pd.DataFrame(columns=["timestamp", "case_id", *valCols],dtype=np.float32)
        for i in range(data.shape[0]):
            # dfi=np.zeros((data.shape[1],nCols))
            dfi=pd.DataFrame(data=np.zeros((data.shape[1],nCols)),columns=["timestamp", "case_id", *valCols],dtype=np.float32)
            c=0
            for j in range(nCols):
                if j==timestamp_col:
                    if j>=data.shape[-1]:
                        dfi["timestamp"]=np.arange(data.shape[1],dtype=int)
                    else: 
                        dfi["timestamp"]=data[i,:,j]
                elif j==case_id:
                    if j>=data.shape[-1]:
                        dfi["case_id"]=np.ones(data.shape[1],dtype=int)*i
                    else: 
                        dfi["case_id"]=data[i,:,j]
                else:
                    # print(f"value_{c}")
                    dfi[f"value_{c}"]=data[i,:,j]
                    c+=1
            df=pd.concat([df,dfi],ignore_index=True).reset_index(drop=True)
            # print(dfi,timestamp_col)
        
        return df



    def _MinMax(self,data):
        """
        fix min max values for border times
        """
        for caso in np.unique(data[self.case_id]):
            # df[df.case_id==i][minT]
            # print(df[(df.case_id==i)&(df.timestamp==df[df.case_id==i]["timestamp"].min())][["case_id","timestamp",*val_cols]])
            if data[(data[self.case_id]==caso)][self.timestamp_col].min()==self.t_min:
                for col in self.value_cols:
                    if data[(data[self.case_id]==caso)&(data[self.timestamp_col]==self.t_min)][col].isna().values[0]:
                        data.loc[(data[self.case_id]==caso)&(data[self.timestamp_col]==self.t_min),col
                                 ]=data[(data[self.case_id]==caso)].loc[data[(data[self.case_id]==caso)][col].first_valid_index()][col]
                        
            else:
                firstCol=pd.DataFrame(columns=data.columns, )
                for col in data.columns:
                    if col==self.case_id:
                        firstCol.loc[0,col]=caso
                    elif col==self.timestamp_col:
                        firstCol.loc[0,col]=self.t_min
                    elif col in self.value_cols:
                        firstCol.loc[0,col]=data[(data[self.case_id]==caso)].loc[data[(data[self.case_id]==caso)][[col]].first_valid_index()][col]
                    else:
                        firstCol.loc[0,col]=data[(data[self.case_id]==caso)].loc[data[(data[self.case_id]==caso)][[col]].first_valid_index()][col]
                firstCol = firstCol.astype(data.dtypes.to_dict())
                # print(firstCol.dtypes)
                data=pd.concat([firstCol,data], ignore_index=False).reset_index(drop=True)
                # print(data[data[self.case_id]==caso])

                

            if data[(data[self.case_id]==caso)][self.timestamp_col].max()==self.t_max:
                for col in self.value_cols:
                    if data[(data[self.case_id]==caso)&(data[self.timestamp_col]==self.t_max)][col].isna().values[0]:
                        data.loc[(data[self.case_id]==caso)&(data[self.timestamp_col]==self.t_max),col
                                 ]=data[(data[self.case_id]==caso)].loc[data[(data[self.case_id]==caso)][col].last_valid_index()][col]
                        
            else:
                lastCol=pd.DataFrame(columns=data.columns, )
                for col in data.columns:
                    if col==self.case_id:
                        lastCol.loc[0,col]=caso
                    elif col==self.timestamp_col:
                        lastCol.loc[0,col]=self.t_max
                    elif col in self.value_cols:
                        lastCol.loc[0,col]=data.loc[data[(data[self.case_id]==caso)][col].last_valid_index(),col]
                    else :
                        lastCol.loc[0,col]=data.loc[data[(data[self.case_id]==caso)][col].last_valid_index(),col]
                lastCol = lastCol.astype(data.dtypes.to_dict())
                # print(lastCol)
            #     # print(firstCol.dtypes)
                data=pd.concat([data,lastCol], ignore_index=False).reset_index(drop=True)
        return data.sort_values([self.case_id,self.timestamp_col]).reset_index(drop=True)



    def _syncronise_over_time(self,df, interpolation=True, ):
        M = []
        if np.issubdtype(df[self.timestamp_col].dtype, np.datetime64):
            for case, group in df.groupby(self.case_id):
                group = group.loc[~group[self.timestamp_col].duplicated(keep='first')]

                if interpolation:
                    interp_values = self._interpolate(group[self.value_cols], 
                                                        (group[self.timestamp_col]-self.t_min).dt.total_seconds().to_numpy(), self.new_domain,
                                                        self.interpolation_type,axis = 0)
                else:
                    interp_values = np.asarray(group[self.value_cols])
                M.append(interp_values)
        else:
            for case, group in df.groupby(self.case_id):
                group = group.loc[~group[self.timestamp_col].duplicated(keep='first')]
                
                if interpolation:
                    interp_values = self._interpolate(group[self.value_cols], (group[self.timestamp_col]-self.t_min).to_numpy(), self.new_domain, axis = 0)
                else:
                    interp_values = np.asarray(group[self.value_cols])
                M.append(interp_values)
        M = np.stack(M, axis = 0)
        return M
    
    def _interpolate(self,value_matrix, original_domain, new_domain, method='linear', axis=0):
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
            # TODO check if more thqn 2 values are not null
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
            # TODO check if more thqn 2 values are not null
            f = interp1d(original_domain[non_nan_idx], flat_value_matrix[:, j][non_nan_idx], kind=method, bounds_error=False, fill_value="extrapolate")
            interp_flat_value_matrix[:, j] = f(new_domain)

        # Reshape back to original dimensions
        interp_value_matrix = interp_flat_value_matrix.reshape((len(new_domain),) + original_shape[1:])

        # Move axis back to original position
        if axis != 0:
            interp_value_matrix = np.moveaxis(interp_value_matrix, 0, axis)

        return interp_value_matrix
    

    def _int_depth(self, query_point, notion='halfspace', solver='neldermead', NRandom=100, option=1, **kwargs):
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
        n_refinements,sphcap_shrink,alpha_Dirichlet,cooling_factor,\
        cap_size,start,space,line_solver,bound_gc=self._check_hyperparDepth(**kwargs)
        if option==2: directions=np.zeros(query_point.shape)
        # if d==1:option=1
        for i in range(l_points):
            # data_component_slice: N_data x D matrix (all functions at time i)
            data_component_slice = self.data_array[:, i, :]

            # query_component: D-dimensional vector (query function at time i)
            query_component = query_point[i, :]
 
            # Compute depth at time i
            if option==1:
                time_component_depth = mvt.depth_approximation(
                    query_component, data_component_slice,
                    notion, solver, NRandom, 
                    option=option,
                    n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                    alpha_Dirichlet=alpha_Dirichlet,
                    cooling_factor=cooling_factor,cap_size=cap_size,
                    start=start,space=space,line_solver=line_solver,bound_gc=bound_gc
                )
            if option==2:
                time_component_depth, directions[i] = mvt.depth_approximation(
                        query_component, data_component_slice,
                        notion, solver, NRandom, 
                        option=2,
                        n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                        alpha_Dirichlet=alpha_Dirichlet,
                        cooling_factor=cooling_factor,cap_size=cap_size,
                        start=start,space=space,line_solver=line_solver,bound_gc=bound_gc
                    )
            total_depth_sum += np.nan_to_num(time_component_depth,nan=0)

        # Average depth over all L time points
        functional_depth_val = total_depth_sum / l_points

        if option==1:return functional_depth_val 
        else:return functional_depth_val,directions

    
    def projection_based_func_depth(self, query,
                                    notion='halfspace', solver='neldermead', NRandom=100,
                                    output_option:Literal["lowest_depth","final_depth_dir"]="lowest_depth", **kwargs):
        """
        Compute projection-based functional depth for query functional data with respect to a reference dataset.

        This function computes depth values of functional observations (in `query`) relative to a 
        reference dataset (`df`) using projection-based methods such as halfspace depth.
        Each function (trajectory) is represented by a sequence of multivariate values over time.
        
        Parameters
        ----------
        query : pandas.DataFrame
            Query dataset containing functional observations whose depth will be computed
            relative to `df`. Must have the same column structure as `df`.

        notion : {"mahalanobis", "halfspace", "zonoid", "projection", "aprojection", "cexpchullstar", "cexpchull", "geometrical"}, default='halfspace'
            Type of functional depth to compute. Currently supports 'halfspace', but 
            can be extended to other projection-based depths.

        solver : str, default='neldermead'
            {'simplegrid', 'refinedgrid', 'simplerandom', 'refinedrandom', 'coordinatedescent', 'randomsimplices', 'neldermead', 'simulatedannealing'},
            The type of solver used to approximate the depth.
            Optimization solver used within the internal depth computation.

        NRandom : int, default=100
            Number of random projections or optimization restarts used in computing 
            projection-based depth.

        n_refinements  
            For ``solver`` = ``refinedrandom`` or ``refinedgrid``, set the maximum of iteration for 
            computing the depth of one point. 
            
        sphcap_shrink  
            For ``solver`` = ``refinedrandom`` or `refinedgrid`, it's the shrinking of the spherical cap. 
            
        alpha_Dirichlet  
            For ``solver`` = ``randomsimplices``. it's the parameter of the Dirichlet distribution. 
            
        cooling_factor  
            For ``solver`` = ``randomsimplices``, it's the cooling factor. 
            
        cap_size 
            For ``solver`` = ``simulatedannealing`` or ``neldermead``, it's the size of the spherical cap. 
            
        start 
            {'mean', 'random'}, 
            For ``solver`` = ``simulatedannealing`` or ``neldermead``, it's the method used to compute the first depth.
            
        space  
            {'sphere', 'euclidean'}, 
            For ``solver`` = ``coordinatedescent`` or ``neldermead``, it's the type of spacecin which
            the solver is running.
            
        line_solver 
            {'uniform', 'goldensection'}, 
            For ``solver`` = ``coordinatedescent``, it's the line searh strategy used by this solver.
            
        bound_gc 
            For ``solver`` = ``neldermead``, it's ``True`` if the search is limited to the closed hemisphere.


        Returns
        -------
        depth_array : np.ndarray of shape (n_query,)
        Array of depth values, where `n_query` is the number of functional observations 
        (unique `case_id`s) in the `query` dataset.
        The first return is the lowest comuted depth regarding all explored directions in space.
        The second return is the direction that best represents the analyzed point, the direction corresponfing to the lowest depth.

            If ``output_option=="lowest_depth"`` returns:
                array_like
                    - Lowest Asymmetrical Projection Detph
        
            If ``output_option=="final_depth_dir"`` returns:
                Tuple of array_like
                    - Lowest Asymmetrical Projection Detph
                    - Lowest depth respective sirection

        Notes
        -----
        - If `timestamp` is of type `datetime64`, it is converted internally to seconds
        relative to the global minimum timestamp (`t_min`).
        - Duplicate timestamps within each `case_id` group are automatically dropped.
        - Interpolation uses linear extrapolation outside the observed time range."""
  
        

        
        self._check_depth(notion)
        if type(query)==np.ndarray:
            query=self._3Dnp_tp_pd(query,self.TSnp, self.CInp)
        if query[self.timestamp_col].max()>self.t_max:
            print(f"Values with {self.timestamp_col} greater the base set domain are excluded")
            query.drop(query[query[self.timestamp_col]>self.t_max].index, inplace=True)
        if query[self.timestamp_col].min()>self.t_min:
            print(f"Values with {self.timestamp_col} smaller the base set domain are excluded")
            query.drop(query[query[self.timestamp_col]<self.t_min].index, inplace=True)
        queryMM=self._MinMax(query)
        query_array = self._syncronise_over_time(queryMM,)
        depth_array = np.empty((query_array.shape[0],), dtype = float)
        if len(self.value_cols)==1 and output_option=="final_depth_dir":
            print("Space dimention is 1, output_option is set to 'lowest_depth'")
            output_option="lowest_depth"
        if output_option=="lowest_depth":
            option=1
            direction_array=None
        elif output_option=="final_depth_dir":
            option=2
            direction_array = np.empty(query_array.shape, dtype = float)
        else:
            option=1
            print(f"Invalid output_option, output_option set to 'lowest_depth'")
        for i in range(query_array.shape[0]):
            if option==1:
                depth_array[i] = self._int_depth(query_array[i, :, :], notion=notion, solver=solver,option=option,
                                            NRandom=NRandom,**kwargs)[0]
            
            elif option==2:
                depth_array[i], direction_array[i] =self._int_depth(query_array[i, :, :], notion=notion, solver=solver,option=option,
                                            NRandom=NRandom,**kwargs)
                
       
        if option==1:return depth_array 
        else:return depth_array,direction_array

    def general_func_depth(self,query,notion='halfspace', **kwargs):
        """
        Compute non projection-based functional depth for query functional data with respect to a reference dataset.

        This function computes depth values of functional observations (in `query`) relative to a 
        reference dataset (`df`) using projection-based methods such as halfspace depth.
        Each function (trajectory) is represented by a sequence of multivariate values over time.
        
        Parameters
        ----------
        query : pandas.DataFrame
            Query dataset containing functional observations whose depth will be computed
            relative to `df`. Must have the same column structure as `df`.

        notion 
            {"mahalanobis", "halfspace", "zonoid", "projection", "aprojection", "cexpchullstar", "cexpchull", "geometrical"},
            Which depth will be computed.


        Returns
        -------
        depth_array : np.ndarray of shape (n_query,)
        Array of depth values, where `n_query` is the number of functional observations 
        (unique `case_id`s) in the `query` dataset.
        The return is the lowest comuted depth regarding all explored directions in space.

        Notes
        -----
        - If `timestamp` is of type `datetime64`, it is converted internally to seconds
        relative to the global minimum timestamp (`t_min`).
        - Duplicate timestamps within each `case_id` group are automatically dropped.
        - Interpolation uses linear extrapolation outside the observed time range.
        """
        self._check_depth(notion)
        if type(query)==np.ndarray:
            query=self._3Dnp_tp_pd(query,self.TSnp, self.CInp)
        if query[self.timestamp_col].max()>self.t_max:
            print(f"Values with {self.timestamp_col} greater the base set domain are excluded")
            query.drop(query[query[self.timestamp_col]>self.t_max].index, inplace=True)
        if query[self.timestamp_col].min()>self.t_min:
            print(f"Values with {self.timestamp_col} smaller the base set domain are excluded")
            query.drop(query[query[self.timestamp_col]<self.t_min].index, inplace=True)
        queryMM=self._MinMax(query)
        query_array = self._syncronise_over_time(queryMM,)
        depth_array = np.empty((query_array.shape[0],), dtype = float)
        pass

    def _check_hyperparDepth(self,**kwargs):
        n_refinements = 10
        sphcap_shrink = 0.5
        alpha_Dirichlet = 1.25
        cooling_factor = 0.95
        cap_size = 1
        start = "mean"
        space = "sphere"
        line_solver = "goldensection"
        bound_gc = True

        for key, value in kwargs.items():
            if key=="n_refinements":
                n_refinements=value
            elif key=="sphcap_shrink":
                sphcap_shrink=value
            elif key=="alpha_Dirichlet":
                alpha_Dirichlet=value
            elif key=="cooling_factor":
                cooling_factor=value
            elif key=="cap_size":
                cap_size=value
            elif key=="start":
                start=value
            elif key=="space":
                space=value
            elif key=="line_solver":
                line_solver=value
            elif key=="bound_gc":
                bound_gc=value
            else:
                print(f"{key} is not a parameter for depth computation")
            
        return n_refinements,sphcap_shrink,alpha_Dirichlet,cooling_factor,cap_size,start,space,line_solver,bound_gc

    def _check_depth(self, depth):
        all_depths = ["mahalanobis", "halfspace", "zonoid", "projection", "aprojection", "cexpchullstar", "cexpchull", "geometrical"]
        if (depth not in all_depths):
            raise ValueError("Depths approximation is available only for depths in %s, got %s."%(all_depths, depth))    
        