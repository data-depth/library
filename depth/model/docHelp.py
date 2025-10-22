mahalanobis__doc__= """

Calculates the Mahalanobis depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    exact : bool, delfaut=True
        The type of the used method. The default is ``exact=False``, which leads to approx-
        imate computation of the Mahalanobis depth using the method defined by the argument ``solver``.
        If ``exact=True``, the Mahalanobis depth is computed exactly, using the closed-form expression.

    mah_estimate : str, {"moment", "mcd"}, default="moment"
        A character string specifying which estimates to use when calculating the Mahalanobis depth; can be "'moment'" or ``'MCD'``,
        determining whether traditional moment or Minimum Covariance Determinant (MCD) 
        estimates for mean and covariance are used. By default ``'moment'`` is used.

    mah_parMcd : float, default=0.75
        is the value of the argument alpha for the function covMcd; is used when
        mah.estimate = ``'MCD'``.
    
    solver : str, default="neldermead"
        The type of solver used to approximate the depth.
        {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}

    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Mahalanobis, P. C. (1936). On the generalized distance in statistics. *Proceedings of the National Institute of Sciences of India*, 12, 49–55.
    
    * Mosler, K. and Mozharovskyi, P. (2022). Choosing among notions of multivariate depth statistics. *Statistical Science*, 37(3), 348-368.

Examples
        >>> import numpy as np
        >>> from depth.model import DepthEucl
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> np.random.seed(0)
        >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> model = DepthEucl().load_dataset(data)
        >>> model.mahalanobis(x)
        [0.17849871 0.10412453 0.1331417  0.13578021 0.3154836  0.29103769
            0.13398989 0.13913017 0.59339051 0.10556139]
        >>> model.mahalanobisDepth
        [0.17849871 0.10412453 0.1331417  0.13578021 0.3154836  0.29103769
            0.13398989 0.13913017 0.59339051 0.10556139]
        >>> model.mahalanobis(x, exact="True", mah_estimate="MCD", mah_parMcd = 0.75)
        [0.17758703 0.10367974 0.131705   0.13575221 0.31847867 0.29034948
            0.13291613 0.13792774 0.59094958 0.10491694]

"""
aprojection__doc__="""

Calculates approximately the asymmetric projection depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    solver : str  {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}, default="neldermead"
        The type of solver used to approximate the depth.
    
    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset


References
    * Dyckerhoff, R. (2004). Data depths satisfying the projection property. *Allgemeines Statistisches Archiv*, 88, 163–190.
    
    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
            >>> import numpy as np
            >>> from depth.model import DepthEucl
            >>> np.random.seed(0)
            >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
            >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
            >>> model = DepthEucl().load_dataset(data)
            >>> model.aprojection(x, NRandom=1000)
            [0.090223   0.19577999 0.15769263 0.20123535 0.10375507 0.14635662
             0.20611053 0.17846703 0.19801984 0.23230606]             
"""
betaSkeleton__doc__= """ 

Calculates the beta-skeleton depth of points w.r.t. a multivariate data set.

Arguments
    x : {array-like} of shape (n_samples,d).
        Matrix of objects (numerical vector as one object) whose depth is to be calculated. 
        Each row contains a d-variate point and should have the same dimension as data.

	beta : int, default=2
        The parameter defining the positionning of the balls’ centers, see `Yang and Modarres (2017)`_ for details.
        By default (together with other arguments) equals
        ``2``, which corresponds to the lens depth, see Liu and Modarres (2011).

	distance : str, default='Lp'
        A character string defining the distance to be used for determining inclusion
        of a point into the lens (influence region), see Yang and Modarres (2017) for
        details. Possibilities are ``'Lp'`` for the Lp-metric (default) or ``'Mahalanobis'`` for
        the Mahalanobis distance adjustment.

	Lp_p : int, default=2
			A non-negative number defining the distance’s power equal ``2`` by default (Euclidean distance)
			is used only when ``distance='Lp'``.

    mah_estimate : str, {"moment", "mcd"}, default="moment"
			A character string specifying which estimates to use when calculating sample
			covariance matrix; can be ``'none'``, ``'moment'`` or ``'MCD'``, determining whether
			traditional moment or Minimum Covariance Determinant (MCD)
			estimates for mean and covariance are used. By default ``'moment'`` is used. Is
			used only when ``distance='Mahalanobis'``.

    mah_parMcd : float, default=0.75
			The value of the argument alpha for Minimum Covariance Determinant (MCD); is used when ``distance='Mahalanobis'`` and ``mah.estimate='MCD'``.
    
    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Elmore, R. T., Hettmansperger, T. P. and Xuan, F. (2006). Spherical data depth and a multivariate median. In R. Y. Lui, R. Serfling, and D. L. Souvaine, (Eds.), *Data Depth: Robust Multivariate Analysis, Computational Geometry and Applications*, *DIMACS Series Discrete Mathematics and Theoretical Computer Science*, 72, American Mathematical Society, Providence, RI, 87–101.
    
    * Liu, Z. and Modarres, R. (2011). Lens data depth and median. *Journal of Nonparametric Statistics*, 23, 1063–1074.
    
    * Kleindessner, M. and Von Luxburg, U. (2017). Lens depth function and k-relative neighborhood graph: Versatile tools for ordinal data analysis. *Journal of Machine Learning Research*, 18, 58, 52.
    
    * Yang, M. and Modarres, R. (2018). :math:`{\\beta}`-skeleton depth functions and medians. *Communications in Statistics - Theory and Methods*, 47, 5127–5143.

Examples
			>>> import numpy as np
			>>> from depth.model import DepthEucl
			>>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
			>>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
			>>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
			>>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
			>>> model=DepthEucl().load_dataset(data)
			>>> model.BetaSkeleton(x)
			[0.16467668 0.336002   0.43702102 0.25827828 0.4204044  0.46894895
 			0.27825225 0.11572372 0.4663003  0.18778579]

"""

cexpchull__doc__="""

Calculates approximately the continuous explected convex hull depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    solver : str  {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}, default="neldermead"
        The type of solver used to approximate the depth.
    
    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Dyckerhoff, R. and Mosler, K. (2011). Weighted-mean trimming of multivariate data. *Journal of Multivariate Analysis*, 102, 405–421.
    
    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
            >>> import numpy as np
            >>> from depth.model import DepthEucl
            >>> np.random.seed(0)
            >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
            >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
            >>> mode=DepthEucl().load_dataset(data)
            >>> mode.cexpchull(x, data, NRandom=1000)
            [0.090223   0.19577999 0.15769263 0.20123535 0.10375507 0.14635662
             0.20611053 0.17846703 0.19801984 0.23230606]             
"""

cexpchullstar__doc__="""

Calculates approximately the continuous modified explected convex hull depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    solver : str  {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}, default="neldermead"
        The type of solver used to approximate the depth.
    
    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Dyckerhoff, R. and Mosler, K. (2011). Weighted-mean trimming of multivariate data. *Journal of Multivariate Analysis*, 102, 405–421.
    
    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
            >>> import numpy as np
            >>> from depth.model import DepthEucl
            >>> np.random.seed(0)
            >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
            >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
            >>> model=DepthEucl().load_dataset(data)
            >>> model.cexpchull(x, NRandom=1000)
            [0.090223   0.19577999 0.15769263 0.20123535 0.10375507 0.14635662
             0.20611053 0.17846703 0.19801984 0.23230606]             
"""

geometrical__doc__="""

Calculates approximately the geometrical depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    solver : str  {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}, default="neldermead"
        The type of solver used to approximate the depth.
    
    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Dyckerhoff, R. and Mosler, K. (2011). Weighted-mean trimming of multivariate data. *Journal of Multivariate Analysis*, 102, 405–421.
    
    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
            >>> import numpy as np
            >>> from depth.model import DepthEucl
            >>> np.random.seed(0)
            >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
            >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
            >>> model=DepthEucl().load_dataset(data)
            >>> model.geometrical(x, NRandom=1000)
            [0.090223   0.19577999 0.15769263 0.20123535 0.10375507 0.14635662
             0.20611053 0.17846703 0.19801984 0.23230606]             
"""

halfspace__doc__="""

Calculates the exact and approximated Tukey (=halfspace, location) depth (Tukey, 1975) of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    exact : bool, default=False
        The type of the used method. The default is ``exact=False``, which leads to approx-
        imate computation of the Tukey depth.
        If ``exact=True``, the Tukey depth is computed exactly, with ``method='recursive'`` by default.
    
    method: str, default='recursive'			
        For ``exact=True``, the Tukey depth is calculated as the minimum over all combinations of k points from data (see Details below).
        In this case parameter method specifies k, with possible values 1 for ``method='recursive'`` (by default), d−2
        for ``method='plane'``, d−1 for ``'method=line'``.
        The name of the method may be given as well as just parameter exact, in which
        case the default method will be used.

    solver : str  {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}, default="neldermead"
        The type of solver used to approximate the depth.
    
    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    CUDA : bool, default=False
        Determines if approximate computation will be performed in GPU.
        avaiable only for simplerandom or refinedrandom

    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Tukey, J. W. (1975). Mathematics and the picturing of data. In R. James (Ed.), *Proceedings of the International Congress of Mathematicians*, Volume 2, Canadian Mathematical Congress, 523–531.
    
    * Donoho, D. L. and M. Gasko (1992). Breakdown properties of location estimates based on halfspace depth and projected outlyingness. *The Annals of Statistics*, 20(4), 1803–1827.
    
    * Dyckerhoff, R. and Mozharovskyi, P. (2016): Exact computation of the halfspace depth. *Computational Statistics and Data Analysis*, 98, 19–30.

    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
        >>> import numpy as np
        >>> from depth.model import DepthEucl
        >>> mat1=[[1, 0, 0],[0, 2, 0],[0, 0, 1]]
        >>> mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        >>> x = np.random.multivariate_normal([1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0], mat1, 200)
        >>> model=DepthEucl().load_dataset(data)
        >>> model.halfspace(x,)
        [0.    0.005 0.005 0.    0.04  0.01  0.    0.    0.04  0.01 ]
        >>> model.halfspace(x, exact=True)
        [0.    0.005 0.005 0.    0.04  0.01  0.    0.    0.04  0.01 ]

"""
    
L2__doc__=""" 

Calculates the L2-depth of points w.r.t. a multivariate data set.
			
Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    mah_estimate : str, {"moment", "mcd"}, default="moment"
        A character string specifying which estimates to use when calculating the Mahalanobis depth; can be "'moment'" or ``'MCD'``,
        determining whether traditional moment or Minimum Covariance Determinant (MCD) 
        estimates for mean and covariance are used. 

    mah_parMcd : float, default=0.75
        is the value of the argument alpha for the function covMcd; is used when
        mah.estimate = ``'MCD'``.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset
    
			
References
    * Zuo, Y. and Serfling, R. (2000). General notions of statistical depth function. *The Annals of Statistics*, 28, 461–482.
    
    * Mosler, K. and Mozharovskyi, P. (2022). Choosing among notions of multivariate depth statistics. *Statistical Science*, 37(3), 348-368.
   
Examples
			>>> import numpy as np
			>>> from depth.model import DepthEucl
			>>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
			>>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
			>>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
			>>> model=DepthEucl().load_dataset(data)
			>>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
			>>> model.L2(x)
			[0.2867197  0.19718391 0.18896649 0.24623271 0.20979579 0.22055673
 			0.20396566 0.20779032 0.24901829 0.26734192]

"""

potential__doc__="""

Calculate the potential of the points w.r.t. a multivariate data set. The potential is the kernel-estimated density multiplied by the prior probability of a class. Different from the data depths, a density estimate measures at a given point how much mass is located around it.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.
        
    pretransform: str, default="1Mom"
        The method of data scaling.
        ``'1Mom'`` or ``'NMom'`` for scaling using data moments.
        ``'1MCD'`` or ``'NMCD'`` for scaling using robust data moments (Minimum Covariance Determinant (MCD).
    
    kernel: str, default="EDKernel"
        ``'EDKernel'`` for the kernel of type 1/(1+kernel.bandwidth*EuclidianDistance2(x,y)),
        ``'GKernel'`` [default and recommended] for the simple Gaussian kernel,
        ``'EKernel'`` exponential kernel: exp(-kernel.bandwidth*EuclidianDistance(x, y)),
        ``'VarGKernel'`` variable Gaussian kernel, where kernel.bandwidth is proportional to the depth.zonoid of a point.
    
    kernel_bandwidth: int, default=0
        the single bandwidth parameter of the kernel. If ``0`` - the Scott`s rule of thumb is used.

    mah_parMcd : float, default=0.75
        Value of the argument alpha for the function covMcd
        
    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Pokotylo, O. and Mosler, K. (2019). Classification with the pot-pot plot. *Statistical Papers*, 60, 903-931.
			
Examples
			>>> import numpy as np
			>>> from depth.model import DepthEucl
			>>> mat1=[[1, 0, 0],[0, 2, 0],[0, 0, 1]]
			>>> mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
			>>> data = np.random.multivariate_normal([0,0,0], mat1, 20)
			>>> model=DepthEucl().load_dataset(data)
			>>> x = np.random.multivariate_normal([1,1,1], mat2, 10)
			>>> model.potential(x,)
			[7.51492797 8.34322926 5.42761506 6.25418171 4.25774485 8.09733146
 			6.65788017 5.11324521 5.74407939 9.26030661]
			>>> model.potential(x, kernel_bandwidth=0.1)
			[13.56510469 13.95553893 11.23251702 12.42491604 10.17527509 13.70947682
 			12.67352469 11.2080649  11.73402562 14.93067103]
			>>> model.potential(x, pretransform = "NMCD", mah_parMcd=0.6, kernel_bandwidth=0.1)
			[11.0603282  11.49509828  8.99303793  8.63168006  7.86456928 11.03588551
 			10.45468945  8.84989798  9.56799496 12.29832608]
"""
projection__doc__="""


Calculates approximately the projection depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    solver : str  {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}, default="neldermead"
        The type of solver used to approximate the depth.
    
    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    CUDA : bool, default=False
        Determines if approximate computation will be performed in GPU.
        avaiable only for simplerandom or refinedrandom

    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Zuo, Y. and Serfling, R. (2000). General notions of statistical depth function. *The Annals of Statistics*, 28, 461–482.

    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
        >>> import numpy as np
        >>> from depth.model import DepthEucl
        >>> np.random.seed(0)
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
        >>> model=DepthEucl().load_dataset(data)
        >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
        >>> model.projection(x, NRandom=1000)
        [0.090223   0.19577999 0.15769263 0.20123535 0.10375507 0.14635662
        0.20611053 0.17846703 0.19801984 0.23230606]             
"""

qhpeeling__doc__= """

Calculates the convex hull peeling depth of points w.r.t. a multivariate data set.

Usage
    qhpeeling(x, data)

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.
    
    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset
       
References
    * Barnett, V. (1976). The ordering of multivariate data. *Journal of the Royal Statistical Society*, *Series A*, 139, 318–355.
    
    * Eddy, W. F. (1981). Graphics for the multivariate two-sample problem: Comment. *Journal of the American Statistical Association*, 76, 287–289.
            
Examples
            >>> from depth.model import DepthEucl
            >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
            >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
            >>> model=DepthEucl().load_dataset(data)
            >>> model.qhpeeling(x)
            [0.   0.   0.   0.   0.   0.   0.01 0.   0.   0.01]

"""

simplicial__doc__ = """

Calculates the simplicial depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.
    
   exact: bool, default=True 			
            ``exact=True`` (by default) implies the exact algorithm, ``exact=False`` implies the approximative algorithm, considering k simplices.

    k : float or int, default=0.05
            |	Number (``k > 1``) or portion (if ``0 < k < 1``) of simplices that are considered if ``exact=False``.
            |	If ``k > 1``, then the algorithmic complexity is polynomial in d but is independent of the number of observations in data, given k. 
            |	If ``0 < k < 1``,then the algorithmic complexity is exponential in the number of observations in data, but the calculation precision stays approximately the same.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Liu , R. Y. (1990). On a notion of data depth based on random simplices. *The Annals of Statistics*, 18, 405–414.

Examples
            >>> import numpy as np
            >>> from depth.model import DepthEucl
            >>> mat1=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
            >>> mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0], mat1, 25)
            >>> model=DepthEucl().load_dataset(data)
            >>> model.simplicial(x,)
            [0.04458498 0.         0.         0.         0.         0.
             0.         0.         0.         0.        ]
"""

simplicialVolume__doc__="""

Calculates the simpicial volume depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    exact : bool, default=True			
            ``exact=True`` (by default) implies the **exact algorithm**, ``exact=False`` implies the **approximative algorithm**, considering k simplices.

    k: float or int, default=0.05		
            |	Number (``k > 1``) or portion (if ``0 < k < 1``) of simplices that are considered if ``exact = F``.
            |	If ``k > 1``, then the algorithmic complexity is polynomial in d but is independent of the number of observations in data, given k. 
            |	If ``0 < k < 1``, then the algorithmic complexity is exponential in the number of observations in data, but the calculation precision stays approximately the same.

    mah_estimate : str, {"moment", "mcd"}, default="moment"
        A character string specifying which estimates to use when calculating the Mahalanobis depth; can be "'moment'" or ``'MCD'``,
        determining whether traditional moment or Minimum Covariance Determinant (MCD) 
        estimates for mean and covariance are used. 

    mah_parMcd : float, default=0.75
        is the value of the argument alpha for the function covMcd; is used when
        mah.estimate = ``'MCD'``.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Oja, H. (1983). Descriptive statistics for multivariate distributions. *Statistics and Probability Letters*, 1, 327–332.

Examples
            >>> import numpy as np
            >>> from depth.model import DepthEucl
            >>> mat1=[[1, 0, 0],[0, 2, 0],[0, 0, 1]]
            >>> mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0], mat1, 20)
            >>> model=DepthEucl().load_dataset(data)
            >>> model.simplicalVolume(x, exact=True)
            [0.45749049 0.34956166 0.2263421  0.68742137 0.94796538 0.51112415
             0.85250931 0.67914988 0.79165292 0.33192247]
            >>> model.simplicalVolume(x, exact=False, k=0.2)
            [0.46826813 0.40138917 0.23189724 0.69025277 0.938543   0.56005713
             0.8113647  0.72220103 0.82036139 0.33908597]
"""

spatial__doc__=""" 

Calculates the spatial depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    mah_estimate : str, {"moment", "mcd"}, default="moment"
        A character string specifying which estimates to use when calculating the Mahalanobis depth; can be "'moment'" or ``'MCD'``,
        determining whether traditional moment or Minimum Covariance Determinant (MCD) 
        estimates for mean and covariance are used. 

    mah_parMcd : float, default=0.75
        is the value of the argument alpha for the function covMcd; is used when
        mah.estimate = ``'MCD'``.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Serfling, R. (2002). A depth function and a scale curve based on spatial quantiles. In Dodge, Y. (Ed.), *Statistical Data Analysis Based on the L1-Norm and Related Methods*, *Statisctics in Industry and Technology*, Birkhäuser, Basel, 25–38.

Examples
        >>> import numpy as np
        >>> from depth.model import DepthEucl
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> model=DepthEucl().load_dataset(data)
        >>> model.spatial(x, )
        [0.22548919451212823, 0.14038895785356165, 0.2745517635029123, 0.35450156620496354,
        0.42373722245348566, 0.34562025044812095, 0.3585616673301636, 0.16916309940691643,
        0.573349631625784, 0.32017213635679687]

"""

zonoid__doc__= """

Calculates the zonoid depth of points w.r.t. a multivariate data set.

Arguments
    x: array-like or None, default=None 
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    solver : str  {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}, default="neldermead"
        The type of solver used to approximate the depth.
    
    NRandom : int, default=1000
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.         

    n_refinements : int, default = 10
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink : float, default = 0.5
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet : float, default = 1.25
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor : float, default = 0.95
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size : int | float, default = 1
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start : str {'mean', 'random'}, default = mean 
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space : str {'sphere', 'euclidean'}, default = 'sphere' 
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver : str {'uniform', 'goldensection'}, default = goldensection
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc : bool, default = True
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.
    
    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
        |        If ``output_option=lowest_depth``, only approximated depths are returned.
        |        If ``output_option=final_depht_dir``, best directions to approximate depths are also returned.
        |        If ``output_option=all_depth``, depths calculated at every iteration are also returned.
        |        If ``output_option=all_depth_directions``, random directions used to project depths are also returned with indices of converging for the solver selected.

    evaluate_dataset : bool, default=False
        Determines if dataset loaded will be evaluated. 
        Automatically sets x to dataset

References
    * Dyckerhoff, R., Koshevoy, G. and Mosler, K. (1996). Zonoid data depth: Theory and computation. In A. Pratt, (Ed.), COMPSTAT 1996, *Proceedings in Computational Statistics*, Physica-Verlag, Heidelberg, 235–240.
    
    * Koshevoy, G. and Mosler, K. (1997). Zonoid trimming for multivariate distributions. *The Annals of Statistics*, 25, 1998–2017.
    
    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
        >>> import numpy as np
        >>> from depth.model import DepthEucl
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> model=DepthEucl().load_dataset(data)
        >>> model.zonoid(x,)
        [0.         0.00769552 0.03087017 0.         0.30945453 0.0142515
         0.         0.01970896 0.02169483 0.        ]

"""
change_dataset__doc__="""

    Modify loaded dataset.

    Arguments
        newDataset:np.ndarray
            New dataset
        
        newDistribution:np.ndarray|None, default=None,
            Distribution related to the dataset
        
        newY:np.ndarray|None, default=None,
            Only for convention.
        
        keepOld:bool, default=False,
            Boolean to determine if current dataset is kept or not.
            If True, newDataset is added in the end of the old one.
    Returns 
        None
    
    Examples
        >>> import numpy as np
        >>> from depth.model import DepthEucl
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> np.random.seed(0)
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> data2 = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> model = DepthEucl().load_dataset(data)
        >>> model.change_dataset(data2,) 
    """

ACA__doc__="""

    Computes the abnormal component analysis
            
    Arguments
        dim: int, default=2
            Number of dimensions to keep in the reduction
        
        sample_size: int, default=None
            Size of the dataset (uniform sampling) to be used in the ACA calculation

        sample: list[int], default=None
            Indices for the dataset to be used in the computation 
        
        notion: str {``'projection'``, ``'halfspace'``}, default="projection"
            Chosen notion for depth computation
        
        solver : str  {``'refinedrandom'``, ``'neldermead'``}, default="neldermead"
            The type of solver used to approximate the depth.
    
        NRandom : int, default=1000
            The total number of iterations to compute the depth. Some solvers are converging
            faster so they are run several time to achieve ``NRandom`` iterations.         

        n_refinements : int, default = 10
            Set the maximum of iteration for computing the depth of one point.
            For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                        
        sphcap_shrink : float, default = 0.5
            It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

        alpha_Dirichlet : float, default = 1.25
            It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

        cooling_factor : float, default = 0.95
            It's the cooling factor. For ``solver='simulatedannealing'``.

        cap_size : int | float, default = 1
            It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

        start : str {'mean', 'random'}, default = mean 
            For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                        
        space : str {'sphere', 'euclidean'}, default = 'sphere' 
            For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                        
        line_solver : str {'uniform', 'goldensection'}, default = goldensection
            For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                        
        bound_gc : bool, default = True
            For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.

    Results
        ACA directions for dimensional reduction
    
    References
        * Valla, R., Mozharovskyi, P., & d'Alché-Buc, F. (2023). Anomaly component analysis. arXiv preprint arXiv:2312.16139.
  
    Examples
        >>> import numpy as np
        >>> from depth.model import DepthEucl
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> np.random.seed(0)
        >>> data1 = np.random.multivariate_normal([0,0,0,0,0], mat1, 990)
        >>> data2 = np.random.multivariate_normal([0,1,1,0,0], mat2, 10)
        >>> data=np.concat((data1,data2),axis=0)
        >>> model = DepthEucl().load_dataset(data)
        >>> model.ACA(dim=2, sample_size=900)
            array([[-0.13558675, -0.13558675],
                    [ 2.65800844,  2.65800844],
                    [-1.38230018, -1.38230018],
                    [-0.11503065, -0.11503065],
                    [ 0.55349281,  0.55349281]])

    """

