Presentation
============


Abstract
--------

Following the seminal idea of Tukey (1975), data depth is a function that measures how close an arbitrary point of the space is located to an implicitly defined center of a data cloud. Having undergone theoretical and computational developments, it is now employed in numerous applications. The :math:`\texttt{data-depth}` library is a software directed to fuse experience of the applicant with recent achievements in the area of data depth. Library :math:`\texttt{data-depth}` provides an implementation for exact and approximate computation of most reasonable and widely applied notions of data depth.


General description of the library
----------------------------------

Learning from data has illustrated uncountable successes in the past decades and became a general methodology for knowledge extraction from real-world observations. Its approaches rely profoundly on providing a meaningful ordering of data. In unsupervised machine learning, data ordering uncovers the structure of raw, unlabelled data, e.g., by data clustering or anomaly detection. It measures degree of adherence to a class in the supervised learning. In data analysis, data ordering explores the geometry of data, allows for their relevant visualisation, estimation of location and scatter, and statistical inference. Intrinsically based on data ordering, cumulative distribution function, quantiles, and ranks are ubiquitous for presentation of data or summary of analysis and crucial for definition of losses in machine learning models.

Introduced by John W. Tukey, data depth is a statistical function that measures centrality of an observation with respect to distribution, with an empirical distribution (on a data set) being its particular case. Consider a data set, say :math:`\textit{X} = {x_1,...,x_n} \, \in \, \mathbb{R}^d`. For an arbitrary point of the same space, :math:`x \, \in \, \mathbb{R}^d`, data depth is a function of :math:`(x,\textit{X})` that maps to :math:`D(x,\textit{X})`, and returns a value between 0 and 1. This value characterises how central :math:`x` is located in :math:`\textit{X}`. The higher the value of the depth, the more central is :math:`x` in :math:`\textit{X}`, and vice versa low values of depth indicate distributional farness of :math:`x`.

By exploiting the geometry of data, the depth function is fully non-parametric, robust to both outliers and heavy tailed distributions, satisfies desirable invariances (like, e.g., affine invariance), and is used in a variety of tasks as a generalisation of quantiles in higher dimensions and an alternative to the probability density. Nowadays, these advantages make data depth vital for many applications: supervised and unsupervised machine learning, robust optimisation, financial risk assessment, statistical quality control, extreme value theory, imputation of missing data, to name but a few. Many notions to define :math:`D(x,\textit{X})` have been developed, with halfspace, simplicial, spatial or projection depths being the most popular ones. For further information, the applicant is referred to surveys, with numerous works appearing each year in this active and rapidly growing field.

Specifications
--------------

12 multivariate depths are implemented in the :math:`\texttt{data-depth}` library, in both exact and approximate - to avoid excessive computational time - version:

 - :ref:`asymmetric projection <Aprojection>` depth [approximate];
 - :ref:`beta-skeleton <BetaSkeleton>` depth, including spherical and lens depths as special cases [exact];
 - :ref:`convex hull peeling <Qhpeeling>` (also called onion) depth [exact];
 - :ref:`halfspace <Halfspace>` (also called Tukey or location) depth [exact and approximate];
 - :ref:`L2 <L2>` depth [exact];
 - :ref:`Mahalanobis <Mahalanobis>` depth [exact and approximate];
 - :ref:`potential <Potential>` depth [exact];
 - :ref:`projection <Projection>` depth [approximate];
 - :ref:`simplicial <Simplicial>` depth [exact and approximate];
 - :ref:`simplicial volume <SimplicialVolume>` (also called Oja) depth [exact and approximate];
 - :ref:`spatial <Spatial>` depth [exact];
 - :ref:`zonoid <Zonoid>` depth [exact and approximate].

The mentioned below publications are presenting, from a general point of view, the data depth methodology and some of its most important applications. More references follow in the sections dedicated to particular functions of the :math:`\texttt{data-depth}` library.

References
----------

* Mosler, K. and Mozharovskyi, P. (2022). Choosing among notions of multivariate depth statistics. *Statistical Science*, 37(3), 348-368.

* Mosler, K. (2013). Depth statistics. Mosler, K. (2013). Depth statistics. In C. Becker, R. Fried, and S. Kuhnt (Eds.), *Robustness and Complex Data Structures: Festschrift in Honour of Ursula Gather*, Springer (Berlin), 17–34.

* Zuo, Y. and Serfling, R. (2000). General notions of statistical depth function. *The Annals of Statistics*, 28, 461–482.

* Mozharovskyi, P. (2015). *Contributions to Depth-based Classification and Computation of the Tukey Depth*. Verlag Dr. Kovac (Hamburg).

* Liu, R. Y., Parelius, J. M., and Singh, K. (1999). Multivariate analysis by data depth: descriptive statistics, graphics and inference (with discussion and a rejoinder by liu and singh). *The Annals of Statistics* 27(3), 783–858.

* Lange, T., Mosler, K., and Mozharovskyi, P. (2014). Fast nonparametric classification based on data depth. *Statistical Papers*, 55, 49–69.

* Lange, T., Mosler, K., and Mozharovskyi, P. (2014). DDα-classification of asymmetric and fattailed data. In: Spiliopoulou, M., Schmidt Thieme, L., Janning, R. (eds), *Data Analysis, Machine Learning and Knowledge Discovery*, Springer (Berlin), 71–78.

* Mozharovskyi, P. (2022). Anomaly detection using data depth: multivariate case. [arXiv:2210.02851]

* Mozharovskyi, P., Josse, J., and Husson, F. (2020). Nonparametric imputation by data depth. *Journal of the American Statistical Association*, 115(529), 241-253.
