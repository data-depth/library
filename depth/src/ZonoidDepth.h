#ifndef __ZonoidDepth__
#define __ZonoidDepth__
/*
   File:         depth.h
   Created by:   Rainer Dyckerhoff
   Last revised: 15.05.2013

   Computation of the zonoid data depth. 
   
   For a description of the algorithm, see:
   Dyckerhoff, R., Koshevoy, G., and Mosler, K. (1996)
   Zonoid Data Depth: Theory and Computation,
   in: Compstat - Proceedings in Computational Statistics, (Albert Prat, ed.), 
   Physica-Verlag, Heidelberg, p. 235--240.
*/   

double ZonoidDepth(vector<TPoint>& x, TPoint& z, int& Error);
/* 
   Calculate the zonoid data depth of the point 'z' with respect to the
   data points 'x'. The number of data points is passed in 'NoPoints',
   the dimension in 'Dimension'. If an error occurs, the error code is
   stored in '*Error'. Possible error codes are:
     0: no error,
     1: simplex algorithm did not terminate within 'MaxIt' iterations,
     2: not enough memory available,
   If no error occured, the return value is the zonoid data depth of 'z'.
   If the error code is 1, the return value is an lower bound to the
   zonoid data depth of 'z'. If the error code is 2, the return value is -1. 
*/

int IsInConvex(vector<TPoint>& x, TPoint& z, int& Error);
int InConvexes(TMatrix& points, TVariables& cardinalities, TMatrix& objects, int& Error, TIntMatrix *areInConvexes);

int GetMeansSds(vector<TPoint>& x, TPoint *means, TPoint *sds);
int Standardize(vector<TPoint> &x, TPoint& means, TPoint& sds);
int Standardize(TPoint &x, TPoint& means, TPoint& sds);

int GetMeansSds(TDMatrix &x, int n, int d, TPoint *means, TPoint *sds);
int Standardize(TDMatrix &x, int n, int d, TPoint& means, TPoint& sds);

#endif

