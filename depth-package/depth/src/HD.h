/******************************************************************************/
/* File:             HD.h                                                     */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     19.06.2015                                               */
/*                                                                            */
/* Contains declarations of functions that compute the exact halfspace depth  */
/* of a point w.r.t. a data cloud.                                            */
/*                                                                            */
/******************************************************************************/

#ifndef __HalfspaceDepth__
#define __HalfspaceDepth__

#include "Matrix.h"

namespace DataDepth {

  /****************************************************************************/
  /* HD_Comb computes the halfspace depth of a point z in d-space w.r.t.      */
  /*   n data points passed in xx.                                            */
  /*   HD_Comb implements the combinatorial algorithm (k = d-1) as described  */
  /*   in Section 3.1 of "Exact computation of the halfspace depth" by        */
  /*   Rainer Dyckerhoff and Pavlo Mozharovskyi (arXiv:1411:6927)             */
  /*                                                                          */
  /* Args:                                                                    */
  /*   z  - the point to calculate the depth for (vector of dimension d),     */
  /*   xx - the data w.r.t. which the depth has to be computed, (matrix of    */
  /*        dimension n x d)                                                  */
  /*   n  - number of the data points,                                        */
  /*   d  - dimension of the Euclidean space.                                 */
  /* Returns:                                                                 */
  /*   depth of z w.r.t. xx in the interval [0,1].                            */
  /****************************************************************************/
  double HD_Comb(const double* z, const double* const* xx, int n, int d);
  double HD_Comb(const double* z, const dyMatrixClass::cMatrix xx, int n, int d);

  /****************************************************************************/
  /* HD_Comb2 computes the halfspace depth of a point z in d-space w.r.t.     */
  /*   n data points passed in xx.                                            */
  /*   HD_Comb2 implements the combinatorial algorithm (k = d-2) as described */
  /*   in Section 3.2 of "Exact computation of the halfspace depth" by        */
  /*   Rainer Dyckerhoff and Pavlo Mozharovskyi (arXiv:1411:6927)             */
  /*                                                                          */
  /* Args:                                                                    */
  /*   z  - the point to calculate the depth for (vector of dimension d),     */
  /*   xx - the data w.r.t. which the depth has to be computed, (matrix of    */
  /*        dimension n x d)                                                  */
  /*   n  - number of the data points,                                        */
  /*   d  - dimension of the Euclidean space.                                 */
  /* Returns:                                                                 */
  /*   depth of z w.r.t. xx in the interval [0,1].                            */
  /****************************************************************************/
  double HD_Comb2(const double* z, const double* const* xx, int n, int d);
  double HD_Comb2(const double* z, const dyMatrixClass::cMatrix xx, int n, int d);

  /****************************************************************************/
  /* HD_Rec computes the halfspace depth of a point z in d-space w.r.t.       */
  /*   n data points passed in xx.                                            */
  /*   HD_Rec implements the recursive algorithm (k = 1) as described in      */
  /*   Section 3.3 of "Exact computation of the halfspace depth" by           */
  /*   Rainer Dyckerhoff and Pavlo Mozharovskyi (arXiv:1411:6927)             */
  /*                                                                          */
  /* Args:                                                                    */
  /*   z  - the point to calculate the depth for (vector of dimension d),     */
  /*   xx - the data w.r.t. which the depth has to be computed, (matrix of    */
  /*        dimension n x d)                                                  */
  /*   n  - number of the data points,                                        */
  /*   d  - dimension of the Euclidean space.                                 */
  /* Returns:                                                                 */
  /*   depth of z w.r.t. xx in the interval [0,1].                            */
  /****************************************************************************/
  double HD_Rec(const double* z, const double* const* xx, int n, int d);
  double HD_Rec(const double* z, const dyMatrixClass::cMatrix xx, int n, int d);

  /****************************************************************************/
  /* HD1 computes the halfspace depth for univariate data.                    */
  /*                                                                          */
  /* Args:                                                                    */
  /*   z  - the point for which to calculate the depth,                       */
  /*   xx - the data w.r.t. which the depth has to be computed, (vector of    */
  /*        dimension n)                                                      */
  /*   n - number of the data points.                                         */
  /* Returns:                                                                 */
  /*   halfspace depth of z w.r.t. x.                                         */
  /****************************************************************************/
  double HD1(double z, const double* xx, int n);

}

#endif
