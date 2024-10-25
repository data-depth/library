/******************************************************************************/
/* File:             WMD.h                                                    */
/* Created by:       Pavlo Mozharovskyi                                       */
/* Last revised:     23.06.2024                                               */
/*                                                                            */
/* Contains declarations of functions that compute the weighted-mean depths   */
/* of a point w.r.t. a data cloud.                                            */
/*                                                                            */
/******************************************************************************/

#ifndef __WieghtedMeanDepth__
#define __WieghtedMeanDepth__

namespace DataDepth {

    /****************************************************************************/
    /* CechStarD1 computes the continuous expected convex hull depth for      */
    /* univariate data.                                                       */
    /*                                                                        */
    /* Args:                                                                  */
    /*   z - the point for which to calculate the depth,                      */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of   */
    /*       dimension n)                                                     */
    /*   n - number of the data points.                                       */
    /* Returns:                                                               */
    /*   continuous expected convex hull (star) depth of z w.r.t. x.          */
    /****************************************************************************/
    double CechStarD1(double z, const double* x, int n);

    /**************************************************************************/
    /* CechD1 computes the continuous expected convex hull depth for          */
    /* univariate data.                                                       */
    /*                                                                        */
    /* Args:                                                                  */
    /*   z - the point for which to calculate the depth,                      */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of   */
    /*       dimension n)                                                     */
    /*   n - number of the data points.                                       */
    /* Returns:                                                               */
    /*   continuous expected convex hull depth of z w.r.t. x.                 */
    /**************************************************************************/
    double CechD1(double z, const double* x, int n);

    /**************************************************************************/
    /* GeomD1 computes the geometric depth for univariate data.               */
    /*                                                                        */
    /* Args:                                                                  */
    /*   z - the point for which to calculate the depth,                      */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of   */
    /*       dimension n)                                                     */
    /*   n - number of the data points.                                       */
    /* Returns:                                                               */
    /*   geometric depth of z w.r.t. x.                                       */
    /**************************************************************************/
    double GeomD1(double z, const double* x, int n);
}

#endif
