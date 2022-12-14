/******************************************************************************/
/* File:             PD.h                                                     */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains declarations of functions that compute the projection depth of a  */
/* point w.r.t. a data cloud.                                                 */
/*                                                                            */
/******************************************************************************/

#ifndef __ProjectionDepth__
#define __ProjectionDepth__

namespace DataDepth {

    /****************************************************************************/
    /* PD1 computes the projection depth for univariate data.                   */
    /*                                                                          */
    /* Args:                                                                    */
    /*   z - the point for which to calculate the depth,                        */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of     */
    /*       dimension n)                                                       */
    /*   n - number of the data points.                                         */
    /* Returns:                                                                 */
    /*   projection depth of z w.r.t. x.                                        */
    /****************************************************************************/
    double PD1(double z, const double* x, int n);

}

#endif
