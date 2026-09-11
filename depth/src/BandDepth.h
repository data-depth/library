/******************************************************************************/
/* File:             BandDepth.h                                              */
/* Created by:       Leonardo Leone                                           */
/* Last revised:     10.09.2026                                               */
/*                                                                            */
/* Contains declarations of functions that compute the Band depth             */
/*                                                                            */
/******************************************************************************/



#ifndef __BandDepth__
#define __BandDepth__


void ComputeSimplicialBandDepth(T3DMatrix x, T3DMatrix X, int m, int n, int t, int d, bool modif, 
               int J, double* depths);