/*
File:             LensDepth.cpp
Created by:       Pavlo Mozharovskyi
First published:  08.03.2018
Last revised:     08.03.2018

A procedure for computing the beta-skeleton depth, a generalization of the lens depth.

For a description of the algorithm, see:
Liu, Z. and Modarres, R. (2011). Lens data depth and median. Journal of Nonparametric Statistics, 23(4), 1063-1074.
Yang, M. and Modarres, R. (2017). Beta-skeleton depth functions and medians. Commmunications in Statistics - Theory and Methods, to appear.
*/


#include "stdafx.h"

#define DISTTYPE_L1           1
#define DISTTYPE_L2           2
#define DISTTYPE_MAX          3
#define DISTTYPE_Lp           4
#define DISTTYPE_MAHALANOBIS  5

void LensDepth(TDMatrix X, TDMatrix x, int d, int n, int nx, double beta,
	int distType, double p, TDMatrix sigma, double* depths){
  // The centers of the two balls
  double* ci = new double[d];
  double* cj = new double[d];
  double b = beta / 2;
  double dist = 0;
  double disti = 0;
  double distj = 0;
  int counts = 0;
  for (int obs = 0; obs < nx; obs++){ // loop through observations
    counts = 0;
    // Loop(s) through pairs of points
    for (int i = 0; i < n - 1; i++){
      for (int j = i + 1; j < n; j++){
        dist = 0;
        disti = 0;
        distj = 0;
        // Calculate centres
        for (int k = 0; k < d; k++){
          ci[k] = X[i][k] * b + X[j][k] * (1 - b);
          cj[k] = X[i][k] * (1 - b) + X[j][k] * b;
        }
        // Calculate distances
        switch (distType){
        case DISTTYPE_L1:
          for (int k = 0; k < d; k++){
            dist  += fabs(X[i][k] - X[j][k]);
            disti += fabs(x[obs][k] - ci[k]);
            distj += fabs(x[obs][k] - cj[k]);
          }
          break;
        case DISTTYPE_L2:
          for (int k = 0; k < d; k++){
            dist  += pow(X[i][k] - X[j][k], 2);
            disti += pow(x[obs][k] - ci[k], 2);
            distj += pow(x[obs][k] - cj[k], 2);
          }
          dist  = sqrt(dist);
          disti = sqrt(disti);
          distj = sqrt(distj);
          break;
        case DISTTYPE_MAX:
          for (int k = 0; k < d; k++){
            dist  = max(dist, fabs(X[i][k] - X[j][k]));
            disti = max(disti, fabs(x[obs][k] - ci[k]));
            distj = max(distj, fabs(x[obs][k] - cj[k]));
          }
          break;
        case DISTTYPE_Lp:
          for (int k = 0; k < d; k++){
            dist  += pow(fabs(X[i][k] - X[j][k]), p);
            disti += pow(fabs(x[obs][k] - ci[k]), p);
            distj += pow(fabs(x[obs][k] - cj[k]), p);
          }
          dist  = pow(dist, 1/p);
          disti = pow(disti, 1/p);
          distj = pow(distj, 1/p);
          break;
        case DISTTYPE_MAHALANOBIS:
          for (int k = 0; k < d; k++){
            for (int l = 0; l < d; l++){
              dist  += (X[i][l] - X[j][l]) * sigma[l][k] * (X[i][k] - X[j][k]);
              disti += (x[obs][l] - ci[l]) * sigma[l][k] * (x[obs][k] - ci[k]);
              distj += (x[obs][l] - cj[l]) * sigma[l][k] * (x[obs][k] - cj[k]);
            }
          }
          dist  = sqrt(dist);
          disti = sqrt(disti);
          distj = sqrt(distj);
          break;
        }
        // Deside whether the point is in the lens (influene region)
        dist *= b;
        if (disti < dist && distj < dist){
          counts++;
        }
      }
    }
    depths[obs] = counts / (double)(n * (n - 1) / 2); // return
  }
  // Release memory
  delete[] ci;
  delete[] cj;
}
