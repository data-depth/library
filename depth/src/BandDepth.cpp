/******************************************************************************/
/* File:             BandDepth.cpp                                            */
/* Created by:       Leonardo Leone                                           */
/* Last revised:     10.09.2026                                               */
/*                                                                            */
/* Contains declarations of functions that compute the Band depth             */
/*                                                                            */
/******************************************************************************/



#include "stdafx.h"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>

const double eps_band = 1e-10;

void ComputeSimplicialBandDepth(T3DMatrix x, T3DMatrix X, int m, int n, int t, int d, bool modif, 
               int J, double* depths){
  // Prepare data structures for the loop through all J combinations
  double* b = new double[d + 1]; b[d] = 1;
  double* z = new double[d + 1];
  int* counters = new int[d + 1];
  TDMatrix A = newM(d + 1, d + 1);
  unsigned long long div0 = choose(n, d + 1); // num simplices
  // Loop for all observations to compute depth for
  for (int iObs = 0; iObs < m; iObs++){
    unsigned long long theCounter = 0;
    //unsigned long long numSimplicesChecked = 0;
    // Loop to check all combinations of J functions out of n
    for (int i = 0; i < d; i++){ counters[i] = i; } counters[d] = d - 1;
    while (counters[0] != n - (d + 1)){
      int i = d;
      while (i > 0 && counters[i] == n - (d + 1) + i){ i--; }
      counters[i]++; int j = i + 1;
      while (j < d + 1){ counters[j] = counters[j - 1] + 1; j++; }
      // Execute logic for a single (d+1)-tuple of functoins:
      bool isInBand = true;
      // Loop for all time points
      for (int iTime = 0; iTime < t; iTime++){
        // Check whether current function is inside simplex for this time point
        for (int j = 0; j < d; j++){
          for (int k = 0; k < d + 1; k++){
            A[j][k] = X[counters[k]][iTime][j];
          }
        }
        for (int k = 0; k < d + 1; k++){
          A[d][k] = 1;
        }
        memcpy(b, x[iObs][iTime], d * sizeof(double)); b[d] = 1;
        if (solveUnique(A, b, z, d + 1)){
          bool isInside = true;
          for (int j = 0; j < d + 1; j++){
            if (z[j] < -eps_band){ isInside = false; break; }
          }
          if (isInside){ // if inside simplex
            if (modif){
              theCounter++;
            }
          }else{ // if outside simplex
            if (!modif){
              isInBand = false;
              break;
            }
          }
        }
      }
      if (!modif){ // if not modified version
        theCounter += isInBand; // add 1 once only (not each time point)
      }
    }
    if (modif){ // if modified version
      depths[iObs] = (double)theCounter / (div0 * t);
    }else{ // if not modified version
      depths[iObs] = (double)theCounter / div0;
    }
  }
  // Release memory
  delete[] b;
  delete[] z;
  delete[] counters;
  deleteM(A);
}

void ComputeModBandDepth(T3DMatrix x, T3DMatrix X, int m, int n, int t, int d, 
                double* depths){
    // double* b = new double[d + 1]; b[d] = 1;
    unsigned long long* NumAbove = new double[m];
    unsigned long long* NumBellow = new double[m];
    unsigned long long* NumEqual = new double[m];
    // double* X1Dview = new double[n];
    unsigned long long totalPairs = d*t*n*(n-1)/2;
    // double* z = new double[d + 1];
    // int* counters = new int[d + 1];
    // TDMatrix A = newM(d + 1, d + 1);
    // unsigned long long div0 = choose(n, d + 1); // num simplices
    
    for (int iDim = 0; iDim < d; iDim++){
      for (int iTime = 0; iTime < t; iTime++){
        for (int iObs = 0; iObs < m; iObs++){
          NumAbove[iObs]=0;
          NumBellow[iObs]=0;
          for (int iX1Dview = 0; iX1Dview < n; iX1Dview++){
            if(x[iObs][iTime][iDim]<X[iX1Dview][iTime][iDim]){
              NumAbove[iObs]++;
            }
            else {if(x[iObs][iTime][iDim]>X[iX1Dview][iTime][iDim]){
              NumBellow[iObs]++;
            }
            else{
              NumEqual[iObs]++;
            }}
          }
          depths[iObs]+=((NumBellow[iObs]*NumAbove[iObs])+
                         (NumEqual[iObs]*(n-NumEqual[iObs]))+
                         (NumEqual[iObs]*(NumEqual[iObs]-1)/2))/totalPairs;
        }
      }
    }
    
  
  
  // Release memory
  delete[] NumAbove;
  delete[] NumBellow;
  delete[] NumEqual;
}

void ComputeBandDepth(T3DMatrix x, T3DMatrix X, int m, int n, int t, int d, 
                double* depths){
    // double* b = new double[d + 1]; b[d] = 1;
    unsigned long long totalPairs = d*t*n*(n-1)/2;
    bool isInside = true;
    // double* z = new double[d + 1];
    // TDMatrix A = newM(d + 1, d + 1);
    // unsigned long long div0 = choose(n, d + 1); // num simplices
    
    for (int iDim = 0; iDim < d; iDim++){
      for (int iObs = 0; iObs < m; iObs++){
        unsigned long long theCounter = 0;
        for (int iFirst = 0; iFirst < n-1; iFirst++){
          for (int iSecond = iFirst+1; iSecond < n; iSecond++){
          isInside = true;
          int iTime = 0;
          while(isInside && iTime < t)
            {
              if(x[iObs][iTime][iDim]>X[iFirst][iTime][iDim] && 
                 x[iObs][iTime][iDim]>X[iSecond][iTime][iDim]){isInside=false;}

              if(x[iObs][iTime][iDim]<X[iFirst][iTime][iDim] && 
                 x[iObs][iTime][iDim]<X[iSecond][iTime][iDim]){isInside=false;}
              iTime++;}
          if (isInside){theCounter += 1}  
          }
        }
        depths[iObs] += (double)theCounter / totalPairs;
      }
    }
    
  
  
  // Release memory
  delete[] b;

}