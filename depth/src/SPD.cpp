/******************************************************************************/
/* File:             SPD.cpp                                                  */
/* Created by:       Leonardo Leone                                           */
/* Last revised:     27.08.2026                                               */
/*                                                                            */
/* Contains functions that compute the skew-adjusted projection depth of a    */
/* point w.r.t a data cloud.                                                  */
/*                                                                            */
/******************************************************************************/

#include <algorithm>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include "SPD.h"

using namespace std;

namespace DataDepth {

	/****************************************************************************/
	/* SPD1 computes the skew-adjusted projection depth for univariate data.    */
	/*                                                                          */
	/* Args:                                                                    */
	/*   z - the point for which to calculate the depth,                        */
	/*   x - the data w.r.t. which the depth has to be computed, (vector of     */
	/*       dimension n)                                                       */
	/*   n - number of the data points.                                         */
	/* Returns:                                                                 */
	/*   skew-adjusted projection depth of z w.r.t. x.                          */
	/****************************************************************************/

	
	double MedCouple(double* xCopy, int n, double med){
		double MC{0.0};
		int n2 = ceil(n * 0.5);
		int n22=(n2+1)*(n2+1);
		double* hMC = new double[n22];
		int n3 = ceil((n22)*0.5);
		int ij{0};
		int k{0};
		for(int i{0}; i <= n2; i++){
			for(int j{n-1}; j >= n2; j--){
				if (xCopy[i]!=xCopy[j]){
					hMC[ij]=(xCopy[j] + xCopy[i] - 2*med)/(xCopy[j] - xCopy[i]);
				}
				else if (xCopy[i]==xCopy[j]){
					if (k==0){k=j-i+1;}
					if (i+j-1<k){hMC[ij]=-1.;}
					if (i+j-1==k){hMC[ij]=0.;}
					if (i+j-1>k){hMC[ij]=1.;}
				}
				ij++;
			}
		}
		nth_element(hMC,hMC+n3,hMC+n22);
		return hMC[n3];


	}

	double SPD1(double z, const double* x, int n) {
		double* xCopy = new double[n]; // Copy of "x" as "median(...)" spoils it
		memcpy(xCopy, x, n * sizeof(double));
        sort(xCopy, xCopy + n); // sort for medcouple
        
		// The median is computed as the ceil(n * 0.5)-th smallest element in the
		// array x
		// double dev,w;
		double dev,w;
		int n1 = ceil(n * 0.25);
		int n2 = ceil(n * 0.5);
		int n3 = ceil(n * 0.75);
		double med{xCopy[n2]}; // values are all sorted for medcouple
		double MC = MedCouple(xCopy, n, med);
		double IQR=xCopy[n3]-xCopy[n1];
		if (z >= med) {
			w=xCopy[n3] + 1.5 * exp(-4*MC)*IQR;
			dev = w-med;
		}
		else {
			w=xCopy[n1] - 1.5 * exp(+3*MC)*IQR;
			dev = med-w;
		}
		delete[] xCopy;
		return 1.0 / (1.0 + fabs(z - med) / dev);
	}
	
}

