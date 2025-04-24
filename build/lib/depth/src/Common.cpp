/*
  File:             Common.cpp
  Created by:       Pavlo Mozharovskyi
  First published:  17.05.2013
  Last revised:     13.11.2015
  
  Commonly used functions.
*/

#include "stdafx.h"
// 3D-array structures
T3DMatrix as3DMatrix(double* arr, int n, int t, int d){
  T3DMatrix mat = new double**[n];
  for (int i = 0; i < n; i++){
    mat[i] = new double*[t];
    for (int j = 0; j < t; j++)
    {
      mat[i][j] = arr + i*t*d + j*d;
    }
  }
  return mat;
}

// by rows
TDMatrix asMatrix(double* arr, int n, int d){
	TDMatrix mat = new double*[n];
	for (int i = 0; i < n; i++)
		mat[i] = arr + i*d;
	return mat;
}

double** newM(int n, int d){
	double* a = new double[n*d];
	return asMatrix(a, n, d);
}

void deleteM(TDMatrix X){
	delete[] X[0];
	delete[] X;
}

TDMatrix copyM(TDMatrix X, int n, int d){
	double* a = new double[n*d];
	memcpy(a, X[0], n*d*sizeof(double));
	return asMatrix(a, n, d);
}

void printMatrix(TDMatrix mat, int n, int d){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < d; j++)
			std::cout << mat[i][j] << "\t";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


unsigned long long choose(unsigned long long n, unsigned long long k){
	unsigned long long r = n--; unsigned long long d = 2;
	while (d <= k){ r *= n--; r /= d++; }
	return r;
}

unsigned long long fact(unsigned long long n){
	unsigned long long r = 1; unsigned long long i = 2;
	while (i <= n){ r *= i++; }
	return r;
}

/* -------------------------------------------------------------------------- */
/* By Rainer Dyckerhoff, modified by Pavlo Mozharovskyi                       */
/* Solves a uniquely solvable system of linear equations                      */
/* -------------------------------------------------------------------------- */
bool solveUnique(TDMatrix A, double* b, double* x, int d){
	int imax, jmax;
	int* colp = new int[d];
	double amax;
	for (int k = 0; k < d - 1; k++) {
		imax = k;
	  jmax = k;
		amax = fabs(A[k][k]);
		colp[k] = k;
		// Spaltenmaximum finden
		for (int i = k + 1; i < d; i++) {
			if (fabs(A[i][k]) > amax) {
				amax = fabs(A[i][k]);
				imax = i;
			}
		}
		// Spaltenmaximum gleich null => complete pivoting
		if (amax < eps_pivot) {
			for (int j = k + 1; j < d; j++) {
				for (int i = k; i < d; i++) {
					if (fabs(A[i][j]) > amax) {
						amax = fabs(A[i][j]);
						imax = i;
						jmax = j;
					}
				}
			}
			if (amax < eps_pivot) {
				delete[] colp;
				return false;
			}
			// Spaltentausch
			for (int i = 0; i < d; i++) {
				double tmp = A[i][k];
				A[i][k] = A[i][jmax];
				A[i][jmax] = tmp;
			}
			colp[k] = jmax;
		}
		// Zeilentausch
		if (imax != k) {
			for (int j = k; j < d; j++) {
				double tmp = A[k][j];
				A[k][j] = A[imax][j];
				A[imax][j] = tmp;
			}
			double tmp = b[k];
			b[k] = b[imax];
			b[imax] = tmp;
		}
		// Elimination
		for (int i = k + 1; i < d; i++) {
			double factor = A[i][k] / A[k][k];
			for (int j = k + 1; j < d; j++){
				A[i][j] -= factor * A[k][j];
			}
			b[i] -= factor * b[k];
		}
	}
	// R?cksubstituition
	colp[d - 1] = d - 1;
	for (int k = d - 1; k >= 0; k--) {
		x[k] = b[k] / A[k][k];
		for (int i = k - 1; i >= 0; i--) b[i] -= x[k] * A[i][k];
	}
	// Spaltenvertauschungen r?ckg?ngig machen
	for (int k = d - 1; k >= 0; k--) {
		if (colp[k] != k) {
			double temp = x[k];
			x[k] = x[colp[k]];
			x[colp[k]] = temp;
		}
	}
	delete[] colp;
	return true;
}

double getDet(TDMatrix A, int d){
    int imax, jmax;
    int* colp = new int[d];
    double amax;
    double det = 1; // The final A's determinant value
    for (int k = 0; k < d - 1; k++) {
        imax = k;
        jmax = k;
        amax = fabs(A[k][k]);
        colp[k] = k;
        // Find column maximum
        for (int i = k + 1; i < d; i++) {
            if (fabs(A[i][k]) > amax) {
                amax = fabs(A[i][k]);
                imax = i;
            }
        }
        // If column maximum = 0, then complete pivoting
        if (amax < eps_pivot) {
            for (int j = k + 1; j < d; j++) {
                for (int i = k; i < d; i++) {
                    if (fabs(A[i][j]) > amax) {
                        amax = fabs(A[i][j]);
                        imax = i;
                        jmax = j;
                    }
                }
            }
            if (amax < eps_pivot) {
                delete[] colp;
                return 0;
            }
            // Column swap
            for (int i = 0; i < d; i++) {
                double tmp = A[i][k];
                A[i][k] = A[i][jmax];
                A[i][jmax] = tmp;
            }
            colp[k] = jmax;
        }
        // Row swap
        if (imax != k) {
            for (int j = k; j < d; j++) {
                double tmp = A[k][j];
                A[k][j] = A[imax][j];
                A[imax][j] = tmp;
            }
            det *= -1;
        }
        // Elimination
        for (int i = k + 1; i < d; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k + 1; j < d; j++){
                A[i][j] -= factor * A[k][j];
            }
        }
    }
    delete[] colp;
    // Calculate the determinant value
    for (int i = 0; i < d; i++){
        det *= A[i][i];
    }
    return det;
}

void lmatrice(TDMatrix A ,TDMatrix B, int n, int l){
	int li=0;
	int k=n-1;
	for(int i=0;i<n;i++){
		if(i!=l){
			int lj=0;
			for(int j=1;j<n;j++){
   				B[li][lj]=A[i][j];
    				lj++;	
			}		
		li++;
		}
	}
}

double determinant(TDMatrix A , int n){
	if(n==1){
  	return A[0][0];
  	}
	double resultat=0.;
	int k=n-1;
 	double signe=1.;
 	TDMatrix B =newM(k,k);
 	for(int i=0;i<n;i++){
		lmatrice(A,B,n,i);
  		resultat=resultat+signe*A[0][i]*determinant(B,k);
  		signe=-signe;
	}
	
	delete[] B;
	return resultat;
}


double* means(TDMatrix X, int n, int d) {
	double* ms = new double[d];
	for (int i = 0; i < d; i++) {
		ms[i] = 0.0;
		for (int j = 0; j < n; j++)
			ms[i] += X[j][i];
		ms[i] /= n;
	}
	return ms;
}

TDMatrix cov(TDMatrix X, int n, int d) {
	double* means = new double[d];
	double* dev = new double[d];
	// zeroing TDMatrix
	TDMatrix covX = newM(d, d);
	for (int k = 0; k < d; k++)
		for (int j = 0; j < d; j++)
			covX[k][j] = 0;
	// means
	for (int i = 0; i < d; i++) {
		means[i] = 0.0;
		for (int j = 0; j < n; j++)
			means[i] += X[j][i];
		means[i] /= n;
	}
	for (int i = 0; i < n; i++) {
		// deviations
		for (int k = 0; k < d; k++) {
			dev[k] = X[i][k] - means[k];
		}
		// add to cov
		for (int k = 0; k < d; k++) {
			for (int j = 0; j < d; j++) {
				covX[k][j] += dev[k] * dev[j];
			}
		}
	}
	//scale
	for (int i = 0; i < d; i++) {
		for (int j = 0; j < d; j++) {
			covX[i][j] /= n - 1;
		}
	}
	delete[] means;
	delete[] dev;
	return covX;
}

void GetProjections(TDMatrix points, int n, int d, TDMatrix directions, int k, TDMatrix projections){
	for (int i = 0; i < k; i++){
		double* projection = projections[i];
		for (int j = 0; j < n; j++){
			double sum = 0;
			for (int l = 0; l < d; l++){
				sum += points[j][l]*directions[i][l];
			}
			projection[j] = sum;
		}
	}
}


void outVector(TVariables& point){

}



bool OUT_ALPHA = false;

void outString(char const * str){
#ifdef DEF_OUT_ALPHA
	if (!OUT_ALPHA) return;
	std::cout << str << std::endl;
#endif
}

//template<typename T>
void outVector(TPoint& point){
#ifdef DEF_OUT_ALPHA
	if (!OUT_ALPHA) return;
	for (int j = 0; j < point.size(); j++){
		std::cout << point[j] << ", ";
	}
	std::cout << std::endl;
#endif
}

void outMatrix(TMatrix& points){
#ifdef DEF_OUT_ALPHA
	if (!OUT_ALPHA) return;
	for (int i = 0; i < points.size(); i++){
		//std::cout << i << ": ";
		for (int j = 0; j < points[i].size(); j++){
			std::cout << points[i][j] << ", ";
		}
		std::cout << std::endl;
	}
#endif
}

void outFeatures(Features fs){
#ifdef DEF_OUT_ALPHA
	if (!OUT_ALPHA) return;
	std::cout << "order\t number\t angle\t error" << std::endl;
	for (int i = 0; i < fs.size(); i++){
		std::cout << fs[i].order << ",\t " << fs[i].number << ",\t " << fs[i].angle << ",\t " << fs[i].error << std::endl;
	}
#endif
}
