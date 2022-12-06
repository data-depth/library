
#include "stdafx.h"

void covMcd(double* m, int i , int j , int* MCD){


}

void Cholesky(TDMatrix a, int n, int& rank, int& error) {
	const double eps = 1e-12;
	error = 1;
	rank = 0;
	for (int j = 0; j < n; j++) {
		double eps1 = eps * fabs(a[j][j]);
		for (int k = 0; k < j; k++) a[j][j] -= a[j][k] * a[j][k];
		if (fabs(a[j][j]) <= eps1) a[j][j] = 0;
		else if (a[j][j] > 0) {
			a[j][j] = sqrt(a[j][j]);
			rank++;
		}
		else return;
		for (int i = j + 1; i < n; i++) {
			double eps2 = eps * fabs(a[i][i]);
			for (int k = 0; k < j; k++) a[i][j] -= a[i][k] * a[j][k];
			if (a[j][j] > 0)
				a[i][j] /= a[j][j];
			else
				if (fabs(a[i][j]) <= eps1 * eps2)
					a[j][j] = 0;
				else
					return;
		}
	}
	error = 0;
}

/****************************************************************************/
/* 'InversePosDef' computes the (pseudo)-inverse of a symmetric positive    */
/*   semi-definite matrix 'a'.                                              */
/*                                                                          */
/* Args:                                                                    */
/*   a  - the matrix of dimension n x n which has to be inverted,           */
/*        only the lower triangular part of 'a' is used in the procedure.   */
/*        Since the inverse is again symmetric, on exit, the lower          */
/*        triangular part of the inverse is stored in the lower triangular  */
/*        part of the matrix 'a'.                                           */
/*   n  - dimension of the matrix 'a',                                      */
/*   rank  - on exit contains the rank of the matrix 'a' (which is the      */
/*           same as the rank of the inverse,                               */
/*   error - error indicator with the following error codes:                */
/*             0: no error,                                                 */
/*             1: error, matrix 'a' is not positive semi-definite.          */
/*                                                                          */
/****************************************************************************/

void InversePosDef(TDMatrix a, int n, int& rank, int& error) {
	const double eps = 1e-12;
	double sum = 0;
	Cholesky(a, n, rank, error);

	for (int j = 0; j < n; j++) {
		if (a[j][j] > eps) {
			a[j][j] = 1 / a[j][j];
			for (int i = j + 1; i < n; i++) {
				if (a[i][i] > eps) {
					sum = 0;
					for (int k = j; k < i; k++) sum -= a[i][k] * a[k][j];
					a[i][j] = sum / a[i][i];
				}
				else a[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j <= i; j++) {
			sum = 0;
			for (int k = i; k < n; k++) sum += a[k][i] * a[k][j];
			a[i][j] = sum;
		}
	}
}



void MahalanobisDepth(TDMatrix X, TDMatrix x, int d, int n, int nx, double* mat_MCD, double *depths){
	double* ms = means(X, n, d);
	TDMatrix s = asMatrix(mat_MCD,d,d);
	
	
	int rank, error;
	InversePosDef(s,d,rank,error);

    for(int ii=0;ii<d;ii++){
		for(int jj=0;jj<ii;jj++){
			// s[ii][jj]=s[jj][ii];
            s[jj][ii]=s[ii][jj];
		}
	}

	double *a = new double[d];
	for (int i = 0; i < nx; i++){
		depths[i] = 0;
		for (int k = 0; k < d; k++){
			a[k] = 0;
			for (int j = 0; j < d; j++){
				a[k] += (x[i][j] - ms[j])*s[j][k];
			}
		}
		for (int j = 0; j < d; j++){
			depths[i] += (x[i][j] - ms[j])*a[j];
		}
		depths[i] = 1.0 / ((depths[i]) + 1);
	}
	delete[] a;
	delete[] ms;
}
