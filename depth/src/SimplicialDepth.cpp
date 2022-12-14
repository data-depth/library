
#include "stdafx.h"

/* -------------------------------------------------------------------------- */
/* Calculates multivariate Liu (=simplicial) depth going through all          */
/* possible simplices                                                         */
/* -------------------------------------------------------------------------- */
void SimplicialDepthsEx(TDMatrix X, TDMatrix x, int d, int n, int nx,
	 double *depths){

	double* b = new double[d + 1]; b[d] = 1;
	double* z = new double[d + 1];
	int* counters = new int[d + 1];
	TDMatrix A = newM(d + 1, d + 1);
	unsigned long long div0 = choose(n, d + 1);

	for (int obs = 0; obs < nx; obs++){
		unsigned long long theCounter = 0;
		unsigned long long numSimplicesChecked = 0;

		for (int i = 0; i < d; i++){ counters[i] = i; }counters[d] = d - 1;
		while (counters[0] != n - (d + 1)){
			int i = d;
			while (i > 0 && counters[i] == n - (d + 1) + i){ i--; }
			counters[i]++; int j = i + 1;
			while (j < d + 1){ counters[j] = counters[j - 1] + 1; j++; }
			
			for (int j = 0; j < d; j++){
				for (int k = 0; k < d + 1; k++){
					A[j][k] = X[counters[k]][j];
				}
			}
			for (int k = 0; k < d + 1; k++){
				A[d][k] = 1;
			}
			memcpy(b, x[obs], d*sizeof(double)); b[d] = 1;
			if (solveUnique(A, b, z, d + 1)){
				bool isInside = true;
				for (int j = 0; j < d + 1; j++){
					if (z[j] < 0){ isInside = false; break; }
				}
				if (isInside){ theCounter++; }
			}
			(numSimplicesChecked) ++;
		}		
		bool sc = numSimplicesChecked == div0;
		double depth = (double)theCounter / div0;
		depths[obs] = depth;
	}

	delete[] b;
	delete[] z;
	delete[] counters;
	deleteM(A);
}


/* -------------------------------------------------------------------------- */
/* Estimates multivariate Liu (=simplicial) depth based on 'k' randomly       */
/* drawn simplices                                                            */
/* -------------------------------------------------------------------------- */
void SimplicialDepthsApx(TDMatrix X, TDMatrix x, int d, int n, int nx,
	unsigned long long k, double *depths){

	double* b = new double[d + 1]; b[d] = 1;
	double* z = new double[d + 1];
	int* counters = new int[d + 1];
	double* a = new double[(d + 1)*(d + 1)];
	TDMatrix A = asMatrix(a, d + 1, d + 1);

	for (int obs = 0; obs < nx; obs++){
	unsigned long long theCounter = 0;

	for (unsigned long long i = 0; i < k; i++){
		// Generate a combination of indices
		for (int j = 0; j < d + 1; j++){
			bool _new = false;
			do{
				_new = true;
				counters[j] = random(n);
				for (int l = 0; l < j; l++){
					if (counters[l] == counters[j]){
						_new = false;
						break;
					}
				}
			} while (!_new);
		}
		// Construct the simplex out of it
		for (int j = 0; j < d; j++){
			for (int l = 0; l < d + 1; l++){
				A[j][l] = X[counters[l]][j];
			}
		}
		for (int l = 0; l < d + 1; l++){
			A[d][l] = 1;
		}
		memcpy(b, x[obs], d*sizeof(double)); b[d] = 1;
		// Check whether 'x' lies inside of this simplex
		solveUnique(A, b, z, d + 1);
		bool isInside = true;
		for (int j = 0; j < d + 1; j++){
			if (z[j] < 0){ isInside = false; break; }
		}
		if (isInside){ theCounter++; }
	}
	double depth = (double)theCounter / k;
	depths[obs] = depth;
}

delete[] b;
delete[] z;
delete[] counters;
delete[] A;
delete[] a;
}

//#include <stdexcept>


/*******************************************************************************************************************************************************
*
*     intSD2
*
******************************************************************************************************************************************************/

unsigned long long intSD2(double** x, int n) {
	const double eps = 1e-10;
	double* alpha = new double[n];
	int nt = 0; // Wie oft ist 0 in Points enthalten ? 
	int nh = 0; // Wie viele Punkte im Halbraum ?
	//  Winkel alpha berechnen und array initialisieren 
	for (int i = 0; i < n; i++) {
		if (hypot(x[i][0], x[i][1]) <= eps)
			nt++;
		else {
			alpha[i - nt] = atan2(x[i][1], x[i][0]);  // alpha in (-pi,pi]
			if (alpha[i - nt] < -M_PI + eps) alpha[i - nt] = M_PI; //  Korrektur f?r Zahlen wie (-1, -1e-16)
			if (alpha[i - nt] <= eps) nh++;
		}
	}
	unsigned long long nn = n - nt;
	// Winkel sortieren
	sort(alpha, alpha + nn);
	// Simpliziale Tiefe berechnen
	unsigned long long result = nn * (nn - 1) * (nn - 2) / 6;
	unsigned long long j = nh;
	for (int i = 0; i < nh; i++) {
		while ((j <= nn - 1) && (alpha[j] - M_PI <= alpha[i] - eps)) j++;
		result -= (j - i - 1) * (j - i - 2) / 2;
	}
	j = 0;
	for (int i = nh; i < nn; i++) {
		while ((j <= nh - 1) && (alpha[j] + M_PI <= alpha[i] - eps)) j++;
		result -= (nn + j - i - 1) * (nn + j - i - 2) / 2;
	}
	delete[] alpha;
	result += choose(nt, 1)*choose(nn, 2) + choose(nt, 2)*choose(nn, 1) + choose(nt, 3);
	return result;
}

void SimplicialDepths2(TDMatrix X, TDMatrix x, int n, int nx, double *depths) {
	if (n <= 0) throw invalid_argument("n <= 0");
	double c = (double)(n * (n - 1) * (n - 2) / 6);  // Anzahl aller Simplizes
	double** xz = newM(n, 2);
	for (int zi = 0; zi < nx; zi++){
		// z in Ursprung verschieben
		for (int i = 0; i < n; i++)
			for (int j = 0; j < 2; j++) 
				xz[i][j] = X[i][j] - x[zi][j];
		unsigned long long result = intSD2(xz, n);
		depths[zi] = result / c;
	}
	deleteM(xz);
}
