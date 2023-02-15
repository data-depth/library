#include "stdafx.h"

void OjaDepthsEx(TDMatrix X, TDMatrix x, int d, int n, int nx, int useCov, 
                 TDMatrix covEst, double *depths){

	int* counters = new int[d + 1];
	TDMatrix A = newM(d+1,d+1);
	unsigned long long div0 = choose(n, d);

	double S = 1;
	if (useCov > 0){
        TDMatrix covX = newM(d,d);;
        for (int k = 0; k < d; k++)
            for (int j = 0; j < d; j++)
                covX[k][j] = covEst[k][j];
        // deleteM(covXtemp); // no need to compute anymore
        //S = pow(fabs(determinant(covX,d)),-0.5);
        S = pow(fabs(getDet(covX, d)), -0.5);
	}

	for (int obs = 0; obs < nx; obs++){
		long double sumVolume = 0;
		unsigned long long numSimplicesChecked = 0;

		int p = d - 1;
		for (int i = 0; i < p; i++){ counters[i] = i; }counters[p] = p - 1;
		while (counters[0] != n - (p + 1)){
			int i = p;
			while (i > 0 && counters[i] == n - (p + 1) + i){ i--; }
			counters[i]++; int j = i + 1;
			while (j < p + 1){ counters[j] = counters[j - 1] + 1; j++; }
			for (int j = 0; j < d; j++){
				for (int k = 0; k < d; k++){
					A[j + 1][k] = X[counters[k]][j];
				}
			}
			for (int j = 0; j < d; j++){
				A[j + 1][d] = x[obs][j];
			}
			for (int k = 0; k < d + 1; k++){
				A[0][k] = 1;
			}
			//double volume = fabs(determinant(A,d+1));
            double volume = fabs(getDet(A, d + 1));
			sumVolume += volume;
			numSimplicesChecked ++;
		}
		double O = sumVolume / fact(d) / div0;

		double depth = 1/(1+O*S);
		depths[obs] = depth;
	}
	
	delete[] counters;
}

void OjaDepthsApx(TDMatrix X, TDMatrix x, int d, int n, int nx,
	unsigned long long k, int useCov, TDMatrix covEst, double *depths){

	int* counters = new int[d + 1];
	TDMatrix A =newM(d+1,d+1);
	
	double S = 1;
	if (useCov > 0){
	  TDMatrix covX =newM(d,d);;
	  for (int l = 0; l < d; l++)
	    for (int j = 0; j < d; j++)
	      covX[l][j] = covEst[l][j];
	  // deleteM(covXtemp); // no need to compute anymore
	  S = pow(fabs(determinant(covX,d)), -0.5);
	}

	for (int obs = 0; obs < nx; obs++){

		long double sumVolume = 0;

		for (unsigned long long i = 0; i < k; i++){
			// Generate a combination of indices
			for (int j = 0; j < d; j++){
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
				for (int l = 0; l < d; l++){
					A[j + 1][ l] = X[counters[l]][j];
				}
			}
			for (int j = 0; j < d; j++){
				A[j + 1][ d] = x[obs][j];
			}
			for (int l = 0; l < d + 1; l++){
				A[0][ l] = 1;
			}
			double volume = fabs(determinant(A,d+1));
			sumVolume += volume;
		}
		double O = sumVolume / fact(d) / k;

		double depth = 1 / (1 + O*S);
		depths[obs] = depth;
	}
	delete[] counters;
}
