// #include "MCD.h"
#include "stdafx.h"
// #include <algorithm>
// #include <cfloat>
// #include "Common.h"
// #include "PD.cpp"
// #include "Mahalanobis.h"
#include <iomanip>

vector<int> split(int n){//split n into (at most 5) bins and return the vector v of number of elements in each bin
	int k, val, remainder;
	// Initialising v
	if (n>=1500){
		k = 5;
		val = 300;
	}
	else{
		k = (int) n/300;
		if (n % 300 == 0) k = k-1;
		val = (int) n/k;
	}
	
	vector<int> v(k,val);//will contain number of elements in each bin

	// Adding extra numbers to add up exactly to n
	if (n<1500){
		remainder = n - k*val;
		for (int i=0; i < v.size(); i++){
			if (remainder>0){
				v[i] += 1; 
				remainder -= 1;
			}
		}
	}
	return v;
}

vector<int> best(int p, vector<double>& all_det ,int rep){ // take the p indices with smallest determinant, rep assumed considered size of all_det
	// take the 10 best results
	vector<int> index(rep, 0); //index to help find 10 best
	iota(index.begin(), index.end(), 0);
	std::nth_element(index.begin(),index.begin()+p-1,index.end(),
		[&](const int& a, const int& b) {
			return (all_det[a] < all_det[b]);
		}
	);
	return index;
}


double det(TDMatrix M, int d){
	TDMatrix A = copyM(M, d, d);
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
	deleteM(A);
    return det;
}

void biased_cov(TDMatrix X, int n, int d, TDMatrix covX) {// same code as cov but division by n instead of n-1
	// covX should be of size d by d !
	double* means = new double[d];
	double* dev = new double[d];
	// zeroing TDMatrix
	// TDMatrix covX = newM(d, d);
	for (int k = 0; k < d; k++){
		for (int j = 0; j < d; j++){
			covX[k][j] = 0;
		}
	}	
	// means
	for (int i = 0; i < d; i++) {
		means[i] = 0.0;
		for (int j = 0; j < n; j++){
			means[i] += X[j][i];
		}
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
			covX[i][j] /= n;
		}
	}
	delete[] means;
	delete[] dev;
}


void unbiased_cov(TDMatrix X, int n, int d, TDMatrix covX) {
	double* means = new double[d];
	double* dev = new double[d];
	// zeroing TDMatrix
	for (int k = 0; k < d; k++){
		for (int j = 0; j < d; j++){
			covX[k][j] = 0;
		}
	}
	// means
	for (int i = 0; i < d; i++) {
		means[i] = 0.0;
		for (int j = 0; j < n; j++){
			means[i] += X[j][i];
		}
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
}

void IndexUp(vector<int>& index, double* distTab){	
	// Sorting of distances index
	// inspired by https://stackoverflow.com/questions/17554242/how-to-obtain-the-index-permutation-after-the-sorting
	// assume index.size() is the corresponding n !
	iota(index.begin(), index.end(), 0);//fill with numbers from 1 to n //maybe use the traditional way if parallelisation
    std::sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (distTab[a] < distTab[b]);
        }
    );
}


void MeanCovUp(vector<int> &index, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h){//TDMatrix or double**?   can get rid of X !
	// Xh should be of size h by d
	// Select first h elements of X with best distances
	for(int i=0;i<h;i++){
		for(int j=0;j<d;j++){
			Xh[i][j] = X[index[i]][j];
		}
	}
	// Finally update T,S, then distTab
	// T :=is the mean of Xh
	for (int i = 0; i < d; i++) {
		T[i] = 0.0;
		for (int j = 0; j < h; j++){
			T[i] += Xh[j][i];
		}
		T[i] /= h;
	}
	biased_cov(Xh, h, d, S);
}

void DistanceUp(TDMatrix X, int n,  int d, double* distTab, double* T,TDMatrix S){
	TDMatrix s = copyM(S, d, d);
	int rank, error;
	InversePosDef(s,d,rank,error);
    for(int ii=0;ii<d;ii++){
		for(int jj=0;jj<ii;jj++){
            s[jj][ii]=s[ii][jj];
		}
	}
	double *a = new double[d];
	for (int i = 0; i < n; i++){
		distTab[i] = 0;
		for (int k = 0; k < d; k++){
			a[k] = 0;
			for (int j = 0; j < d; j++){
				a[k] += (X[i][j] - T[j])*s[j][k];
			}
		}
		for (int j = 0; j < d; j++){
			distTab[i] += (X[i][j] - T[j])*a[j];
		}
	}
	for (int i = 0; i < n; i++){
		distTab[i] = sqrt(distTab[i]);
	}
	deleteM(s);
	delete[] a;
}

void cstep(vector<int>& index, double* distTab, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h){
	// start assuming index already has the knowledge
	MeanCovUp(index, T, S, X, Xh, n, d, h);
	DistanceUp(X, n, d, distTab, T, S);
	IndexUp(index, distTab);
}

void cstep_TSstart(vector<int>& index, double* distTab, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h){
	// start assuming T mean and S covariance already have the knowledge
	DistanceUp(X, n, d, distTab, T, S);
	IndexUp(index, distTab);
	MeanCovUp(index, T, S, X, Xh, n, d, h);
}

void finite_mcd_routine(int nstep, vector<int>& index, double* distTab, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h){
	for (int i = 0; i < nstep; i++){
		cstep(index, distTab, T, S, X, Xh, n, d, h);
	}
}

void mcd_routine(vector<int>& index, double* distTab, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h){
	MeanCovUp(index, T, S, X, Xh, n, d, h);
	double detVal = det(S,d);
	double newDetVal;
	cstep(index, distTab, T, S, X, Xh, n, d, h);
	MeanCovUp(index, T, S, X, Xh, n, d, h);
	newDetVal = det(S,d);
	while(detVal > newDetVal){ // or !=
		detVal = newDetVal;
		cstep(index, distTab, T, S, X, Xh, n, d, h);
		MeanCovUp(index, T, S, X, Xh, n, d, h);
		newDetVal = det(S,d);
	}
}

void ExactUnivariateMcd(TDMatrix X, int n, int h, double* T, TDMatrix M){
	// Remember it makes no sense to pick h=0, h should be >0
	vector<double> means(n-h+1,0); // means on h elements
	vector<double> SQ(n-h+1,0); // sum of squares on h elements
	double float_h = (double) h;
	// Compute order statistics
	vector<double> Y(n);
	for(int i=0; i<n; i++) Y[i] = X[i][0];
	sort(Y.begin(),Y.end());
	// Init:First one
	// mean
	for(int i=0; i<h; i++) means[0] += Y[i];
	means[0] = means[0] / float_h;
	// SQ
	double temp;
	for(int i=0; i<h; i++){
		temp = Y[i]-means[0];
		SQ[0] += temp*temp;
	}
	// Compute the others by iteration, and look for optimal
	int bestIndex = 0;
	double bestSQ = SQ[0];
	// cout << " PROCESS " << endl;
	for(int i=1; i<n-h+1; i++){
		means[i] = (float_h*means[i-1]-Y[i-1]+Y[i+h-1])/float_h;
		SQ[i] = SQ[i-1] - Y[i-1]*Y[i-1] + Y[i+h-1]*Y[i+h-1] - float_h*means[i]*means[i] + float_h*means[i-1]*means[i-1];
		cout <<  SQ[i] <<  " ";
		if (SQ[i]<SQ[bestIndex]) bestIndex = i;
	}
	T[0] = means[bestIndex];// should actually be 1x1
	M[0][0] = SQ[bestIndex]/float_h;
}

void Mcd(TDMatrix X, int n, int d, int h, double* mat_MCD, double chisqr05, double chisqr0975, int mfull, int nstep, bool hiRegimeCompleteLastComp,int *seed, bool seeded){
	TDMatrix M = asMatrix(mat_MCD,d,d); //address of the matrix to be computed
	TDMatrix Xh = newM(h, d);
	double* T = new double[d];
	double* distTab = new double[n];
	vector<int> index(n, 0);
	std::random_device rd;
	std::mt19937 g(rd());
	if (seeded){
		g.seed(*seed);
	}
    int bestIndex;
    double finalDet = DBL_MAX;
    double tempDet;
	if (h==n){
		//return original covariance matrix
		biased_cov(X,n,d,M);
	}
	else if (d==1){
		// use original Rousseuw and Leroy 1987 algorithm
		// cout << "UNIVARIATE METHOD" << endl;
		ExactUnivariateMcd(X, n, h, T, M); //or M?
	}
	else{// h<n,d>=2
		if (n>600){
			// randomly shuffled copy of data
			int nNest = min(n,1500);
			TDMatrix Xnest = newM(nNest, d);
			vector<int> tempInd(n, 0);
			iota(tempInd.begin(), tempInd.end(), 0);
			std::shuffle(tempInd.begin(), tempInd.end(), g);
			for (int i=0; i<nNest ;i++){
				for(int j=0;j<d;j++){
					Xnest[i][j] = X[tempInd[i]][j];
				}
			}

			vector<int> binNum = split(n);
			int k = binNum.size();
			vector<int> splitIndices(k,0);
			for (int i=1; i < k; i++){
				splitIndices[i] = splitIndices[i-1] + binNum[i-1];
			}
			int rep = (int)ceil((double)500/k);
			int repTotal = rep * k;
			vector<vector<vector<int>>> all_index(k, vector<vector<int>>(rep));
			vector<vector<double>> all_det(k, vector<double>(rep));
			vector<vector<TDMatrix>> all_cov(k, vector<TDMatrix>(rep));
			vector<vector<double*>> all_T(k, vector<double*>(rep));
			vector<vector<double*>> all_distTab(k, vector<double*>(rep));
			vector<vector<int>> best_indices(k);//ou alors ne pas initialiser et remplir direct plus tard avec best func
			vector<int> hsub(k);
			for (int i = 0;i<k;i++){
				hsub[i] = (int)ceil((double) (h*binNum[i]/n));
				TDMatrix Xhsub = newM(hsub[i], d);
				for (int j = 0;j<rep;j++){
					all_cov[i][j] = newM(d,d);
					all_T[i][j] = new double[d];
					all_distTab[i][j] = new double[n];
					all_index[i][j].resize(binNum[i]);
					iota(all_index[i][j].begin(), all_index[i][j].end(), 0);
					std::shuffle(all_index[i][j].begin(), all_index[i][j].end(), g);
					cstep(all_index[i][j], all_distTab[i][j], all_T[i][j], all_cov[i][j], Xnest + splitIndices[i], Xhsub, binNum[i], d, hsub[i]);
					cstep(all_index[i][j], all_distTab[i][j], all_T[i][j], all_cov[i][j], Xnest + splitIndices[i], Xhsub, binNum[i], d, hsub[i]);
					MeanCovUp(all_index[i][j], all_T[i][j], all_cov[i][j], Xnest + splitIndices[i], Xhsub, binNum[i], d, hsub[i]);
					all_det[i][j] = det(all_cov[i][j],d);
				}
				deleteM(Xhsub);
				best_indices[i] = best(10, all_det[i] , rep);
			}

			// Merging

			// First count the occurences
			int counters[1500] = {0};
			for (int i = 0;i<k;i++){ 
				for (int m = 0;m<10;m++){
					for (int p=0;p<hsub[i];p++){
					counters[all_index[i][best_indices[i][m]][p]+splitIndices[i]] += 1; //index corresponds to the element in Xnest which is the p-th element of the m-th best solution in nest i
					}
				}
			}
			// Count how many distinct elements (nMerged) in Xmerged
			int nMerged = 0;
			// int sum = 0;
			for (int i=0;i<1500;i++){
				if ( counters[i] > 0 ) nMerged +=1;
			}

			// Now build Xmerged
			TDMatrix Xmerged = newM(nMerged, d);
			// Reinitialise counters, to then check first occurence or not
			for (int i=0;i<1500;i++) counters[i] = 0;
			// Fill in Xmerged
			int movingInd = 0;// index for Xmerged filling step by step
			int tempoInd;// current index when checking through the best solutions to avoid recomputation of the index at each call
			for (int i = 0;i<k;i++){ 
				for (int m = 0;m<10;m++){
					for (int p=0;p<hsub[i];p++){
						tempoInd = all_index[i][best_indices[i][m]][p]+splitIndices[i];
						if ( counters[tempoInd] == 0 ){//"if not already seen", avoid duplicating
							counters[tempoInd] += 1;//avoid future duplicates
							//copy
							for (int j=0;j<d;j++){
								Xmerged[movingInd][j] = Xnest[tempoInd][j];
							}
							movingInd +=1;
						}
					}
				}
			}
			// Compute on nSelect (10 best (T,S) of each subsets)
			int nSelect = k*10;
			int hMerged =  (int)ceil((double) (h*nMerged/n));
			double* mergedDistTab = new double[nMerged];
			TDMatrix XhMerged = newM(hMerged, d);
			vector<double> merged_all_det(nSelect);
			vector<vector<int>> merged_all_index(nSelect, vector<int>(nMerged));
			int indConvert;
			for (int i = 0;i<k;i++){ 
				for (int m = 0;m<10;m++){
					indConvert = i*10+m;
					iota(merged_all_index[indConvert].begin(), merged_all_index[indConvert].end(), 0); // vector ={0,..,nMerged-1}
					std::shuffle(merged_all_index[indConvert].begin(), merged_all_index[indConvert].end(), g);
					cstep_TSstart(merged_all_index[indConvert], mergedDistTab, all_T[i][best_indices[i][m]],all_cov[i][best_indices[i][m]], Xmerged, XhMerged, nMerged, d, hMerged);
					cstep_TSstart(merged_all_index[indConvert], mergedDistTab, all_T[i][best_indices[i][m]],all_cov[i][best_indices[i][m]], Xmerged, XhMerged, nMerged, d, hMerged);
					merged_all_det[indConvert] = det(all_cov[i][best_indices[i][m]],d); //all_cov[i][best_indices[i][m]] has been changed inplace right before
				}
			}

			vector<int> mergedBest = best(mfull, merged_all_det , nSelect);
			// Full dataset computation

			// for simplicity recopy the previous best results (T,S) in a fresh compilation
			vector<TDMatrix> full_all_cov(mfull);
			vector<double*> full_all_T(mfull);
			int convertedM, convertedI; // i<k, m <10 vs ind<nSelect
			for(int i = 0; i<mfull ; i++){
				convertedM = mergedBest[i] % 10 ;
				convertedI = mergedBest[i] / 10 ;
				full_all_cov[i] = copyM(all_cov[convertedI][best_indices[convertedI][convertedM]],d,d);
				full_all_T[i] = all_T[convertedI][best_indices[convertedI][convertedM]]; //be careful not to erase original!
			}
			for(int i = 0; i<mfull ; i++){
				DistanceUp(X, n, d, distTab, full_all_T[i], full_all_cov[i]);
				IndexUp(index, distTab);
				if (hiRegimeCompleteLastComp){
					mcd_routine(index, distTab, full_all_T[i], full_all_cov[i],  X,  Xh, n, d, h);
				}
				else{
					finite_mcd_routine(nstep, index, distTab, full_all_T[i], full_all_cov[i],  X,  Xh, n, d, h);
				}
				tempDet = det(full_all_cov[i],d);
				if (tempDet < finalDet){
					finalDet = tempDet;
					bestIndex = i;
				}
			}
			// copy result in T and M
			for(int i=0 ; i<d ; i++){
				T[i] = full_all_T[bestIndex][i];
				for (int j=0; j<d ; j++){
					M[i][j] = full_all_cov[bestIndex][i][j];
				}
			}
			// Deleting
			// Nested
			deleteM(Xnest);
			// !!! don't forget to delete the double* inside all_T,all_cov, all_distTab (maybe further there)
			for (int i = 0;i<k;i++){
				for (int j = 0;j<rep;j++){
					delete[] all_distTab[i][j];
					delete[] all_T[i][j];//à voir si on bouge plus loin ou copie
					deleteM(all_cov[i][j]);// ou delete[], à voir à bouger plus loin
				}
			}
			//Merged
			deleteM(Xmerged);
			delete[] mergedDistTab;
			deleteM(XhMerged);
			//Full
			for (int i = 0 ; i<mfull ; i++){
				deleteM(full_all_cov[i]);
			}


		}
		else{// h<n,d>=2,n<=600
			// initialise index vector
		    for (int i = 0 ; i != index.size() ; i++) { // assume index.size() is n !
				index[i] = i;
			}
			// big loop 500
			TDMatrix* all_cov = new TDMatrix[500];
			vector<int>* all_index = new vector<int>[500];
			double* all_det = new double[500];
			for(int i=0;i<500;i++){
				// random p+1 subset initialisation + Cstep
				all_index[i] = vector<int>(n);
				all_cov[i] = newM(d , d);
				iota(all_index[i].begin(), all_index[i].end(), 0);
				std::shuffle(all_index[i].begin(), all_index[i].end(), g);// can we improve complexity?
				/// before check h vs d+1 !!!!!!!!!!
                // two C-steps
				cstep(all_index[i], distTab, T, all_cov[i], X, Xh, n, d, d+1);//parallelisation?
				cstep(all_index[i], distTab, T, all_cov[i], X, Xh, n, d, h);//parallelisation?
                // update Cov & T according to most recent update of index
				MeanCovUp(all_index[i], T, all_cov[i], X, Xh, n, d, h);
				all_det[i] = det(all_cov[i],d);
			}			
			// take the 10 best results
			vector<int> index500(500, 0); //index to help find 10 best
			iota(index500.begin(), index500.end(), 0);
			std::nth_element(index500.begin(),index500.begin()+9,index500.end(),
				[&](const int& a, const int& b) {
					return (all_det[a] < all_det[b]);
				}
			); //ten first elements now are index of ten smallest det
            int ind;
			// run until convergence for each of the 10 best
			for(int i=0;i<10;i++){
				ind = index500[i];
				// run until convergence
				mcd_routine(all_index[ind], distTab, T, all_cov[ind], X, Xh, n, d, h);
				tempDet = det(all_cov[ind],d);
				if (tempDet<finalDet) {
					finalDet = tempDet;
					bestIndex = ind;
				}
			}
			MeanCovUp(all_index[bestIndex], T, M, X, Xh, n, d, h);
		}

	}


    cout << " M cov " << endl;
    for (int k=0; k < d; k++){
    	for (int p=0; p < d; p++){
    		std::cout << M[k][p] << " " ;
    	}
    	std::cout << std::endl ;
    }

	// Ultimate reweighting
	// std::cout << "det M before reweighting " << det(M,d) << std::endl;
	DistanceUp(X, n, d, distTab, T, M);
	double medi = DataDepth::med(distTab,n);
	double medi2 = medi*medi;
	// if (medi2==0){
	// 	for(int i=0;i<n;i++) distTab[i] = 0;
	// }
	// else{
	// 	for(int i=0;i<n;i++) distTab[i] *= sqrt(1.39/medi2);//sqrt(chisqr(d,0.5)/medi2);
	// }
	// // write new array checking which elements are kept
	// bool* indices = new bool[n];
	// int ncount = 0;
	// double chival = 2.72;//sqrt(chisqr(d,0.975));
	// for(int i=0;i<n;i++){
	// 	if (distTab[i] <= chival){
	// 		indices[i] = true;
	// 		ncount += 1;
	// 	}
	// 	else{
	// 	indices[i] = false;
	// 	std::cout << " out ";
	// 	for(int j=0;j<d;j++) std::cout << X[i][j];
	// 	std::cout << std::endl;
	// 	}
	// }

	// Alter. version of reweighting
	for(int i=0;i<d;i++){
		for(int j=0;j<d;j++){
					M[i][j] = medi2 * M[i][j]/chisqr05;
				}
	}
	DistanceUp(X, n, d, distTab, T, M);
	bool* indices = new bool[n];
	int ncount = 0;
	double chival = sqrt(chisqr0975);
	for(int i=0;i<n;i++){
		if (distTab[i] <= chival){
			indices[i] = true;
			ncount += 1;
		}
		else{
		indices[i] = false;
		}
	}


	double** X1 = newM(ncount, d);
	int icount = 0;
	for(int i=0;i<n;i++){
		if (indices[i]) {
			X1[icount] = X[i];
			icount++;
		}
	}
	unbiased_cov(X1,ncount,d,M);
	delete[] indices;
	// End of transforming step
	delete[] T;
	delete[] distTab;
	deleteM(Xh);
	delete[] M;
	delete[] X1;
}