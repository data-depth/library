/******************************************************************************/
/* File:             ZD.cpp                                                   */
/* Created by:       Rainer Dyckerhoff                                        */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains functions that compute the zonoid depth of a point w.r.t. a       */
/* data cloud.                                                                */
/*                                                                            */
/******************************************************************************/

#include <algorithm>
#include <float.h>
#include "ZD.h"


using namespace std;
using namespace dyMatrixClass;

namespace DataDepth {

	/* Definition of constants */
	const double eps = 1e-8;                // recommended value 1e-8;
	const double accuracy = 1e-10;          // recommended value 1e-10;
	const int MaxIt = 100000;               // recommended value 1000; 

	struct SortRec {
		double v;
		const double* p; // Pointer to a point
	};

	// An object of class 'cZonoidDepth' is created in 'ZD'. 
	// This object is responsible for performing the revised simplex
	// algorithm which is used in the algorithm for computing the 
	// zonoiud depth

	class cZonoidDepth {
	private:
		int n, d, ItCount;
		double lowerbound;
		const double* const* x;
		const double* z;
		vector<vector<double>> rs;
		vector<int> bv;
		vector<bool> znegative;
		vector<SortRec> x_sort;
		vector<unsigned short> RowInverted;
		void RSInit();
		void MakeCanonical();
		void CancelRow(int ip);
		bool AddColumn();
		bool NonBasis(int v);
		bool PhaseIGeneratePivotColumn(int* PivotColumn);
		int FindPivotRow();
		void RSStep(int PivotRow, int PivotColumn);
		bool NoZeroRow(int* PivotRow, int* PivotColumn);
	public:
		double depth(const double* zz, const double* const* xx, int nPoints, int nDim, int& Error);
	};


	// Dynamically allocated arrays: rs, bv

	void cZonoidDepth::RSInit() {
		/* Initialize the revised simplex tableau. */
		/* Basis = Identity matrix. */
		rs.resize(d + 2);
		for (int i = 0; i < d + 2; i++) rs[i].resize(d + 3);
		for (int i = 1; i <= d + 1; i++)
			for (int j = 1; j <= d + 1; j++) rs[i][j] = (i == j);
		/*  All simplex multipliers are equal to unity. */
		for (int j = 1; j <= d + 1; j++) rs[0][j] = 1;
		/* RHS = z,1  */
		/* Objective = 1 + sum of the z_i  */
		rs[0][d + 2] = rs[d + 1][d + 2] = 1;
		for (int i = 1; i <= d; i++)
			rs[0][d + 2] += rs[i][d + 2] = (znegative[i-1] ? -z[i - 1] : z[i-1]);
		/* Initially all basis variables are artificial variables. */
		bv.resize(d + 1);
		for (int i = 0; i <= d; i++) bv[i] = -1;
	}

	void cZonoidDepth::MakeCanonical() {
		/* Convert master problem to canonical form. */
		znegative.resize(d);
		for (int j = 0; j < d; j++) znegative[j] = (z[j] < 0);
	}

	void cZonoidDepth::CancelRow(int ip) {
		/* Delete a zero row from the RS tableau. */
		for (int i = 0; i <= d + 1; i++) rs[i][ip] = 0;
		for (int j = 1; j <= d + 2; j++) rs[ip][j] = 0;
	}

	bool cZonoidDepth::AddColumn() {
		/* Solve the subproblem, generate the pivot column and adjoin it to the
		   to the RS tableau. */
		/* Generate the coefficient of the subproblem's objective. */
		for (int k = 0; k < n; k++) {
			x_sort[k].v = 0;
			for (int j = 0; j < d; j++)	x_sort[k].v += rs[0][j + 1] * (znegative[j] ? -x[k][j] : x[k][j]);
			x_sort[k].p = x[k];
		}
		/* Sort the coefficients in decreasing order. */
		sort(x_sort.begin(), x_sort.end(), [](SortRec a, SortRec b) { return a.v > b.v; });
		/* Find the maximum of the subproblem as well as the extreme point
		   at which it is assmed. */
		int card{ 0 };
		double max = -rs[0][d + 1];
		double sum{ -1 };
		for (int k = 1; k <= n; k++) {
			sum += x_sort[k - 1].v;
			double rtmp = sum / k;
			if (rtmp > max) {
				max = rtmp;
				card = k;
			}
		}
		max += rs[0][d + 1];

		/* If the maximum is less than zero, the value of the objective of the
		   MP cannot be decreased. */
		if (max < eps) return false; /* Solution found. */
		/* If the relative error is less than 'accuracy', the iteration is stopped
		   as well. */
		if (rs[0][d + 2] - max > lowerbound) lowerbound = rs[0][d + 2] - max;
		if ((rs[0][d + 2] - lowerbound) / lowerbound < accuracy) return false;
		/* If the number of iterations exceeds 'MaxIt', the iteration is stopped. */
		if (++ItCount > MaxIt) return false;

		/*  Generate the new pivot column for the MP. */
		rs[0][0] = max;
		for (int i = 1; i <= d + 1; i++) rs[i][0] = rs[i][d + 1];
		for (int j = 0; j < d; j++) {
			double sum{ 0 };
			for (int k = 0; k < card; k++) sum += (znegative[j] ? -x_sort[k].p[j] : x_sort[k].p[j]);
			sum /= card;
			for (int i = 1; i <= d + 1; i++) rs[i][0] += rs[i][j + 1] * sum;
		}
		return true;
	}

	bool cZonoidDepth::NonBasis(int v) {
		/* Check whether 'v' is a basis variable. */
		for (int i = 0; i <= d; i++) if (bv[i] == v) return false;
		return true;
	}

	bool cZonoidDepth::PhaseIGeneratePivotColumn(int* PivotColumn) {
		/* Generate the new pivot column in phase I of the simplex algorithm. */
		/* Find the pivot column */
		rs[0][0] = -rs[0][d + 1];
		*PivotColumn = 0;
		for (int k = 1; k <= n; k++)
			if (NonBasis(k)) {
				double rtmp = 0;
				for (int j = 1; j <= d; j++) rtmp += rs[0][j] * (znegative[j - 1] ? -x[k - 1][j - 1] : x[k - 1][j - 1]);
				if (rtmp > rs[0][0]) {
					rs[0][0] = rtmp;
					*PivotColumn = k;
				}
			}

		if ((rs[0][0] += rs[0][d + 1]) < eps) return false;
		/*  Generate the  pivot column */
		for (int i = 1; i <= d + 1; i++) {
			rs[i][0] = rs[i][d + 1];
			for (int j = 1; j <= d; j++) rs[i][0] += rs[i][j] * (znegative[j-1] ? -x[*PivotColumn - 1][j - 1] : x[*PivotColumn - 1][j - 1]);
		}
		return true;
	}

	int cZonoidDepth::FindPivotRow() {
		/* Find the pivot row. */
		vector<int> I;
		I.resize(d + 1);
		double min = DBL_MAX;
		for (int i = 1; i <= d + 1; i++)
			if (rs[i][0] > eps) {
				double quot = rs[i][d + 2] / rs[i][0];
				if (quot <= min + eps) {
					if (quot < min - eps) {
						I.clear();
						min = quot;
					}
					I.push_back(i);
				}
			}
		if (I.size() <= 1)
			return I[0];
		else
			return I[rand() % I.size()];
	}

	void cZonoidDepth::RSStep(int PivotRow, int PivotColumn) {
		/* Update the revised simplex tableau. */
		/* Calculate the new tableau. */
		double pivot = rs[PivotRow][0];
		for (int j = 1; j <= d + 2; j++) {
			rs[PivotRow][j] /= pivot;
			for (int i = 0; i <= d + 1; i++)
				if (i != PivotRow) rs[i][j] -= rs[PivotRow][j] * rs[i][0];
		}
		/* 'PivotColumn' goes into the basis. */
		bv[PivotRow - 1] = PivotColumn;
	}

	bool cZonoidDepth::NoZeroRow(int* PivotRow, int* PivotColumn) {
		/* Check if a given row of the is a zero row. If a nonzero element is
		found, it is returned in '*PivcotColumn'. */
		/* Find a non-zero element. */
		*PivotColumn = 0;
		for (int k = n; k > 0; k--)
			if (NonBasis(k)) {
				double rtmp = rs[*PivotRow][d + 1];
				for (int j = 1; j <= d; j++) rtmp += rs[*PivotRow][j] * (znegative[j - 1] ? -x[k - 1][j - 1] : x[k - 1][j - 1]);
				if (fabs(rtmp) > eps) {
					*PivotColumn = k;
					for (int i = 0; i <= d + 1; i++) {
						rs[i][0] = rs[i][d + 1];
						for (int j = 1; j <= d; j++)
							rs[i][0] += rs[i][j] * (znegative[j - 1] ? -x[*PivotColumn - 1][j - 1] : x[*PivotColumn - 1][j - 1]);
					}
					return true;
				}
			}
		return false;
	}


	/****************************************************************************/
	/* 'cZonoidDepth::depth' computes the zonoid depth for multivariate data.   */
	/*                                                                          */
	/* This the only publicly callable function of an object of class           */
	/* 'cZonoidDepth'. Here the actual calculation of the zonoid depth is done. */
	/*                                                                          */
	/* For a description of the algorithm, see:                                 */
	/* Dyckerhoff, R., Koshevoy, G., and Mosler, K. (1996)                      */
	/* Zonoid Data Depth : Theory and Computation,                              */
	/* in : Compstat - Proceedings in Computational Statistics, (Albert         */
	/* Prat, ed.), Physica-Verlag, Heidelberg, p. 235--240.                     */
	/*                                                                          */
	/* Args:                                                                    */
	/*   zz - the point for which to calculate the depth, (vector of dimension  */
	/*        d)                                                                */
	/*   xz - the data w.r.t. which the depth has to be computed, (matrix of    */
	/*        dimension n x d)                                                  */
	/*   nPoints - number of the data points,                                   */
	/*   nDim    - dimension of the Euclidean space,                            */
	/*   Error   - error indicator, possible error codes are                    */
	/*               0: no error,                                               */
	/*               1 : simplex algorithm did not terminate within 'MaxIt'     */
	/*					 iterations,                                            */
	/*	             2 : not enough memory available,                           */
	/*                                                                          */
	/* Returns:                                                                 */
	/*   zonoid depth of z w.r.t. x. (if error code is 0)                       */
	/*	 If the error code is 1, the return value is an lower bound to the      */
	/*   zonoid data depth of 'z'. If the error code is 2, the return value     */
	/*   is - 1.                                                                */
	/*                                                                          */
	/****************************************************************************/

	double cZonoidDepth::depth(const double* zz, const double* const* xx, int nPoints, int nDim, int& Error) {
		n = nPoints;
		d = nDim;

		Error = 0;
		double result = 0.0;

		x = xx;
		z = zz;

		MakeCanonical();  /* Convert tableau to canonical form. */

		/* Phase I */

		RSInit(); /* Initialize tableau und basis variables. */
		/* Try to eliminate the artificial variables from the basis to get a
		   basic feasible solution. */
		int PivotColumn; 
		while (PhaseIGeneratePivotColumn(&PivotColumn))
			RSStep(FindPivotRow(), PivotColumn);
		/* If the maximum objective is greater than zero, no basic feasible
		   solution exists. Thus, 'z' lies outside the convex hull of 'x' and the
		   zonoid data depth is 0. */
		if (fabs(rs[0][d + 2]) < eps) {
			/* Check if there are still artificial variables on zero level in the basis
			   and remove them from the basis. */
			for (int row = 1; row <= d + 1; row++)
				if (bv[row - 1] < 0) {
					if (NoZeroRow(&row, &PivotColumn))
						RSStep(row, PivotColumn);
					else
						CancelRow(row);
				}

			/*  Phase II  */

			/* Try to allocate memory for 'x_sort'. */
//			x_sort = new(nothrow)SortRec[n];
			x_sort.resize(n);
			if (x_sort.size() == n) { /* Allocation successful. */
				lowerbound = 1.0 / n; /* Lower bound for the objective of the MP. */
				/* Reinitialize the objective of the MP. */
				for (int j = 1; j <= d + 2; j++) {
					rs[0][j] = 0;
					for (int k = 1; k <= d + 1; k++) rs[0][j] += rs[k][j];
				}
				/* Revised simplex algorithm */
				ItCount = 0;
				while (AddColumn())
					RSStep(FindPivotRow(), 0);
				if (ItCount > MaxIt) {
//					cout << "Error\n";
					Error = 1;
				}
				result = 1.0 / (n * rs[0][d + 2]); /* Return zonoid data depth. */
			}
			else { /* Memory for 'x_sort' could not be allocated. */
				Error = 2;
				result = -1.0;
			}
		}
		return result;
	}


	/* Definition of public functions */

    /****************************************************************************/
    /*                                                                          */
	/* The following functions can be called to calculate the zonoid depth of   */
	/* a point w.r.t. a multivariate data cloud.                                */
	/*                                                                          */
	/* Thes efunctions are simple wrapper functions that create an object of    */
	/* class 'cZonoidDepth', initialize it and do the actual computation.       */
    /*                                                                          */
    /****************************************************************************/

	double ZD(const double* zz, const double* const* xx, int nPoints, int nDim, int& Error) {
		cZonoidDepth zon{};
		return zon.depth(zz, xx, nPoints, nDim, Error);
	}

	double ZD(const double* zz, const cMatrix xx, int nPoints, int nDim, int& Error) {
		cZonoidDepth zon{};
		unique_ptr<double*[]> x{ new double*[nPoints] };
		for (int i = 0; i < nPoints; i++) x[i] = xx[i];
		return zon.depth(zz, x.get(), nPoints, nDim, Error);
	}


	/****************************************************************************/
	/* ZD1 computes the zonoid depth for univariate data.                       */
	/*                                                                          */
	/* The algorithm used here is capable of computing the zonoid depth with a  */
	/* complexity of O(n). The algorithm will be published in a forthcoming     */
	/* paper. Do not use this algorithm without permission of the author.       */
	/*                                                                          */
	/* Args:                                                                    */
	/*   z - the point for which to calculate the depth,                        */
	/*   x - the data w.r.t. which the depth has to be computed, (vector of     */
	/*       dimension n),                                                      */
	/*   n - number of the data points.                                         */
	/*   sorted - optional parameter that indicates that the array x is sorted  */
	/*            in ascending order when 'sorted' is true. Otherwise, the      */
	/*            array x is not sorted                                         */
	/* Returns:                                                                 */
	/*   zonoid depth of z w.r.t. x.                                            */
	/****************************************************************************/

	double ZD1(double z, const double* xx, int n, bool sorted) {
		double Sum = 0, SumNeu, mark, delta, temp, min = DBL_MAX, max = -DBL_MAX;
		double *x;
		int l, r, i, j;

		x = new double[n];
		for (int i = 0; i < n; i++) {
			x[i] = xx[i];
			Sum += x[i];
		};
		if (z * n == Sum) return 1.0;

		if (z * n > Sum) {
			for (int i = 0; i < n; i++) x[i] = -x[i];
			z = -z;
		}
		l = 0;
		r = n - 1;
		Sum = 0;
		while (l < r) {
			i = l;
			j = r;
			mark = x[(l + r) / 2];
			/* A problem could occur, if mark = x[r] and all other elements are less than a[r].
			  In that case i == r+1 after the do-while-loop and we run into an infinite loop.
			  */
			do {
				while (x[i] < mark) i++;
				while (x[j] > mark) j--;
				if (i <= j) {
					temp = x[i];
					x[i] = x[j];
					x[j] = temp;
					i++;
					j--;
				}
			} while (i <= j);
			SumNeu = Sum;
			for (int k = l; k < i; k++) SumNeu += x[k];
			if (SumNeu <= z * i) {
				l = i;
				Sum = SumNeu;
			}
			else {
				r = i - 1;
			}
		}
		delta = (l * z - Sum) / (x[l] - z);

		delete[] x;
		return (l + delta) / n;
	}



}
