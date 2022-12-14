/*
  File:         depth.cpp
  Created by:   Rainer Dyckerhoff
  Last revised: 15.05.2013

  Computation of the zonoid data depth.

  For a description of the algorithm, see:
  Dyckerhoff, R., Koshevoy, G., and Mosler, K. (1996)
  Zonoid Data Depth: Theory and Computation,
  in: A. Compstat - Proceedings in Computational Statistics
      (Albert Prat, ed.), Physica-Verlag, Heidelberg, pp. 235--240.
*/

#include "stdafx.h"

/* Definition of constants */
static const double eps = 1e-8;
static const double accuracy = 1e-10;
static const int MaxIt = 1000;
/* Definition of types */
typedef vector<vector<double> > TMatrix;  //typedef double TRevSimplexTableau[MaxD + 2][MaxD + 3];
typedef vector<int> TVariablen;          // typedef int  TVariablen[MaxD + 1];

/* Definition of static variables */
static int n, d, ItCount;
static double lowerbound;
static TMatrix rs;
static TVariablen bv;
static vector<SortRec> x_sort;
static vector<unsigned short> RowInverted;

/* Definition of static functions */

static void RSInit(TPoint& z)
/* Initialize the revised simplex tableau. */
{
  int i, j;
  /* Basis = Identity matrix. */
  rs.resize(d+2);
  for (i = 0; i < d+2; i++) rs[i].resize(d+3);
  for (i = 1; i <= d + 1; i++) 
    for (j = 1; j <= d + 1; j++) rs[i][j] = (i == j);
  /*  All simplex multipliers are equal to unity. */
  for (j = 1; j <= d + 1; j++) rs[0][j] = 1;
  /* RHS = z,1  */
  /* Objective = 1 + sum of the z_i  */
  rs[0][d + 2] = rs[d + 1][d + 2] = 1;
  for (i = 1; i <= d; i++)
    rs[0][d + 2] += rs[i][d + 2] = z[i-1];
  /* Initially all basis variables are artificial variables. */
  bv.resize(d+1);
  for (i = 0; i <= d; i++) bv[i] = -1;
}

static void MakeCanonical(vector<TPoint>& x, TPoint& z)
/* Convert master problem to canonical form. */
{
  int i, j;
  RowInverted.resize(d);

  for (j = 0; j < d; j++)
  {
    RowInverted[j] = z[j] < 0; // PM(2018-06-22)
    if (RowInverted[j]) {
      for (i = 0; i < n; i++) x[i][j] = -x[i][j];
      z[j] = -z[j];
    }
  }
}

static void MakeOriginal(vector<TPoint>& x, TPoint& z)
/* Reconstruct the original data. */
{
  int i, j;

  for (j = 0; j < d; j++)
    if (RowInverted[j]) {
      for (i = 0; i < n; i++) x[i][j] = -x[i][j];
      z[j] = -z[j];
    }
}

static void CancelRow(int ip)
/* Delete a zero row from the RS tableau. */
{
  int i, j;

  for (i = 0; i <= d + 1; i++) rs[i][ip] = 0;
  for (j = 1; j <= d + 2; j++) rs[ip][j] = 0;
}

static int Compare(SortRec a, SortRec b)
/* This routine is passed to the sort routine. */
{
  return (a.v > b.v) ;
}

static bool AddColumn(vector<TPoint>& x)
/* Solve the subproblem, generate the pivot column and adjoin it to the
   to the RS tableau. */
{
  int    card, i, j, k;
  double max, sum, rtmp;

  /* Generate the coefficient of the subproblem's objective. */
  for (k = 0; k < n; k++) {
    for (x_sort[k].v = 0, j = 0; j < d; j++) 
      x_sort[k].v += rs[0][j + 1] * x[k][j];
    x_sort[k].p = &x[k];
  }
  /* Sort the coefficients in decreasing order. */
  sort(x_sort.begin(), x_sort.end(), Compare);
  /* Find the maximum of the subproblem as well as the extreme point
     at which it is assmed. */
  card = 0;
  max = -rs[0][d + 1];
  sum = -1;
  for (k = 1; k <= n; k++)
    if ((rtmp = (sum += x_sort[k-1].v) / k) > max) {
      max = rtmp;
      card = k;
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
  if ( ++ItCount > MaxIt ) return false;

  /*  Generate the new pivot column for the MP. */
  rs[0][0] = max;
  for (i = 1; i <= d + 1; i++) rs[i][0] = rs[i][d + 1];
  for (j = 0; j < d; j++) {
    for (sum = 0, k = 0; k < card; k++) sum += x_sort[k].p->operator[](j);
    for (sum /= card, i = 1; i <= d + 1; i++) rs[i][0] += rs[i][j + 1] * sum;
  }
  return true;
}

static bool NonBasis(int v)
/* Check whether 'v' is a basis variable. */
{
  int i;

  for (i = 0; i <= d; i++) if (bv[i] == v) return false;
  return true;
}

static bool PhaseIGeneratePivotColumn(vector<TPoint>& x, int *PivotColumn)
/* Generate the new pivot column in phase I of the simplex algorithm. */
{
  int i, j, k;
  double rtmp;

  /* Find the pivot column */
  rs[0][0] = -rs[0][d + 1];
  *PivotColumn = 0;
  for (k = 1; k <= n; k++)
    if (NonBasis(k)) {
      for (rtmp = 0, j = 1; j <= d; j++) rtmp += rs[0][j] * x[k-1][j-1];
      if (rtmp > rs[0][0]) {
        rs[0][0] = rtmp;
        *PivotColumn = k;
      }
    }

  if ((rs[0][0] += rs[0][d + 1]) < eps) return false;
  /*  Generate the  pivot column */
  for (i = 1; i <= d + 1; i++) {
    rs[i][0] = rs[i][d + 1];
    for (j = 1; j <= d; j++) rs[i][0] += rs[i][j] * x[*PivotColumn-1][j-1];
  }
  return true;
}

static int FindPivotRow()
/* Find the pivot row. */
{
  int i;
  double min, quot;
  vector<int> I;
  
  I.resize(d+1);
  min = DBL_MAX;
  for (i = 1; i <= d + 1; i++)
    if (rs[i][0] > eps) {
      quot = rs[i][d + 2] / rs[i][0];
      if (quot <= min+eps) {
		if (quot < min-eps) {
          I.clear();
          min = quot;
		}
	    I.push_back(i);
      }
	}
	if (I.size() <= 1) 
	  return I[0];
	else
	  return I[random(I.size())];
}

static void RSStep(int PivotRow, int PivotColumn)
/* Update the revised simplex tableau. */
{
  int i, j;
  double pivot;

  /* Calculate the new tableau. */
  pivot = rs[PivotRow][0];
  for (j = 1; j <= d + 2; j++) {
    rs[PivotRow][j] /= pivot;
    for (i = 0; i <= d + 1; i++)
    if (i != PivotRow) rs[i][j] -= rs[PivotRow][j] * rs[i][0];
  }
  /* 'PivotColumn' goes into the basis. */
  bv[PivotRow - 1] = PivotColumn;
}

static bool NoZeroRow(vector<TPoint>& x, int * PivotRow, int * PivotColumn)
/* Check if a given row of the is a zero row. If a nonzero element is
found, it is returned in '*PivcotColumn'. */
{
  int i, j, k;
  double rtmp;

  /* Find a non-zero element. */
  *PivotColumn = 0;
  for (k = n; k > 0; k--)
    if (NonBasis(k)) {
      rtmp = rs[*PivotRow][d + 1];
      for (j = 1; j <= d; j++) rtmp += rs[*PivotRow][j] * x[k-1][j-1];
      if (fabs(rtmp) > eps) {
        *PivotColumn = k;
        for (i = 0; i <= d + 1; i++) {
          rs[i][0] = rs[i][d + 1];
          for (j = 1; j <= d; j++)
            rs[i][0] += rs[i][j] * x[*PivotColumn-1][j-1];
        }
        return true;
      }
    }
  return false;
}

/* Standardizing functions */

int GetMeansSds(vector<TPoint>& x, TPoint *means, TPoint *sds){
/*
	Get means and standard deviations, coordinatewise
*/
	int _n = x.size();int _d = x[0].size();means->resize(_d);sds->resize(_d);
	for (int j = 0; j < _d; j++){
		double tmpMean = 0;double tmpVar = 0;
		for (int i = 0; i < _n; i++){
			tmpMean += x[i][j];
		}
		(*means)[j] = tmpMean/_n;
		for (int i = 0; i < _n; i++){
			tmpVar += std::pow(x[i][j] - (*means)[j], 2);
		}
		(*sds)[j] = sqrt(tmpVar/(_n - 1));
	}
	return 0;
}

int Standardize(vector<TPoint> &x, TPoint& means, TPoint& sds){
/*
	Standardize data cloud, coordinatewise
*/
	int _n = x.size();int _d = x[0].size();
	for (int i = 0; i < _n; i++){
		for (int j = 0; j < _d; j++){
			x[i][j] = (x[i][j] - means[j])/sds[j];
		}
	}
	return 0;
}

int GetMeansSds(TDMatrix& x, int n, int d, TPoint *means, TPoint *sds){
	/*
	Get means and standard deviations, coordinatewise
	*/
	for (int j = 0; j < d; j++){
		double tmpMean = 0; double tmpVar = 0;
		for (int i = 0; i < n; i++){
			tmpMean += x[i][j];
		}
		(*means)[j] = tmpMean / n;
		for (int i = 0; i < n; i++){
			tmpVar += std::pow(x[i][j] - (*means)[j], 2);
		}
		(*sds)[j] = sqrt(tmpVar / (n - 1));
	}
	return 0;
}

int Standardize(TDMatrix &x, int n, int d, TPoint& means, TPoint& sds){
	/*
	Standardize data cloud, coordinatewise
	*/;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < d; j++){
			x[i][j] = (x[i][j] - means[j]) / sds[j];
		}
	}
	return 0;
}

int Standardize(TPoint &x, TPoint& means, TPoint& sds){
/*
	Standardize point, coordinatewise
*/
	int _d = x.size();
	for (int i = 0; i < _d; i++){
			x[i] = (x[i] - means[i])/sds[i];
	}
	return 0;
}

int Unstandardize(vector<TPoint> &x, TPoint& means, TPoint& sds){
/*
	Unstandardize data cloud, coordinatewise
*/
	int _n = x.size();int _d = x[0].size();
	for (int i = 0; i < _n; i++){
		for (int j = 0; j < _d; j++){
			x[i][j] = x[i][j]*sds[j] + means[j];
		}
	}
	return 0;
}

int Unstandardize(TPoint &x, TPoint& means, TPoint& sds){
/*
	Unstandardize point, coordinatewise
*/
	int _d = x.size();
	for (int i = 0; i < _d; i++){
			x[i] = x[i]*sds[i] + means[i];
	}
	return 0;
}

/* Definition of public functions */

double ZonoidDepth(vector<TPoint>& x, TPoint& z, int& Error)
/*
   Calculate the zonoid data depth of the point 'z' with respect to the
   data points 'x'. The number of data points is passed in 'NoPoints',
   the dimension in 'Dimension'. If an error occurs, the error code is
   stored in '*Error'. Possible error codes are:
     0: no error,
     1: simplex algorithm did not terminate within 'MaxIt' iterations.
     2: not enough memory available,
   If no error occured, the return value is the zonoid data depth of 'z'.
   If the error code is 1, the return value is an lower bound to the
   zonoid data depth of 'z'. If the error code is 2, the return value is -1.
*/
{
  int j, k, row, PivotColumn;

  n = x.size();
  d = z.size();

  Error = 0;

  MakeCanonical(x,z);  /* Convert tableau to canonical form. */

  /* Phase I */

  RSInit(z); /* Initialize tableau und basis variables. */
  /* Try to eliminate the artificial variables from the basis to get a
     basic feasible solution. */
  while (PhaseIGeneratePivotColumn(x, &PivotColumn))
    RSStep(FindPivotRow(), PivotColumn);
  /* If the maximum objective is greater than zero, no basic feasible
     solution exists. Thus, 'z' lies outside the convex hull of 'x' and the
     zonoid data depth is 0. */
  if (fabs(rs[0][d + 2]) > eps) {
    MakeOriginal(x,z); /* Reconstruct the original data. */
    return 0;          /* Return zonoid data depth. */
  }
  /* Check if there are still artificial variables on zero level in the basis
     and remove them from the basis. */
  for (row = 1; row <= d + 1; row++)
    if (bv[row - 1] < 0) {
      if (NoZeroRow(x, &row, &PivotColumn))
        RSStep(row, PivotColumn);
      else
        CancelRow(row);
    }

  /*  Phase II  */

  /* Try to allocate memory for 'x_sort'. */
  x_sort.resize(n);
  if (x_sort.size() == n) { /* Allocation successful. */
    lowerbound = 1.0 / n; /* Lower bound for the objective of the MP. */
    /* Reinitialize the objective of the MP. */
    for (j = 1; j <= d + 2; j++)
    for (rs[0][j] = 0, k = 1; k <= d + 1; k++) rs[0][j] += rs[k][j];
    /* Revised simplex algorithm */
    ItCount = 0;
    while (AddColumn(x)) 
		RSStep(FindPivotRow(), 0);
    if ( ItCount > MaxIt ) Error = 1;

//    free(x_sort);       /* Free the memory allocated for 'x_sort'. */
    MakeOriginal(x,z);  /* Reconstruct the original data. */
    return 1 / (n * rs[0][d + 2]); /* Return zonoid data depth. */
  }
  else { /* Memory for 'x_sort' could not be allocated. */
    Error = 2;
    MakeOriginal(x,z);  /* Reconstruct original data. */
    return -1;
  }
}

int InConvexes(TMatrix& points, TVariables& cardinalities, TMatrix& objects, int& Error, TIntMatrix *areInConvexes)
/*
   Check if the points are inside of the convex hull.
   1: Point lies inside of the convex hull of the data
   0: Point lies beyond the convex hull of the data
*/
{
	d = points[0].size();

	// Prepare a structure indicating if each point lies inside the convex hull of each class
	int m = objects.size();
	int q = cardinalities.size();
  
  areInConvexes->resize(m);
  for (int i = 0; i < m; i++){(*areInConvexes)[i].resize(q);}
  TIntMatrix &separateAnswers = (*areInConvexes);  // a link to output. just not to rewrite all occurances of separateAnswers  

	// Split into separate data sets and
	// check if each point lies inside each of the convex hulls
	int startIndex = 0;
	for (int i = 0; i < q; i++){ // Cycling through data sets
		n = cardinalities[i];
		TMatrix x(n);
		for (int j = 0; j < cardinalities[i]; j++){
			x[j] = points[startIndex + j];
		}
		/* Standardize */
		TPoint means(d);TPoint sds(d);
		GetMeansSds(x, &means, &sds);
		Standardize(x, means, sds);
		for (int j = 0; j < m; j++){ // Cycling through points
			int PivotColumn;
			TPoint z = objects[j];

			/* Standardize */
			Standardize(z, means, sds);			

			/* Rainer's codes (slightly modified) */
			Error = 0;
			
			MakeCanonical(x,z);  /* Convert tableau to canonical form. */
			
			/* Phase I */

			RSInit(z); /* Initialize tableau und basis variables. */
			/* Try to eliminate the artificial variables from the basis to get a
			basic feasible solution. */
			while (PhaseIGeneratePivotColumn(x, &PivotColumn))
				RSStep(FindPivotRow(), PivotColumn);
			/* If the maximum objective is greater than zero, no basic feasible
			solution exists. Thus, 'z' lies outside the convex hull of 'x'. */
			if (fabs(rs[0][d + 2]) > eps) {
				MakeOriginal(x,z); /* Reconstruct the original data. */
				Unstandardize(z, means, sds);
				separateAnswers[j][i] = 0; /* Point lies outside the convex hull. */
			}else{
				MakeOriginal(x,z); /* Reconstruct the original data. */
				Unstandardize(z, means, sds);
				separateAnswers[j][i] = 1; /* Point lies inside of the convex hull of the data. */
			}
		}
		startIndex+=cardinalities[i];
	}

	return 0;
}
