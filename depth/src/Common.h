/*
  File:             Common.h
  Created by:       Pavlo Mozharovskyi
  First published:  17.05.2013
  Last revised:     13.11.2015
  
  Commonly used functions.
*/

#pragma once

const double eps_pivot = 1e-10;

//#define DEF_OUT_ALPHA
extern bool OUT_ALPHA;
#ifndef _MSC_VER
#define DEF_OUT_ALPHA
#endif


void outString(char const * str);
//template<typename T>
void outVector(TPoint& point);
void outMatrix(TMatrix& points);
void outFeatures(Features fs);

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

unsigned long long choose(unsigned long long n, unsigned long long k);
unsigned long long fact(unsigned long long n);
bool solveUnique(TDMatrix A, double* b, double* x, int d);
double getDet(TDMatrix A, int d);

double determinant(TDMatrix& m);
double* means(TDMatrix X, int n, int d);
TDMatrix cov(TDMatrix X, int n, int d);

void GetDirections(TDMatrix directions, int k, int d);
void GetProjections(TDMatrix points, int n, int d, TDMatrix directions, int k, TDMatrix projections);
