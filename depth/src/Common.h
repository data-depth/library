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
void GetProjections(TDMatrix points, int n, int d, TDMatrix directions, int k, 
                    TDMatrix projections);

template<typename T>
int QuickCompareAcs(T &p1, T &p2) {
  return p1 < p2;
}

template<typename T>
int QuickCompareDes(T &p1, T &p2) {
  return p1 > p2;
}

template<typename T>
void QuickSwap(T *p1, T *p2) {
  T pTmp = *p1;
  *p1 = *p2;
  *p2 = pTmp;
};

/* -------------------------------------------------------------------------- */
/* quickSort from http://www.proggen.org/doku.php?id=algo:quicksort           */
/* (modified, templated)                                                      */
/* -------------------------------------------------------------------------- */
template<typename T>
void quick_sort(T *values, int left, int right, int(*cmp)(T& x, T& y),
                void(*swap)(T* x, T* y)){
  int i = left, j = right;
  T pivot = values[(left + right) >> 1];
  do{
    while (cmp(values[i], pivot)){++i;}
    while (cmp(pivot, values[j])){--j;}
    if (i < j){swap(&values[i], &values[j]);++i;--j;
    }else{if (i == j){++i;--j;break;}}
  }while (i <= j);
  if (left < j){quick_sort(values, left, j, cmp, swap);}
  if (i < right){quick_sort(values, i, right, cmp, swap);}
}
template<typename T> void quick_sort(T *values, int left, int right){
  quick_sort(values, left, right, QuickCompareAcs, QuickSwap);
}
template<typename T> void quick_sort_rev(T *values, int left, int right){
  quick_sort(values, left, right, QuickCompareDes, QuickSwap);
}
