/*
  File:             DataStructures.cpp
  Created by:       Pavlo Mozharovskyi
  First published:  28.02.2013
  Last revised:     28.02.2013

  Defines the data structures used in the package.
*/
#pragma once

#ifndef PI2
#define PI2 1.5707963267948966192313216916398
#endif
#ifndef PI
#define PI (PI2*2)
#endif

//typedef vector<vector<int> > NumericMatrix;

typedef vector<double> TPoint;
typedef vector<vector<double> > TMatrix;
typedef vector<vector<int> > TIntMatrix;
typedef vector<int> TVariables;

typedef double** TDMatrix;
typedef double*** T3DMatrix;

// by rows
TDMatrix asMatrix(double* arr, int n, int d);
T3DMatrix as3DMatrix(double* arr, int n, int t, int d);

double** newM(int n, int d);
void deleteM(TDMatrix X);
TDMatrix copyM(TDMatrix X, int n, int d);
void printMatrix(TDMatrix mat, int n, int d);


struct UPoint{
	 int pattern;
	 double value;
	 UPoint(int pattern = 0, double value = 0){
		 this->pattern = pattern;
		 this->value = value;
	 }
 };

 struct Feature{
	 unsigned int order;
	 int number;
	 double angle;
	 unsigned int error;
	 Feature(unsigned int order = 0, int number = 0, double angle = 0, unsigned int error = INT_MAX){
		 this->order = order;
		 this->number = number;
		 this->angle = angle;
		 this->error = error;
	 }
 };

struct OrderRec {
	int order;
	double value;
	OrderRec(int order = -1, double value = 0) {
		this->order = order;
		this->value = value;
	}
};

struct SortRec { 
	double v; 
	TPoint* p; 
	SortRec(double v = 0, TPoint* p = NULL) {
		this->v = v;
		this->p = p;
	}
};

struct SortRecClassified {
	int c; 
	double v; 
	SortRecClassified(int c = 0, double v = 0, int p = 0) {
		this->c = c;
		this->v = v;
	}
};

typedef vector<Feature> Features;
typedef vector<UPoint> UPoints;
