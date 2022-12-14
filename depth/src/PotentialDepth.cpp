#include "stdafx.h"

double EuclidianDistance(TPoint& x, TPoint& y){
	double accu = 0;
	for (int i = 0; i < x.size(); i++){
		accu += pow(x[i] - y[i], 2);
	}
	return sqrt(accu);
}

double EuclidianDistance2(TPoint& x, TPoint& y){
  double accu = 0;
	for (int i = 0; i< x.size(); i++){
		accu += pow(x[i] - y[i], 2);
	}
	return (accu);
}

// alpha - kernel sharpness. sharp - a more
double EDKernel (TPoint& x, TPoint& y, double a){
	return 1/(1+a*EuclidianDistance2(x, y));
}

// ss - sigma squared. sharp - a less
double GKernel (TPoint& x, TPoint& y, double ss){
  int d = x.size();
	return pow((2*PI*ss), - d/2) * exp(-EuclidianDistance2(x, y) / (2*ss));   //04.04.2014  added power d
}

// ss - sigma squared. sharp - a less
double VarGKernel(TPoint& x, TPoint& y, double ss){
	int d = x.size();
	return pow((2 * PI*ss), -d / 2) * exp(-EuclidianDistance2(x, y) / (2 * ss));   //04.04.2014  added power d
}

// alpha - kernel sharpness. sharp - a more
double EKernel (TPoint& x, TPoint& y, double a){
	return exp(-a*EuclidianDistance(x, y));
}

// alpha - triangle sharpness. sharp - a more. a in (0..pi/2)
double TriangleKernel (TPoint& x, TPoint& y, double a){
	return 1/(EuclidianDistance(x, y)+0.000001)*tan(a);
}

void PotentialDepths(TMatrix& points, TVariables& cardinalities, /*OUT*/ TMatrix& depths, double (*Kernel) (TPoint& x, TPoint& y, double a), double a){
	PotentialDepths(points, cardinalities, points, depths, Kernel, a, 0);
}

void PotentialDepths(TMatrix& points, TVariables& cardinalities, TMatrix& testpoints, /*OUT*/ TMatrix& depths, double (*Kernel) (TPoint& x, TPoint& y, double ss), double ss, int ignoreself){
	int classBeginning = 0;
	
	bool var = Kernel == VarGKernel;
	double weight = 1;
	TMatrix* classPoints;
	TPoint* classPointsDepths;
	int error;
	
	// loop classes
	for (int i = 0; i < cardinalities.size(); classBeginning += cardinalities[i], i++){

		if (var){
			if (classPoints) delete[] classPoints;
			classPoints = new TMatrix(points.begin() + classBeginning, points.begin() + classBeginning + cardinalities[i]);
			if (!classPointsDepths)
				classPointsDepths = new TPoint(cardinalities[i]);
			else if (classPointsDepths->size() < cardinalities[i])
				classPointsDepths->resize(cardinalities[i]);

			for (int c = 0; c < cardinalities[i]; c++){
				(*classPointsDepths)[c] = 1 - ZonoidDepth(*classPoints, points[classBeginning + c], error);
			}
		}

		// loop all the points, find their potential relatively to class i
		for (int p = 0; p < testpoints.size(); p++){
			double pointDepth = 0;

			// loop the points of i-th class, find the point's potential
			for (int c = 0; c < cardinalities[i]; c++){
				//      if (ignoreself && p == classBeginning + c) // ignore the potential created by this point
				//        continue;

				if (var)
					weight = (*classPointsDepths)[c];

				pointDepth += Kernel(testpoints[p], points[classBeginning + c], ss*weight);
			}
			depths[p][i] = pointDepth;
		}
    
		if (false) {  //28.05.2014 no normalization
			int n = ignoreself ? points.size() - 1 : points.size();

			// normalize
			for (int p = 0; p < testpoints.size(); p++){
				depths[p][i] *= 1.0 / n;  //04.04.2014   1.0*cardinalities[i]/points.size()/Kernel(points[0], points[0], a);
			}
		}
	}

	if (var){
		delete[] classPoints;
		delete[] classPointsDepths;
	}
}

