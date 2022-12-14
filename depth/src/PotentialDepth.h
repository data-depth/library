#ifndef PotentialDepth_h
#define PotentialDepth_h

// Kernel constant: 1
// alpha - kernel sharpness. sharp - a more
double EDKernel (TPoint& x, TPoint& y, double a);

// Kernel constant: 2
// ss - sigma squared. sharp - a less
double GKernel (TPoint& x, TPoint& y, double ss);

// Kernel constant: 5
// ss - sigma squared. sharp - a less
double VarGKernel(TPoint& x, TPoint& y, double ss);

// Kernel constant: 3
// alpha - kernel sharpness. sharp - a more
double EKernel (TPoint& x, TPoint& y, double a);

// Kernel constant: 4
// alpha - triangle sharpness. sharp - a more. a in (0..pi/2)
double TriangleKernel (TPoint& x, TPoint& y, double a);

void PotentialDepths(TMatrix& points, TVariables& cardinalities, /*OUT*/ TMatrix& depths, double (*Kernel) (TPoint& x, TPoint& y, double a), double a);

void PotentialDepths(TMatrix& points, TVariables& cardinalities, TMatrix& testpoints, /*OUT*/ TMatrix& depths, double (*Kernel) (TPoint& x, TPoint& y, double a), double a, int ignoreself);

#endif
