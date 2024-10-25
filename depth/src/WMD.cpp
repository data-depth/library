/******************************************************************************/
/* File:             WMD.cpp                                                  */
/* Created by:       Pavlo Mozharovskyi                                       */
/* Last revised:     23.06.2024                                               */
/*                                                                            */
/* Contains functions that compute the weighted-mean depths of a  point w.r.t */
/* a data cloud.                                                              */
/*                                                                            */
/******************************************************************************/

#include <algorithm>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include "DataStructures.h"
#include "Common.h"
#include "WMD.h"

#include <iostream>
#include <fstream>

using namespace std;

namespace DataDepth {

    double cGamma(double a, double b){
        if ((a >= 0) && (b >= 0) && (b < a + 1)){
            return tgamma(a + 1) / tgamma(b + 1) / tgamma(a - b + 1);
        }else{
            return 0;
        }
    }

    /**************************************************************************/
    /* 'calcSupFunCechStarD1' computes the value of the support function of   */
    /* the continuous expected convex hull depth.                             */
    /* The provided values should already be the projected data.              */
    /**************************************************************************/
    double calcSupFunCechStarD1(double alpha, const double* y, int n){
        double sum = 0;
        for (int j = 0; j < n; j++){
            sum += (pow((double)(j + 1), 1 / alpha) - pow((double)j, 1 / alpha)) * y[j];
        }
        sum /= pow((double)n, 1 / alpha);
        return sum;
    }

    double calcSupFunInfCechStarD1(double alpha, const double* y, int n){
        double sum = 0;
        for (int j = 0; j < n; j++){
            sum += (pow((double)(j + 1), 1 / alpha) - pow((double)j, 1 / alpha)) * y[n - j - 1];
        }
        sum /= pow((double)n, 1 / alpha);
        return sum;
    }

    double calcSupFunCechD1(double alpha, const double* y, int n){
        double sum = 0;
        for (int j = 0; j < n; j++){
            sum += (cGamma((double)(j + 1), 1 / alpha) - cGamma((double)j, 1 / alpha)) * y[j];
        }
        sum /= cGamma((double)n, 1 / alpha);
        return sum;
    }

    double calcSupFunInfCechD1(double alpha, const double* y, int n){
        double sum = 0;
        for (int j = 0; j < n; j++){
            sum += (cGamma((double)(j + 1), 1/alpha) - cGamma((double)j, 1 / alpha)) * y[n - j - 1];
        }
        sum /= cGamma((double)n, 1 / alpha);
        return sum;
    }

    double calcSupFunGeomD1(double alpha, const double* y, int n){
        if (alpha == 1){
            return 0;
        }
        double sum = 0;
        for (int j = 0; j < n; j++){
            sum += (1 - alpha) / (1 - pow(alpha, (double)n)) * pow(alpha, n - j - 1) * y[j];
        }
        return sum;
    }

    double calcSupFunInfGeomD1(double alpha, const double* y, int n){
        if (alpha == 1){
            return 0;
        }
        double sum = 0;
        for (int j = 0; j < n; j++){
            sum += (1 - alpha) / (1 - pow(alpha, (double)n)) * pow(alpha, n - j - 1) * y[n - j - 1];
        }
        return sum;
    }

    /**************************************************************************/
    /* CechStarD1 computes the continuous expected conves hull depth for      */
    /* univariate data.                                                       */
    /*                                                                        */
    /* Args:                                                                  */
    /*   z - the point for which to calculate the depth,                      */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of   */
    /*       dimension n)                                                     */
    /*   n - number of the data points.                                       */
    /* Returns:                                                               */
    /*   continuous expected conves hull (start) depth of z w.r.t. x.         */
    /**************************************************************************/

	double CechStarD1(double z, const double* x, int n) {
        // Prepare the sorted version of x
        double* xCopy = new double[n]; // Copy of "x" as "quick_sort" spoils it
        memcpy(xCopy, x, n * sizeof(double));
        quick_sort(xCopy, 0, n - 1);
        // The binary search algorithm
        double minAlpha = 0;
        double maxAlpha = 1;
        while (maxAlpha - minAlpha > eps){
            double curAlpha = (minAlpha + maxAlpha) / (double)2;
            double curPointU = calcSupFunCechStarD1(curAlpha, xCopy, n);
            double curPointD = calcSupFunInfCechStarD1(curAlpha, xCopy, n);
            if ((z < curPointU) && (z > curPointD)){
                minAlpha = curAlpha;
            }else{
                maxAlpha = curAlpha;
            }
        }
        delete[] xCopy;
        if (minAlpha == 0){
            return 0;
        }else{
            return (minAlpha + maxAlpha) / (double)2;
        }
	}

    /**************************************************************************/
    /* CechD1 computes the continuous expected conves hull depth for          */
    /* univariate data.                                                       */
    /*                                                                        */
    /* Args:                                                                  */
    /*   z - the point for which to calculate the depth,                      */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of   */
    /*       dimension n)                                                     */
    /*   n - number of the data points.                                       */
    /* Returns:                                                               */
    /*   continuous expected conves hull depth of z w.r.t. x.                 */
    /**************************************************************************/

    double CechD1(double z, const double* x, int n) {
        // Prepare the sorted version of x
        double* xCopy = new double[n]; // Copy of "x" as "quick_sort" spoils it
        memcpy(xCopy, x, n * sizeof(double));
        quick_sort(xCopy, 0, n - 1);
        // The binary search algorithm
        double minAlpha = 0;
        double maxAlpha = 1;
        while (maxAlpha - minAlpha > eps){
            double curAlpha = (minAlpha + maxAlpha) / (double)2;
            double curPointU = calcSupFunCechD1(curAlpha, xCopy, n);
            double curPointD = calcSupFunInfCechD1(curAlpha, xCopy, n);
            if ((z < curPointU) && (z > curPointD)){
                minAlpha = curAlpha;
            }else{
                maxAlpha = curAlpha;
            }
        }
        delete[] xCopy;
        if (minAlpha == 0){
            return 0;
        }else{
            return (minAlpha + maxAlpha) / (double)2;
        }
    }

    /****************************************************************************/
    /* GeomD1 computes the geometrical depth for                              */
    /*                                                                        */
    /* Args:                                                                  */
    /*   z - the point for which to calculate the depth,                      */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of   */
    /*       dimension n)                                                     */
    /*   n - number of the data points.                                       */
    /* Returns:                                                               */
    /*   continuous expected conves hull depth of z w.r.t. x.                 */
    /**************************************************************************/

    double GeomD1(double z, const double* x, int n) {
                // Prepare the sorted version of x
        double* xCopy = new double[n]; // Copy of "x" as "quick_sort(...)" spoils it
        memcpy(xCopy, x, n * sizeof(double));
        quick_sort(xCopy, 0, n - 1);
        // The binary search algorithm
        double minAlpha = 0;
        double maxAlpha = 1;
        while (maxAlpha - minAlpha > eps){
            double curAlpha = (minAlpha + maxAlpha) / (double)2;
            double curPointU = calcSupFunGeomD1(curAlpha, xCopy, n);
            double curPointD = calcSupFunInfGeomD1(curAlpha, xCopy, n);
            if ((z < curPointU) && (z > curPointD)){
                minAlpha = curAlpha;
            }else{
                maxAlpha = curAlpha;
            }
        }
        delete[] xCopy;
        if (minAlpha == 0){
            return 0;
        }else{
            return (minAlpha + maxAlpha) / (double)2;
        }
    }
}
