/******************************************************************************/
/* File:             Matrix.h                                                 */
/* Created by:       Rainer Dyckerhoff                                        */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* A simple matrix class that implements basic operations for matrices        */
/*                                                                            */
/* Defines a class 'cMatrix' that represents matrices as well as a class      */
/* 'cVector' that is derived from 'cMatrix' and represents vectors            */
/*                                                                            */
/* The components of these matrices and vectors are not templated and are     */
/* always doubles.                                                            */
/*                                                                            */
/******************************************************************************/

#ifndef DYCKERHOFF_MATRIX_H
#define DYCKERHOFF_MATRIX_H

#include <cassert>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <complex>
#include <memory>
#include <functional>


namespace dyMatrixClass {
	std::ostream& shortMat(std::ostream & os);
	std::ostream& longMat(std::ostream & os);

	enum class matrix_direction { mLeft = 1, mRight = 2, mLeftRight = 3};

	class cMatrix {
	protected:
		int m, n;
		std::unique_ptr<double[]> elems;
	public:
		cMatrix() = default;
		cMatrix(cMatrix&& m);															// Move constructor
		cMatrix& operator=(cMatrix&& m);												// Move assignment
		cMatrix(const cMatrix&);														// Copy constructor
		cMatrix& operator=(const cMatrix&);												// Copy assignment

		cMatrix(int m, int n)															// Constructor
			: m{ m }, n{ n }, elems{ new double[m * n]{} } {};
		cMatrix(std::initializer_list<std::initializer_list<double>> args);				// List initializer constructor
		cMatrix(std::initializer_list<double> args);									// List initializer constructor (column vector)

		void SetSize(int _m, int _n);
		long index(int i, int j) const { return i * n + j; }
		double& operator()(int i, int j) const { return elems[i * n + j]; }
		double& operator()(int i) const { return elems[i]; }
		double* operator[](int i) const { return elems.get() + i * n; }

		double* data() { return elems.get(); }
		const double* data() const { return elems.get(); }
		double* begin() { return elems.get(); }
		const double* begin() const { return elems.get(); }
		double* end() { return elems.get() + m * n; }
		const double* end() const { return elems.get() + m * n; }

		int Rows() const { return m; }
		int Cols() const { return n; }

		cMatrix Row(int i) const;
		cMatrix Col(int j) const;
		cMatrix Diag() const;
		cMatrix SubMatrix(int i1, int i2, int j1, int j2) const;

		operator double() const;

		void XChangeRows(int i1, int i2) { std::swap_ranges(elems.get() + i1*n, elems.get() + (i1 + 1)*n, elems.get() + i2*n); }
		void XChangeCols(int j1, int j2) { for (int i = 0; i < m; i++) std::swap(elems[i*n + j1], elems[i*n + j2]); }

		cMatrix& apply(std::function<double(double)> f);

		cMatrix& operator*=(double c);
		cMatrix& operator/=(double c);

		cMatrix& operator+=(const cMatrix& B);
		cMatrix& operator-=(const cMatrix& B);
		cMatrix& operator*=(const cMatrix& B);

		cMatrix operator-();

		friend cMatrix operator+(const cMatrix& A, const cMatrix& B);
		friend cMatrix operator-(const cMatrix& A, const cMatrix& B);
		friend cMatrix operator*(const cMatrix& A, const cMatrix& B);

		friend bool operator==(const cMatrix& A, const cMatrix& B);
		friend bool operator!=(const cMatrix& A, const cMatrix& B);
		friend bool operator<=(const cMatrix& A, const cMatrix& B);
		friend bool operator>=(const cMatrix& A, const cMatrix& B);
		friend bool operator< (const cMatrix& A, const cMatrix& B);
		friend bool operator> (const cMatrix& A, const cMatrix& B);

		friend std::ostream& operator<< (std::ostream& os, const cMatrix& A);
		friend cMatrix Trans(const cMatrix& A);

		friend cMatrix operator*(const cMatrix& A, double c);
		friend cMatrix operator*(double c, const cMatrix& A);
		friend cMatrix operator/(const cMatrix& A, double c);

		cMatrix& SetEpsToZero(double eps);
		cMatrix& SetDiag(double D[]) { for (int i = 0; i < std::min(m, n); i++) elems[index(i, i)] = D[i]; return *this; };
	};

	class cVector : public cMatrix {
	public:
		cVector() = default;

		cVector& operator=(cMatrix&& m);												// Move assignment
		cVector& operator=(const cMatrix&);												// Copy assignment

		cVector(int n) : cMatrix(n, 1) {};												// Constructor
			

		cVector(std::initializer_list<double> args);									// List initializer constructor (column vector)

		double& operator[](int i) const { return elems[i]; }
		void SetSize(int _n) { cMatrix::SetSize(1, _n); }
	};

}

#endif
