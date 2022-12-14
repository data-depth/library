/******************************************************************************/
/* File:             Matrix.cpp                                               */
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

#include <memory>
#include <numeric>
#include "Matrix.h"

namespace dyMatrixClass {


	using namespace std;

	//********** Matrix format flag handling *****************************************************

	enum class matrix_format_flags { fShort, fLong };

	long& matrix_format_flag(ios_base& s) {
		static int my_index = ios_base::xalloc();
		return s.iword(my_index);
	}

	long get_matrix_format_flag(ios_base& s) {
		return matrix_format_flag(s);
	}

	void set_matrix_format_flag(ios_base& s, long n) {
		matrix_format_flag(s) = n;
	}

	static void set_matrix_format(ios_base& s, matrix_format_flags mat_fmt)	{
		matrix_format_flag(s) = (long)mat_fmt;
	}

	ostream& shortMat(ostream & os) {
		set_matrix_format(os, matrix_format_flags::fShort);
		return os;
	}

	ostream& longMat(ostream & os) {
		set_matrix_format(os, matrix_format_flags::fLong);
		return os;
	}


	//********** Matrix class ********************************************************************

#undef _MESSAGES_ 

	cMatrix::cMatrix(cMatrix&& A)											// Move constructor 
		: m{ A.m }, n{ A.n }, elems{ move(A.elems) } {
		//cout << "Matrix move constructor" << endl;
	};

	cMatrix& cMatrix::operator=(cMatrix&& A) {								// Move assignment
		m = A.m;
		n = A.n;
		elems = move(A.elems);
		//cout << "Matrix move assignment" << endl;
		return *this;
	};

	cMatrix::cMatrix(const cMatrix& A)										// Copy constructor
		: m{ A.m }, n{ A.n }, elems{ new double[A.m*A.n] } {
		std::uninitialized_copy_n(A.elems.get(), m*n, elems.get());
		//cout << "Matrix copy constructor" << endl;
	}

	cMatrix& cMatrix::operator=(const cMatrix& A) {							// Copy assignment
		if (this != &A) {
			m = A.m;
			n = A.n;
			elems = unique_ptr<double[]>(new double[m*n]);
			std::uninitialized_copy_n(A.elems.get(), m*n, elems.get());
		}
		//cout << "Matrix copy assignment" << endl;
		return *this;
	}

	cMatrix::cMatrix(std::initializer_list<std::initializer_list<double>> args) {					// List initializer constructor
		m = args.size();
		n = 0;
		for (int i = 0; i < args.size(); i++) n = std::max(n, (int)args.begin()[i].size());
		elems = unique_ptr<double[]>(new double[m*n]{});
		for (int i = 0; i < args.size(); i++) std::uninitialized_copy(args.begin()[i].begin(), args.begin()[i].end(), elems.get() + i*n);
		//cout << "Matrix 2-dim list constructor" << endl;
	}

	cMatrix::cMatrix(std::initializer_list<double> args) {										// List initializer constructor (column vector)
		m = args.size();
		n = 1;
		elems = unique_ptr<double[]>(new double[m]{});
		std::uninitialized_copy(args.begin(), args.end(), elems.get());
		//cout << "Matrix 1-dim list constructor" << endl;
	}

	void cMatrix::SetSize(int _m, int _n) {
		m = _m;
		n = _n;
		elems = unique_ptr<double[]>(new double[m*n]{});
	};

	cMatrix cMatrix::Row(int i) const {
		cMatrix x(1, n);
		uninitialized_copy_n(elems.get() + i*n, n, x.elems.get());
		return x;
	}

	cMatrix cMatrix::Col(int j) const {
		cMatrix x(m, 1);
		for (int i = 0; i < m; i++) x(i, 0) = elems[i*n + j];
		return x;
	}

	cMatrix cMatrix::Diag() const {
		cMatrix x(min(m,n), 1);
		for (int i = 0; i < min(m,n); i++) x(i, 0) = elems[i*n + i];
		return x;
	}

	cMatrix cMatrix::SubMatrix(int i1, int i2, int j1, int j2) const {
		cMatrix A(i2 - i1 + 1, j2 - j1 + 1);
		for (int i = i1; i <= i2; i++)
			for (int j = j1; j <= j2; j++)
				A(i - i1, j - j1) = elems[i*n + j];
		return A;
	}


	cMatrix& cMatrix::apply(std::function<double(double)> f) {
		for (int i = 0; i < m*n; i++) elems[i] = f(elems[i]);
		return *this;
	}

	cMatrix& cMatrix::operator*=(double c) {
		for (int i = 0; i < m*n; i++) elems[i] *= c;
		return *this;
	}

	cMatrix& cMatrix::operator/=(double c) {
		for (int i = 0; i < m*n; i++) elems[i] /= c;
		return *this;
	}

	cMatrix& cMatrix::operator+=(const cMatrix& B) {
		assert((m == B.m) && (n == B.n));
		for (int i = 0; i < m*n; i++) elems[i] += B.elems[i];
		return *this;
	}

	cMatrix& cMatrix::operator-=(const cMatrix& B) {
		assert((m == B.m) && (n == B.n));
		for (int i = 0; i < m*n; i++) elems[i] -= B.elems[i];
		return *this;
	}

	cMatrix& cMatrix::operator*=(const cMatrix& B) {
		assert((n == B.m));
		cMatrix& A{ *this };
		cMatrix tmp(m, B.n);
		for (int i = 0; i < tmp.m; i++) {
			for (int j = 0; j < tmp.n; j++) {
				for (int k = 0; k < n; k++) tmp(i, j) += A(i, k) * B(k, j);
			}
		}
		return *this = std::move(tmp);
	}

	cMatrix cMatrix::operator-() {
		cMatrix B(m, n);
		for (int i = 0; i < m*n; i++) B.elems[i] = -elems[i];
		return B;
	}

	cMatrix operator*(const cMatrix& A, double c) {
		cMatrix res(A.m, A.n);
		for (int i = 0; i < res.m * res.n; i++) res.elems[i] = c * A.elems[i];
		return res;
	}

	cMatrix operator*(double c, const cMatrix& A) {
		cMatrix res(A.m, A.n);
		for (int i = 0; i < res.m * res.n; i++) res.elems[i] = c*A.elems[i];
		return res;
	}

	cMatrix operator/(const cMatrix& A, double c) {
		cMatrix res(A.m, A.n);
		for (int i = 0; i < res.m * res.n; i++) res.elems[i] = A.elems[i] / c;
		return res;
	}

	cMatrix operator+(const cMatrix& A, const cMatrix& B) {
		assert((A.m == B.m) && (A.n == B.n));
		cMatrix res(A.m, A.n);
		for (int i = 0; i < res.m * res.n; i++) res.elems[i] = A.elems[i] + B.elems[i];
		return res;
	}

	cMatrix operator-(const cMatrix& A, const cMatrix& B) {
		assert((A.m == B.m) && (A.n == B.n));
		cMatrix res(A.m, A.n);
		for (int i = 0; i < res.m * res.n; i++) res.elems[i] = A.elems[i] - B.elems[i];
		return res;
	}

	cMatrix operator*(const cMatrix& A, const cMatrix& B) {
		assert((A.n == B.m));
		cMatrix res(A.m, B.n);
		for (int i = 0; i < res.m; i++)
			for (int j = 0; j < res.n; j++)
				for (int k = 0; k < A.n; k++)
					res(i, j) += A(i, k) * B(k, j);
		return res;
	}

	bool operator==(const cMatrix& A, const cMatrix& B) {
		if ((A.m != B.m) || (A.n != B.n)) return false;
		for (int i = 0; i < A.m * A.n; i++)
			if (A.elems[i] != B.elems[i]) return false;
		return true;
	}

	bool operator!=(const cMatrix& A, const cMatrix& B) {
		if ((A.m != B.m) || (A.n != B.n)) return true;
		for (int i = 0; i < A.m * A.n; i++)
			if (A.elems[i] != B.elems[i]) return true;
		return false;
	}

	bool operator<=(const cMatrix& A, const cMatrix& B) {
		assert((A.m == B.m) && (A.n == B.n));
		for (int i = 0; i < A.m * A.n; i++)
			if (A.elems[i] > B.elems[i]) return false;
		return true;
	}

	bool operator>=(const cMatrix& A, const cMatrix& B) {
		assert((A.m == B.m) && (A.n == B.n));
		for (int i = 0; i < A.m * A.n; i++)
			if (A.elems[i] < B.elems[i]) return false;
		return true;
	}

	bool operator<(const cMatrix& A, const cMatrix& B) {
		assert((A.m == B.m) && (A.n == B.n));
		for (int i = 0; i < A.m * A.n; i++)
			if (A.elems[i] >= B.elems[i]) return false;
		return true;
	}

	bool operator>(const cMatrix& A, const cMatrix& B) {
		assert((A.m == B.m) && (A.n == B.n));
		for (int i = 0; i < A.m * A.n; i++)
			if (A.elems[i] <= B.elems[i]) return false;
		return true;
	}

	cMatrix::operator double() const {
		assert((m == 1) && (n == 1));
		return elems[0];
	}

	cMatrix Trans(const cMatrix& A) {
		cMatrix res(A.n, A.m);
		for (int i = 0; i < res.m; i++)
			for (int j = 0; j < res.n; j++) res(i, j) = A(j, i);
		return res;
	}

	std::ostream& operator<<(std::ostream& os, const cMatrix& A) {
		if (get_matrix_format_flag(os) == (long)matrix_format_flags::fShort) {
			os << "{";
			for (int i = 0; i < A.m; i++) {
				os << "{";
				for (int j = 0; j < A.n; j++) {
					std::cout << ((fabs(A(i,j)) < 1e-14) ? 0 : A(i, j));
					if (j < A.n - 1) std::cout << ",";
					else std::cout << "}";
				}
				if (i < A.m - 1) std::cout << ",";
			}
			os << "}";
		}
		else {
			for (int i = 0; i < A.m; i++) {
				os << "[";
				for (int j = 0; j < A.n; j++) {
					std::cout << std::setw(10) << ((fabs(A(i, j)) < 1e-14) ? 0 : A(i, j));
					if (j < A.n - 1) std::cout << ",";
				}
				os << "]" << std::endl;
			}
		}
		return os;
	}

	//***************** Vector *******************************************************

	cVector& cVector::operator=(cMatrix&& A) {								// Move assignment
		cMatrix::operator=(move(A));
		return *this;
	};

	cVector& cVector::operator=(const cMatrix& A) {							// Copy assignment
		if (this != &A) {
			cMatrix::operator=(A);
		}
		return *this;
	}

	cVector::cVector(std::initializer_list<double> args) {					// List initializer constructor (column vector)
		m = args.size();
		n = 1;
		elems = unique_ptr<double[]>(new double[m]{});
		std::uninitialized_copy(args.begin(), args.end(), elems.get());
		cout << "Vector list constructor" << endl;
	}

	cMatrix& cMatrix::SetEpsToZero(double eps) {
		for (int i = 0; i < m*n; i++)
			if (fabs(elems[i]) < eps) elems[i] = 0;
		return *this;
	}

}
