// Copyright (C) 2009 Cornell University
// All rights reserved.
// Original Author: Theodore Kim (http://www.cs.cornell.edu/~tedkim)

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
// MATRIX.h: interface for the MATRIX class.
//
//////////////////////////////////////////////////////////////////////

#ifndef MATRIX_H
#define MATRIX_H

#include "SETTINGS.h"
#include <map>
#include <iostream>
#include <cstdio>
#include "VECTOR.h"
#include "VEC3.h"
#include "MATRIX3.h"

using namespace std;

//////////////////////////////////////////////////////////////////////
// An arbitrary dimension matrix class
//////////////////////////////////////////////////////////////////////
class MATRIX {

public:
  MATRIX();
  MATRIX(int rows, int cols);
  MATRIX(int rows, int cols, Real* data);
  MATRIX(const char* filename);
  MATRIX(const MATRIX& m);
  // matrix with "vec" along the diagonal
  MATRIX(VECTOR& vec);

  // make a copy of a 3x3 matrix
  MATRIX(MATRIX3& matrix3);
  virtual ~MATRIX();

  inline Real& operator()(int row, int col) {
    return _matrix[row * _cols + col];
  };
 
  int& rows() { return _rows; };
  int& cols() { return _cols; };

  // return a pointer to the beginning of a row
  Real* row(int index) { return &_matrix[index * _cols]; };

  // wipe the whole matrix
  void clear();

  // write the matrix to a binary file
  // everything is always written as a double
  void write(const char* filename);

  // read from a binary file
  // everything is always read in as a double, then
  // converted if necessary
  void read(const char* filename);

  // resize the matrix and wipe to zero
  void resizeAndWipe(int rows, int cols);

  // overload operators
  MATRIX& operator=(const MATRIX m);
  MATRIX& operator-=(const MATRIX& m);
  MATRIX& operator+=(const MATRIX& m);
  MATRIX& operator*=(const Real& alpha);

  // return the matrix diagonal
  VECTOR diagonal();

  // return the transpose of the current matrix
  MATRIX transpose();

  // raw data pointer
  Real* data() { return _matrix; };

	// 
	Real sum()
	{
		Real sum = 0.0;
		for( int i = 0; i < _rows*_cols; i++ )
		{
			sum += _matrix[i];
		}
		return sum;
	}

  // stomp the current matrix with the given matrix 
  // starting at row number "row". It is your responsibility
  // to ensure that you don't fall off the end of this matrix.
  void setSubmatrix(MATRIX& matrix, int row);

  // Stomp the current matrix with the given vector,
  // starting at (row,col). So the first elemenet of the vector will
  // be copied into (row,col), the next into (row+1, col), etc.
  void setVector( VECTOR& vector, int row, int col );

  void copyRowFrom( MATRIX& src, int srcRow, int row );

  // BLAS axpy operation: B += alpha * A, where B is this matrix
  //
  // Note that axpy actually applies to vectors, but in this
  // case we can just treat the matrix as a vector and multiply
  // all its elements by alpha
  void axpy(Real alpha, MATRIX& A);

  // same as axpy above, but matrix contents are stomped as well
  void clearingAxpy(Real alpha, MATRIX& A);

  // BLAS gemm operation: C += alpha * A * B
  // where C is this matrix
  void gemm(Real alpha, MATRIX& A, MATRIX& B);
  void gemm(MATRIX& A, MATRIX& B) { gemm(1.0f, A, B); };

  // same as gemm above, but matrix contents are stomped as well
  void clearingGemm(Real alpha, MATRIX& A, MATRIX& B);
  void clearingGemm(MATRIX& A, MATRIX& B) { clearingGemm(1.0f, A, B); };

  // BLAS gemv operation: y = alpha * A * x
  // where A is this matrix
  VECTOR gemv(VEC3F& x);
  VECTOR gemv(Real alpha, VEC3F& x);
  //Untested -- don't uncomment until a test case comes up
  //VECTOR gemv(VECTOR& x);
  
  // solve the linear system Ax = b, return x in the passed in b
  void solve(VECTOR& b);

  // multiply in place
  // this * x = y
  // Do *NOT* let x == y!
  void multiplyInplace(VECTOR& x, VECTOR& y);

  void subMatrixMultiplyInplace( VECTOR& x, VECTOR& prod, int subRows, int subCols, bool transpose );

  // Assumes matrix is symmetric
  void uppertriMultiplyInplace( VECTOR& x, VECTOR& prod );

  // solve for eigenvalues
  void eigensystem(VECTOR& eigenvalues, MATRIX& eigenvectors);
 
  // copy this matrix to MATRIX3 type
  void copiesInto(MATRIX3& matrix3);

	// Returns the Frobenius-norm of the difference between this and B
	Real differenceFrobeniusSq( MATRIX& B );

protected:
  int _rows;
  int _cols;

  Real* _matrix;
};

// overloaded operators
VECTOR operator*(MATRIX& A, VECTOR& x);
MATRIX operator*(MATRIX& A, Real alpha);
MATRIX operator*(MATRIX& A, MATRIX& B);
ostream& operator<<(ostream &out, MATRIX& matrix);

// multiply by the transpose of A
VECTOR operator^(MATRIX& A, VECTOR& x);
MATRIX operator^(MATRIX& A, MATRIX& B);

#endif
