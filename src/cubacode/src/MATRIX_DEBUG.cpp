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
// MATRIX_DEBUG.cpp: 
//
// This is a *slow* implementation of the MATRIX.h header. It does not
// use ATLAS, LAPACK, BLAS, or MKL to facilitate debugging. As it also
// does not need to link to those libraries, it is also self-contained
// and much easier to compile.
//
// Note that the "solve" function is *not* implemented however, since
// I can't seem to find slow, simple, reliable LU code anywhere.
//
// Support for tet inversion makes the inclusion of PetSc's eigenvalue
// solver necessary unfortunately, which affects the self-containment
// of the code. Suggestions for alternatives are welcome.
//
//////////////////////////////////////////////////////////////////////

#include "MATRIX.h"
#include <stdio.h>
#include <string.h>

#ifdef USE_OMP
#include <omp.h>
#endif

//////////////////////////////////////////////////////////////////////
// Constructor for the full matrix
//////////////////////////////////////////////////////////////////////
MATRIX::MATRIX() :
  _rows(0), _cols(0)
{
  _matrix = NULL;
}

MATRIX::MATRIX(int rows, int cols) :
  _rows(rows), _cols(cols)
{
	try {
		_matrix = new Real[_rows * _cols];
	}
	catch(...) {
		cerr << "[ WARNING ] Could not allocate matrix of " << (_rows*_cols*sizeof(Real)/1024.0/1024.0) << " MB. It will core-dump if your code does not check for failure!" << endl;
		_matrix = NULL;
	}

	if( _matrix != NULL ) {
		clear();
	}
}

MATRIX::MATRIX(int rows, int cols, Real* data) :
  _rows(rows), _cols(cols)
{
  _matrix = new Real[_rows * _cols];
  for (int x = 0; x < _rows * _cols; x++)
    _matrix[x] = data[x];
}

MATRIX::MATRIX(const char* filename)
{
	_matrix = NULL;
  read(filename);
}

MATRIX::MATRIX(const MATRIX& m)
{
  _cols = m._cols;
  _rows = m._rows;

  _matrix = new Real[_rows * _cols];
  for (int x = 0; x < _rows * _cols; x++)
    _matrix[x] = m._matrix[x];
}

MATRIX::MATRIX(VECTOR& vec)
{
  _cols = vec.size();
  _rows = vec.size();

  _matrix = new Real[_rows * _cols];
  clear();

  for (int x = 0; x < vec.size(); x++)
    (*this)(x,x) = vec(x);
}

MATRIX::MATRIX(MATRIX3& matrix3)
{
  _cols = 3;
  _rows = 3;

  _matrix = new Real[9];
  clear();

  for (int y = 0; y < 3; y++)
    for (int x = 0; x < 3; x++)
      (*this)(x,y) = matrix3(x,y);
}

MATRIX::~MATRIX()
{
  delete[] _matrix;
}

//////////////////////////////////////////////////////////////////////
// wipe the whole matrix
//////////////////////////////////////////////////////////////////////
void MATRIX::clear()
{
  memset(_matrix, 0, _rows * _cols * sizeof(Real));
}

//////////////////////////////////////////////////////////////////////
// resize and wipe the matrix
//////////////////////////////////////////////////////////////////////
void MATRIX::resizeAndWipe(int rows, int cols)
{
  if (_rows != rows || _cols != cols)
    delete[] _matrix;
  _rows = rows;
  _cols = cols;

  _matrix = new Real[_rows * _cols];
  clear();
}

//////////////////////////////////////////////////////////////////////
// write the matrix to a file
//////////////////////////////////////////////////////////////////////
void MATRIX::write(const char* filename)
{
  FILE* file;
  file = fopen(filename, "wb");

  // write dimensions
  fwrite((void*)&_rows, sizeof(int), 1, file);
  fwrite((void*)&_cols, sizeof(int), 1, file);

  // always write out as a double
  if (sizeof(Real) != sizeof(double))
  {
    double* matrixDouble = new double[_rows * _cols];
    for (int x = 0; x < _rows * _cols; x++)
      matrixDouble[x] = _matrix[x];

    fwrite((void*)matrixDouble, sizeof(double), _rows * _cols, file);
    delete[] matrixDouble;
    fclose(file);
  }
  else
    fwrite((void*)_matrix, sizeof(Real), _rows * _cols, file);
  fclose(file);
}

//////////////////////////////////////////////////////////////////////
// read matrix from a file
//////////////////////////////////////////////////////////////////////
void MATRIX::read(const char* filename)
{
  FILE* file;
  file = fopen(filename, "rb");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __LINE__ << " : File " << filename << " not found! " << endl;
    return;
  }

  // read dimensions
  fread((void*)&_rows, sizeof(int), 1, file);
  fread((void*)&_cols, sizeof(int), 1, file);

  // read data
  if( _matrix != NULL ) delete[] _matrix;
  _matrix = new Real[_rows * _cols];

  // always read in a double
  if (sizeof(Real) != sizeof(double))
  {
    double* matrixDouble = new double[_rows * _cols];
    fread((void*)matrixDouble, sizeof(double), _rows * _cols, file);
    for (int x = 0; x < _rows * _cols; x++)
      _matrix[x] = matrixDouble[x];
    delete[] matrixDouble;
  }
  else 
    fread((void*)_matrix, sizeof(Real), _rows * _cols, file);
  fclose(file);
}

//////////////////////////////////////////////////////////////////////
// return transpose of current matrix
//////////////////////////////////////////////////////////////////////
MATRIX MATRIX::transpose()
{
  MATRIX toReturn(_cols, _rows);

  for (int y = 0; y < _cols; y++)
    for (int x = 0; x < _rows; x++)
      toReturn(y,x) = (*this)(x,y);

  return toReturn;
}

//////////////////////////////////////////////////////////////////////
// Matrix-vector multiply
//////////////////////////////////////////////////////////////////////
VECTOR operator*(MATRIX& A, VECTOR& x) 
{
  VECTOR y(A.rows());

#if BOUNDS_CHECKING_ENABLED
  if (A.cols() != x.size())
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Vector dim mismatch! " << endl;
#endif

  for (int i = 0; i < A.rows(); i++)
    for (int j = 0; j < A.cols(); j++)
      y(i) += x(j) * A(i, j);

  return y;
}

//////////////////////////////////////////////////////////////////////
// Matrix-vector multiply
//////////////////////////////////////////////////////////////////////
MATRIX operator*(MATRIX& A, Real alpha) 
{
  MATRIX y(A.rows(), A.cols());

  for (int i = 0; i < A.rows(); i++)
    for (int j = 0; j < A.cols(); j++)
      y(i,j) = A(i, j) * alpha;

  return y;
}

//////////////////////////////////////////////////////////////////////
// Matrix-Matrix multiply
//////////////////////////////////////////////////////////////////////
MATRIX operator*(MATRIX& A, MATRIX& B) 
{
  MATRIX y(A.rows(), B.cols());

#if BOUNDS_CHECKING_ENABLED
  if (A.cols() != B.rows())
  {
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Matrix dimensions do not match! " << endl;
    return y;
  }
#endif

  for (int i = 0; i < A.rows(); i++)
    for (int j = 0; j < B.cols(); j++)
      for (int k = 0; k < A.cols(); k++)
        y(i,j) += A(i, k) * B(k, j);

  return y;
}

//////////////////////////////////////////////////////////////////////
// scale vector by a constant
//////////////////////////////////////////////////////////////////////
MATRIX& MATRIX::operator*=(const Real& alpha)
{
  for (int x = 0; x < _cols * _rows; x++)
    _matrix[x] *= alpha;
  
  return *this;
}

//////////////////////////////////////////////////////////////////////
// Matrix-vector multiply where A is transposed
//////////////////////////////////////////////////////////////////////
VECTOR operator^(MATRIX& A, VECTOR& x) 
{
  VECTOR y(A.cols());

#if BOUNDS_CHECKING_ENABLED
  if (A.rows() != x.size())
  {
    cout << __FILE__ << " " << __LINE__ << " : Transposed Matrix-Vector dim mismatch! " << endl;
    cout << "Matrix: " << A.rows() << " " << A.cols() << endl;
    cout << "Vector: " << x.size() << endl;
  }
#endif

  for (int i = 0; i < A.rows(); i++)
    for (int j = 0; j < A.cols(); j++)
      y(j) += x(i) * A(i, j);

  return y;
}

//////////////////////////////////////////////////////////////////////
// Matrix^T -Matrix multiply
//////////////////////////////////////////////////////////////////////
MATRIX operator^(MATRIX& A, MATRIX& B) 
{
  MATRIX y(A.cols(), B.cols());

#if BOUNDS_CHECKING_ENABLED
  if (A.rows() != B.rows())
  {
    cout << __FILE__ << " " << __LINE__ << " : Transposed Matrix-Matrix dimensions do not match! " << endl;
    return y;
  }
#endif

  for (int i = 0; i < A.cols(); i++)
    for (int j = 0; j < B.cols(); j++)
      for (int k = 0; k < A.rows(); k++)
        y(i,j) += A(k, i) * B(k, j);

  return y;
}

//////////////////////////////////////////////////////////////////////
// Print matrix to stream
//////////////////////////////////////////////////////////////////////
ostream& operator<<(ostream &out, MATRIX& matrix)
{
  out << "[" << endl; 
  for (int row = 0; row < matrix.rows(); row++)
  {
    for (int col = 0; col < matrix.cols(); col++)
      out << matrix(row, col) << " ";
    out << endl;
  }
  out << "]" << endl;
  return out;
}

//////////////////////////////////////////////////////////////////////
// Deep copy equality operator
//////////////////////////////////////////////////////////////////////
MATRIX& MATRIX::operator=(const MATRIX m)
{
  if (m._cols != _cols || m._rows != _rows)
  {
    delete[] _matrix;
    _cols = m._cols;
    _rows = m._rows;

    _matrix = new Real[_rows * _cols];
  }
  for (int x = 0; x < _rows * _cols; x++)
    _matrix[x] = m._matrix[x];

  return *this;
}

//////////////////////////////////////////////////////////////////////
// self minus
//////////////////////////////////////////////////////////////////////
MATRIX& MATRIX::operator-=(const MATRIX& m)
{
  if (m._cols != _cols || m._rows != _rows)
  {
    delete[] _matrix;
    _cols = m._cols;
    _rows = m._rows;

    _matrix = new Real[_rows * _cols];
  }
  for (int x = 0; x < _rows * _cols; x++)
    _matrix[x] -= m._matrix[x];

  return *this;
}

//////////////////////////////////////////////////////////////////////
// self plus
//////////////////////////////////////////////////////////////////////
MATRIX& MATRIX::operator+=(const MATRIX& m)
{
  if (m._cols != _cols || m._rows != _rows)
  {
    delete[] _matrix;
    _cols = m._cols;
    _rows = m._rows;

    _matrix = new Real[_rows * _cols];
  }
  for (int x = 0; x < _rows * _cols; x++)
    _matrix[x] += m._matrix[x];

  return *this;
}

//////////////////////////////////////////////////////////////////////
// Return the matrix diagonal
//////////////////////////////////////////////////////////////////////
VECTOR MATRIX::diagonal()
{
  int minDim = (_rows > _cols) ? _cols : _rows;
  VECTOR diag(minDim);
  for (int x = 0; x < minDim; x++)
    diag(x) = (*this)(x,x);

  return diag;
}

//////////////////////////////////////////////////////////////////////
// stomp the current matrix with the given matrix starting at "row". 
// It is your responsibility to ensure that you don't fall off the 
// end of this matrix.
//////////////////////////////////////////////////////////////////////
void MATRIX::setSubmatrix(MATRIX& matrix, int row)
{
  int totalSize = matrix.rows() * matrix.cols();
  int index = row * _cols;

  for (int x = 0; x < totalSize; x++, index++)
    _matrix[index] = matrix._matrix[x];
}

//////////////////////////////////////////////////////////////////////
// Stomp the current matrix with the given vector,
// starting at (row,col). So the first elemenet of the vector will
// be copied into (row,col), the next into (row+1, col), etc.
//////////////////////////////////////////////////////////////////////
void MATRIX::setVector( VECTOR& vector, int row, int col )
{
	for( int j = 0; j < vector.size(); j++ ) {
		(*this)(row+j, col) = vector(j);
	}
}

//////////////////////////////////////////////////////////////////////
// This assumes row-major storage
//////////////////////////////////////////////////////////////////////
void MATRIX::copyRowFrom( MATRIX& src, int srcRow, int row )
{
#if BOUNDS_CHECKING_ENABLED
	if (src.cols() != _cols)
		cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : Matrix #cols do not match! " << endl;
#endif
	memcpy( &(*this)(row,0), &src(srcRow,0), sizeof(Real)*_cols );
}

//////////////////////////////////////////////////////////////////////
// BLAS axpy operation: B += alpha * A, where B is this matrix
//////////////////////////////////////////////////////////////////////
void MATRIX::axpy(Real alpha, MATRIX& A)
{
#if BOUNDS_CHECKING_ENABLED
  if (A.rows() != _rows || A.cols() != _cols)
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : Matrix sizes do not match! " << endl;
#endif

  int total = _rows * _cols;
  for (int x = 0; x < total; x++)
    _matrix[x] += alpha * A._matrix[x];
}

//////////////////////////////////////////////////////////////////////
// BLAS axpy operation: B = alpha * A, where B is this matrix, and 
// current contents of B are stomped
//////////////////////////////////////////////////////////////////////
void MATRIX::clearingAxpy(Real alpha, MATRIX& A)
{
#if BOUNDS_CHECKING_ENABLED
  if (A.rows() != _rows || A.cols() != _cols)
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : Matrix sizes do not match! " << endl;
#endif

  int total = _rows * _cols;
  for (int x = 0; x < total; x++)
    _matrix[x] = alpha * A._matrix[x];
}

//////////////////////////////////////////////////////////////////////
// BLAS gemm operation: C += alpha * A * B where C is this matrix
//////////////////////////////////////////////////////////////////////
void MATRIX::gemm(Real alpha, MATRIX& A, MATRIX& B)
{
#if BOUNDS_CHECKING_ENABLED
  if (A.cols() != B.rows() || A.rows() != _rows || B.cols() != _cols)
  {
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Matrix dimensions do not match! " << endl;
    cout << "Matrix A: " << A.cols() << " " << A.cols() << endl;
    cout << "Matrix B: " << B.cols() << " " << B.cols() << endl;
    return;
  }
#endif

#ifdef USE_OMP
#pragma omp parallel for schedule(static) default(shared)
#endif
  for (int i = 0; i < A.rows(); i++)
    for (int j = 0; j < B.cols(); j++)
    {
      Real sum = 0.0;
      for (int k = 0; k < A.cols(); k++)
        sum += A(i, k) * B(k, j);

      (*this)(i,j) += alpha * sum;
    }
}

/*
 * Untested -- don't uncomment until a test case comes up
 * 
//////////////////////////////////////////////////////////////////////
// BLAS gemv operation: y += A * x where A is this matrix
//////////////////////////////////////////////////////////////////////
VECTOR MATRIX::gemv(VECTOR& x)
{
  if (x.size() != _rows)
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Vector dimensions do not match!" << endl;

  VECTOR y(x.size());
  for (int j = 0; j < _cols; j++)
    for (int i = 0; i < _rows; i++)
      y(j) += (*this)(i,j) * x(j);

  return y;
}
*/

//////////////////////////////////////////////////////////////////////
// BLAS gemv operation: y += A * x where A is this matrix
//////////////////////////////////////////////////////////////////////
VECTOR MATRIX::gemv(VEC3F& x)
{
#if BOUNDS_CHECKING_ENABLED
  if (_cols != 3)
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Vector dimensions do not match!" << endl;
#endif

  VECTOR y(_rows);
  for (int j = 0; j < _cols; j++)
    for (int i = 0; i < _rows; i++)
      y(i) += (*this)(i,j) * x[i];

  return y;
}

//////////////////////////////////////////////////////////////////////
// BLAS gemv operation: y += alpha * A * x where A is this matrix
//////////////////////////////////////////////////////////////////////
VECTOR MATRIX::gemv(Real alpha, VEC3F& x)
{
#if BOUNDS_CHECKING_ENABLED
  if (_cols != 3)
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Vector dimensions do not match!" << endl;
#endif

  VECTOR y(_rows);
  for (int j = 0; j < _cols; j++)
  {
    for (int i = 0; i < _rows; i++)
      y(i) += (*this)(i,j) * x[i];
    y(j) *= alpha;
  }

  return y;
}

//////////////////////////////////////////////////////////////////////
// BLAS gemm operation: C = alpha * A * B where C is this matrix and
// current contents of C are stomped
//////////////////////////////////////////////////////////////////////
void MATRIX::clearingGemm(Real alpha, MATRIX& A, MATRIX& B)
{
#if BOUNDS_CHECKING_ENABLED
  if (A.cols() != B.rows() || A.rows() != _rows || B.cols() != _cols)
  {
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Matrix dimensions do not match! " << endl;
    cout << "Matrix A: " << A.rows() << " " << A.cols() << endl;
    cout << "Matrix B: " << B.rows() << " " << B.cols() << endl;
    cout << "Matrix C: " << this->rows() << " " << this->cols() << endl;
    return;
  }
#endif

  for (int i = 0; i < A.rows(); i++)
    for (int j = 0; j < B.cols(); j++)
    {
      Real sum = 0.0;
      for (int k = 0; k < A.cols(); k++)
        sum += A(i, k) * B(k, j);

      (*this)(i,j) = alpha * sum;
    }
}

//////////////////////////////////////////////////////////////////////
// Matrix-vector multiply
//////////////////////////////////////////////////////////////////////
void MATRIX::multiplyInplace(VECTOR& x, VECTOR& y) 
{
#if BOUNDS_CHECKING_ENABLED
  if (_cols != x.size())
    cout << __FILE__ << " " << __LINE__ << " : Matrix-Vector dim mismatch! " << endl;
#endif

  // do the product into another vector in case x is also y
  VECTOR z(y.size());
  z.clear();
#ifdef USE_OMP
#pragma omp parallel for schedule(static) default(shared)
#endif
  for (int i = 0; i < _rows; i++)
    for (int j = 0; j < _cols; j++)
      z(i) += x(j) * (*this)(i, j);
  y = z;
}

void MATRIX::subMatrixMultiplyInplace( VECTOR& x, VECTOR& prod, int subRows, int subCols, bool transpose )
{
	cout << __FILE__ << " " << __LINE__ << " : MATRIX_DEBUG.cpp is being used, so"
			 << "subMatrixMultiplyInplace is not implemented" << endl;
}

void MATRIX::uppertriMultiplyInplace( VECTOR& x, VECTOR& prod )
{
	cout << __FILE__ << " " << __LINE__ << " : MATRIX_DEBUG.cpp is being used, so"
			 << "uppertriMultiplyInplace is not implemented" << endl;
}

//////////////////////////////////////////////////////////////////////
// solve the linear system Ax = b, return x in the passed in b
//////////////////////////////////////////////////////////////////////
void MATRIX::solve(VECTOR& b)
{
  cout << __FILE__ << " " << __LINE__ << " : MATRIX_DEBUG.cpp is being used, so" 
       << "there is no LU solver. Use MATRIX_FAST.cpp." << endl;
}

void MATRIX::eigensystem(VECTOR &eigenvalues, MATRIX &eigenvectors)
{
  cout << __FILE__ << " " << __LINE__ << " : MATRIX_DEBUG.cpp is being used, so" 
       << "there is no eigen solver. Use MATRIX_FAST.cpp." << endl;
}

//////////////////////////////////////////////////////////////////////
// solve for the eigensystem of the matrix
//////////////////////////////////////////////////////////////////////

// (stevenan) 10/23/08 Removed PETSC dependency - we don't need it for the shells project.
#if 0
#include "petsc.h"
#include "petscblaslapack.h"
void MATRIX::eigensystem(VECTOR& eigenvalues, MATRIX& eigenvectors)
{
  // basic error checking
  if (_rows != _cols)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " Matrix must be square to get eigenvalues! " << endl;
    return;
  }

  // resize result space
  eigenvalues.resizeAndWipe(_rows);
  eigenvectors.resizeAndWipe(_rows, _rows);

  PetscBLASInt rowsize = _rows;
  PetscBLASInt worksize = 5 * _rows;
  PetscScalar* work;
  PetscReal*   valuesReal;
  PetscReal*   valuesImag;
  PetscScalar* input;
  PetscScalar* vectors;

  // allocate space
  PetscMalloc(2 * _rows * sizeof(PetscReal),&valuesReal);
  PetscMalloc(worksize * sizeof(PetscReal),&work);
  PetscMalloc(_rows * _rows * sizeof(PetscReal),&input);
  PetscMalloc(_rows * _rows * sizeof(PetscScalar),&vectors);
  valuesImag = valuesReal + _rows;

  // copy matrix into PetSc array
  for (int x = 0; x < _rows * _rows; x++)
    input[x] = _matrix[x];
 
  // the actual LAPACK call
  PetscBLASInt error;
  LAPACKgeev_("V","N", &rowsize, input, &rowsize, 
              valuesReal, valuesImag, 
              vectors, &rowsize, NULL, &rowsize,
              work, &worksize, &error);

  if (error != 0)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " eigenvalue solver bombed!" << endl;
  }

  // copy out results
  for (int x = 0; x < _rows; x++)
    eigenvalues(x) = valuesReal[x];

  for (int x = 0; x < _rows; x++)
    for (int y = 0; y < _rows; y++)
      eigenvectors(x,y) = vectors[x + y * _cols];
 
  // cleanup
  PetscFree(input);
  PetscFree(valuesReal);
  PetscFree(work);
  PetscFree(vectors);
}
#endif

//////////////////////////////////////////////////////////////////////
// copy this matrix to MATRIX3 type
//////////////////////////////////////////////////////////////////////
void MATRIX::copiesInto(MATRIX3& matrix3)
{
  if (_rows != 3 || _cols != 3)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << "Trying to copy a MATRIX that is not 3x3 to a MATRIX3!" << endl;
    return;
  }
  for (int y = 0; y < 3; y++)
    for (int x = 0; x < 3; x++)
      matrix3(x,y) = (*this)(x,y);
}

Real MATRIX::differenceFrobeniusSq( MATRIX& B )
{
	Real frobSq = 0.0;
	for( int i = 0; i < _rows; i++ ) {
		for( int j = 0; j < _cols; j++ ) {
			Real diff = (*this)(i,j) - B(i,j);
			frobSq += diff*diff;
		}
	}
	return frobSq;
}
