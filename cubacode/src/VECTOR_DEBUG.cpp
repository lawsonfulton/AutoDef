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
// VECTOR.h: interface for the VECTOR class.
//
//////////////////////////////////////////////////////////////////////

#include "VECTOR.h"
#include <stdio.h>
#include <string.h>

#ifdef USE_OMP
#include <omp.h>
#endif

//////////////////////////////////////////////////////////////////////
// Constructor for the full vector
//////////////////////////////////////////////////////////////////////
VECTOR::VECTOR() :
  _size(0)
{
  _vector = NULL;
}

VECTOR::VECTOR(int size) :
  _size(size)
{
  _vector = new Real[_size];
  clear();
}

VECTOR::VECTOR(const char* filename)
{
  read(filename);
}

VECTOR::VECTOR(const VECTOR& v) 
{
  _size = v._size;
  _vector = new Real[_size];
  for (int x = 0; x < _size; x++)
    _vector[x] = v._vector[x];
}

VECTOR::VECTOR(FILE* file)
{
  // read dimensions
  fread((void*)&_size, sizeof(int), 1, file);

  // read data
  _vector = new Real[_size];
  if (sizeof(Real) == sizeof(float))
  {
    double* vecDouble = new double[_size];
    fread((void*)vecDouble, sizeof(double), _size, file);
    for (int x = 0; x < _size; x++)
      _vector[x] = vecDouble[x];
    delete[] vecDouble;
  }
  else
    fread((void*)_vector, sizeof(Real), _size, file);
}

VECTOR::~VECTOR()
{
  delete[] _vector;
}

//////////////////////////////////////////////////////////////////////
// dot product with another vector
//////////////////////////////////////////////////////////////////////
Real VECTOR::operator*(const VECTOR& vector)
{
#if BOUNDS_CHECKING_ENABLED
  if (vector._size != _size)
  {
    cout << __FILE__ << " " << __LINE__ << " VECTOR dot sizes do not match!: " << endl;
    return 123456.0f;
  }
#endif
  Real total = 0.0f;
  for (int x = 0; x < _size; x++)
    total += vector._vector[x] * _vector[x];

  return total;
}

//////////////////////////////////////////////////////////////////////
// scale vector by a constant
//////////////////////////////////////////////////////////////////////
VECTOR& VECTOR::operator*=(const Real& alpha)
{
  for (int x = 0; x < _size; x++)
    _vector[x] *= alpha;
  
  return *this;
}

//////////////////////////////////////////////////////////////////////
// wipe the whole vector
//////////////////////////////////////////////////////////////////////
void VECTOR::clear()
{
  memset(_vector, 0, _size * sizeof(Real));
}

//////////////////////////////////////////////////////////////////////
// resize and wipe the vector
//////////////////////////////////////////////////////////////////////
void VECTOR::resizeAndWipe(int size)
{
  if (_size != size)
    delete[] _vector;
  _size = size;

  _vector = new Real[_size];
  clear();
}

//////////////////////////////////////////////////////////////////////
// write the vector to a file
//////////////////////////////////////////////////////////////////////
void VECTOR::write(const char* filename)
{
  FILE* file;
  file = fopen(filename, "wb");

  // write dimensions
  fwrite((void*)&_size, sizeof(int), 1, file);

  // write data
  if (sizeof(Real) == sizeof(float))
  {
    double* vecDouble = new double[_size];
    for (int x = 0; x < _size; x++)
      vecDouble[x] = _vector[x];
    fwrite((void*)vecDouble, sizeof(double), _size, file);
    delete[] vecDouble;
  } 
  else
    fwrite((void*)_vector, sizeof(Real), _size, file);
  fclose(file);
}

//////////////////////////////////////////////////////////////////////
// write the vector to a file
//////////////////////////////////////////////////////////////////////
void VECTOR::write(FILE* file)
{
  // write dimensions
  fwrite((void*)&_size, sizeof(int), 1, file);

  // write data
  if (sizeof(Real) == sizeof(float))
  {
    double* vecDouble = new double[_size];
    for (int x = 0; x < _size; x++)
      vecDouble[x] = _vector[x];
    fwrite((void*)vecDouble, sizeof(double), _size, file);
    delete[] vecDouble;
  } 
  else
    fwrite((void*)_vector, sizeof(Real), _size, file);
}

//////////////////////////////////////////////////////////////////////
// read vector from a file
//////////////////////////////////////////////////////////////////////
bool VECTOR::read(const char* filename)
{
  FILE* file;
  file = fopen(filename, "rb");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __LINE__ << " : File " << filename << " not found! " << endl;
    return false;
  }

  // read dimensions
  fread((void*)&_size, sizeof(int), 1, file);

  // read data
  delete[] _vector;
  _vector = new Real[_size];

  if (sizeof(Real) == sizeof(float))
  {
    double* vecDouble = new double[_size];
    fread((void*)vecDouble, sizeof(double), _size, file);
    for (int x = 0; x < _size; x++)
      _vector[x] = vecDouble[x];
    delete[] vecDouble;
  }
  else
    fread((void*)_vector, sizeof(Real), _size, file);
  fclose(file);

  return true;
}

//////////////////////////////////////////////////////////////////////
// Deep copy equality operator
//////////////////////////////////////////////////////////////////////
VECTOR& VECTOR::operator=(VECTOR m)
{
  if (m.size() != _size)
  {
    delete[] _vector;
    _size= m.size();
    _vector= new Real[_size];
  }
  for (int x = 0; x < _size; x++)
    _vector[x] = m._vector[x];

  return *this;
}

//////////////////////////////////////////////////////////////////////
// Self-add operator
//////////////////////////////////////////////////////////////////////
VECTOR& VECTOR::operator+=(const VECTOR& m)
{
#if BOUNDS_CHECKING_ENABLED
  if (m._size != _size)
    cout << __FILE__ << " " << __LINE__ << " : Vector sizes don't match! " << endl;
#endif

  for (int x = 0; x < _size; x++)
    _vector[x] += m._vector[x];
  return *this;
}

//////////////////////////////////////////////////////////////////////
// Self-subtract operator
//////////////////////////////////////////////////////////////////////
VECTOR& VECTOR::operator-=(const VECTOR& m)
{
#if BOUNDS_CHECKING_ENABLED
  if (m._size != _size)
  {
    cout << __FILE__ << " " << __LINE__ << " : Vector sizes don't match! " << endl;
    cout << " this: " << _size << endl;
    cout << " input: " << m._size << endl;
  }
#endif

  for (int x = 0; x < _size; x++)
    _vector[x] -= m._vector[x];
  return *this;
}

//////////////////////////////////////////////////////////////////////
// compute the 2 norm
//////////////////////////////////////////////////////////////////////
Real VECTOR::norm2()
{
  Real total = 0;
  for (int x = 0; x < _size; x++)
    total += _vector[x] * _vector[x];
  return sqrtf(total);
}

//////////////////////////////////////////////////////////////////////
// largest element in vector
//////////////////////////////////////////////////////////////////////
Real VECTOR::max_()
{
	Real vmax = _vector[0];

	for ( int i = 1; i < _size; i++ )
	{
		if ( _vector[i] > vmax )
			vmax = _vector[i];
	}

	return vmax;
}

//////////////////////////////////////////////////////////////////////
// smallest element in vector
//////////////////////////////////////////////////////////////////////
Real VECTOR::min_()
{
	Real vmin = _vector[0];

	for ( int i = 1; i < _size; i++ )
	{
		if ( _vector[i] < vmin )
			vmin = _vector[i];
	}

	return vmin;
}

//////////////////////////////////////////////////////////////////////
// largest absolute element in vector
//////////////////////////////////////////////////////////////////////
Real VECTOR::absmax()
{
	Real vmax = fabs( _vector[0] );

	for ( int i = 1; i < _size; i++ )
	{
		if ( fabs( _vector[i] ) > vmax )
			vmax = fabs( _vector[i] );
	}

	return vmax;
}

//////////////////////////////////////////////////////////////////////
// smallest absolute element in vector
//////////////////////////////////////////////////////////////////////
Real VECTOR::absmin()
{
	Real vmin = fabs( _vector[0] );

	for ( int i = 1; i < _size; i++ )
	{
		if ( fabs( _vector[i] ) < vmin )
			vmin = fabs( _vector[i] );
	}

	return vmin;
}

//////////////////////////////////////////////////////////////////////
// swap contents with another vector
//////////////////////////////////////////////////////////////////////
void VECTOR::swap(VECTOR& vec)
{
  Real* temp = _vector;
  _vector = vec._vector;
  vec._vector = temp;
}

//////////////////////////////////////////////////////////////////////
// BLAS axpy operation: y += alpha * x, where y is this vector
//////////////////////////////////////////////////////////////////////
void VECTOR::axpy(Real alpha, VECTOR& x)
{
#if BOUNDS_CHECKING_ENABLED
  if (_size != x._size)
    cout << __FILE__ << " " << __LINE__ << " : Vector size do not match! " << endl;
#endif

#ifdef USE_OMP
#pragma omp parallel for schedule(static) default(shared)
#endif
  for (int i = 0; i < _size; i++)
    _vector[i] += alpha * x._vector[i];
}

//////////////////////////////////////////////////////////////////////
// same as axpy above, but vector contents are stomped as well
//////////////////////////////////////////////////////////////////////
void VECTOR::clearingAxpy(Real alpha, VECTOR& x)
{
#if BOUNDS_CHECKING_ENABLED
  if (_size != x._size)
    cout << __FILE__ << " " << __LINE__ << " : Vector size do not match! " << endl;
#endif

  for (int i = 0; i < _size; i++)
    _vector[i] = alpha * x._vector[i];
}

//////////////////////////////////////////////////////////////////////
// return the sum of the vector
//////////////////////////////////////////////////////////////////////
Real VECTOR::sum()
{
  Real total = 0.0f;
  for (int x = 0; x < _size; x++)
    total += _vector[x];
  return total;
}

//////////////////////////////////////////////////////////////////////
// in-place copy, since operator= must allocate a new VECTOR
//////////////////////////////////////////////////////////////////////
void VECTOR::copyInplace(VECTOR& vector)
{
#if BOUNDS_CHECKING_ENABLED
  if (_size != vector._size)
    cout << __FILE__ << " " << __LINE__ << " : Vector sizes do not match! " << endl;
#endif
  for (int x = 0; x < _size; x++)
    _vector[x] = vector._vector[x];
}

void VECTOR::copyInplace(VECTOR& vector, int start)
{
#if BOUNDS_CHECKING_ENABLED
  if ( (_size - start) < vector._size)
    cout << __FILE__ << " " << __LINE__ << " : Vector too small for copy! " << endl;
#endif
  for (int x = 0; x < vector._size; x++)
    _vector[x+start] = vector._vector[x];
}

//////////////////////////////////////////////////////////////////////
// add two vectors
//////////////////////////////////////////////////////////////////////
VECTOR operator+(VECTOR& x, VECTOR& y) 
{
  VECTOR z(x.size());

  for (int i = 0; i < x.size(); i++)
    z(i) = x(i) + y(i);

  return z;
}

//////////////////////////////////////////////////////////////////////
// subtract two vectors
//////////////////////////////////////////////////////////////////////
VECTOR operator-(VECTOR& x, VECTOR& y) 
{
  VECTOR z(x.size());

  for (int i = 0; i < x.size(); i++)
    z(i) = x(i) - y(i);

  return z;
}
