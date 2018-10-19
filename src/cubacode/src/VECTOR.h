// VECTOR.h: interface for the VECTOR class.
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
//
//////////////////////////////////////////////////////////////////////

#ifndef VECTOR_H
#define VECTOR_H

#include "SETTINGS.h"
#include <map>
#include <iostream>
#include <cstdio>
#include <cmath>

using namespace std;

//////////////////////////////////////////////////////////////////////
// An arbitrary dimension vector class
//////////////////////////////////////////////////////////////////////
class VECTOR {

public:
  VECTOR();
  VECTOR(int size);
  VECTOR(const char* filename);
  VECTOR(const VECTOR& v);
  VECTOR(FILE* file);
  ~VECTOR();

  inline Real& operator()(int index) { return _vector[index]; };
 
  int& size() { return _size; };

  // wipe the whole vector
  void clear();

  // write the vector to a binary file
  // everything is always written as a double
  void write(const char* filename);
  void write(FILE* file);

  // read the vector to a binary file
  // everything is always read in as a double, then
  // converted if necessary
  // Returns true if successfully read.
  bool read(const char* filename);

  // resize the vector and wipe to zero
  void resizeAndWipe(int size);

  // overloaded operators
  VECTOR& operator=(VECTOR m);
  VECTOR& operator+=(const VECTOR& v);
  VECTOR& operator-=(const VECTOR& v);
  VECTOR& operator*=(const Real& alpha);

  // (stevenan) The trailing underscore is to make this windows-compatible.
  // http://polingplace.blogspot.com/2007/04/stdmin-and-stdmax-versus-visual-studio.html
  // Thanks, Microsoft.
  Real max_();
  Real min_();

  Real absmax();
  Real absmin();

  // 2 norm
  Real norm2();

  // dot product
  Real operator*(const VECTOR& vector);

  // swap contents with another vector --
  // it's your job to ensure that they are the same size
  void swap(VECTOR& vector);

  // raw data pointer
  Real* data() { return _vector; };

  // BLAS axpy operation: y += alpha * x, where y is this vector
  void axpy(Real alpha, VECTOR& x);

  // same as axpy above, but vector contents are stomped as well
  void clearingAxpy(Real alpha, VECTOR& x);

  // sum of all the elements
  Real sum();

  // in-place copy, since operator= must allocate a new VECTOR
  void copyInplace(VECTOR& vector);

	// In place copy from one vector to a specified location
	// in this vector.
	void copyInplace(VECTOR &vector, int startLoc);

private:
  int _size;
  Real* _vector;
};

//////////////////////////////////////////////////////////////////////
// dump vector to iostream
//////////////////////////////////////////////////////////////////////
inline ostream &operator<<(ostream &out, VECTOR& vector)
{
  out << "[" << endl; 
  for (int x = 0; x < vector.size(); x++)
    out << vector(x) << endl;
  out << "]" << endl;
  return out;
}

// overloaded operators
VECTOR operator-(VECTOR& x, VECTOR& y);
VECTOR operator+(VECTOR& x, VECTOR& y);

#endif
