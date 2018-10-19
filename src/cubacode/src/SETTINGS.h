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
// SETTINGS.h: Project-wide options set in one place
//
//////////////////////////////////////////////////////////////////////

#ifndef SETTINGS_H
#define SETTINGS_H

#define BOUNDS_CHECKING_ENABLED 0
//#define USING_OPENMP 

// At least Intel MKL or ATLAS must be installed in order
// to compile and run "CubatureViewer". Select one and only one to use
//#define USING_MKL

// select single or double precision
#define DOUBLE_PRECISION
#define Real double
#define EPSILON 0.000000000001
//#define SINGLE_PRECISION
//#define Real float

#endif
