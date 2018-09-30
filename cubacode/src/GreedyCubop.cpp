// Copyright (C) 2009 Cornell University
// All rights reserved.
// Original Author: Steven An (http://www.cs.cornell.edu/~stevenan)

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


#include "GreedyCubop.h"

//#define USE_OMP
//#define USE_CBLAS

#include <algorithm>
#include <cassert>
#include <iostream>
#include <ctime>
#include <set>
#include "MATRIX.h"
#include "steve.h"
#include <stdio.h>
#include <string.h>

#if USE_CBLAS
	#include <mkl_types.h>
	#include <mkl_cblas.h>
#endif

#ifdef USE_OMP
	#include <omp.h>
#endif

#include "nnls.h"

using namespace std;


/**
 * Wrapper for the NNLS C code.
 * We use doubles instead of Reals here, because the C code assumes doubles.
 */
class NnlsSolver
{
public:

	NnlsSolver( int rows, int maxCols ) :
		_rows( rows ),
		_maxCols( maxCols )
	{
		_workB = new double[ _rows ];
		_workA = new double[ _rows * _maxCols ];
		_workW = new double[ _maxCols ];
		_workZ = new double[ _rows ];
		_workIndex = new int[ _maxCols ];
	} 

	~NnlsSolver()
	{
		delete[] _workB;
		delete[] _workA;
		delete[] _workW;
		delete[] _workZ;
		delete[] _workIndex;
	}

	bool solve( double* A_colmajor, int numCols, VECTOR& b, VECTOR& x, double& residualNormOut  )
	{
		assert( numCols <= _maxCols );
		assert( x.size() == numCols );
		assert( b.size() == _rows );

		// make copies of A and b, since nnls() clobbers them
		memcpy( _workB, b.data(), _rows * sizeof(double) );
		memcpy( _workA, A_colmajor, numCols * _rows * sizeof(double) );

		// do it
		nnls( _workA, _rows, _rows, numCols,
			_workB, x.data(), &residualNormOut,
			_workW, _workZ, _workIndex, &_workMode );

		if( _workMode == 1 ) return true;
		else if( _workMode == 3 ) return false;
		else {
			cerr << "PROGRAMMER ERROR CALLING NNLS";
			return false;
		}
	}

	bool solveRowMajor( double* A_rowmajor, int allocedCols, int numCols, VECTOR& b, VECTOR& x, double& residualNormOut )
	{
		assert( numCols <= _maxCols );
		assert( x.size() == numCols );
		assert( b.size() == _rows );

		// make copies of A and b, since nnls() clobbers them
		memcpy( _workB, b.data(), _rows * sizeof(double) );

		// But, copy A into column major form, since nnls expects that
		// So, can't use memcpy here.
		for( int i = 0; i < _rows; i++ ) {
			for( int j = 0; j < numCols; j++ ) {
				_workA[ j*_rows + i ] = A_rowmajor[ i*allocedCols + j ];
			}
		}

		// do it
		nnls( _workA, _rows, _rows, numCols,
			_workB, x.data(), &residualNormOut,
			_workW, _workZ, _workIndex, &_workMode );

		if( _workMode == 1 ) return true;
		else if( _workMode == 3 ) return false;
		else {
			cerr <<"PROGRAMMER ERROR CALLING NNLS";
			return false;
		}
	}

private:

	int _rows;
	int _maxCols;

	double* _workB;
	double* _workA;
	double* _workW;
	double* _workZ;
	int* _workIndex;
	int _workMode;
};

//----------------------------------------
//  y = alpha*A*x
//----------------------------------------
void MatrixMult_ColMajor( int m, int n, double alpha, double* A, double* x, double* y )
{
#ifdef USE_CBLAS
	cblas_dgemv(
		CblasColMajor, 	// important
		CblasNoTrans, 
		m, n, 
		1.0, A, m, x, 1, 
		0.0, y, 1);
#else
	for( int i = 0; i < m; i++ )
	{
		y[i] = 0.0;

		for( int k = 0; k < n; k++ )
		{
			y[i] += alpha * A[ k*m + i ] * x[k];
		}
	}
#endif
}

void MatrixMult_RowMajor( int m, int n, double alpha, Real* A, int stride, Real* x, Real* y )
{
#ifdef USE_CBLAS
	cblas_dgemv(
		CblasRowMajor, 	// important
		CblasNoTrans, 
		m, n, 
		1.0, A, m, x, 1, 
		0.0, y, 1);
#else
	for( int i = 0; i < m; i++ )
	{
		y[i] = 0.0;

		for( int k = 0; k < n; k++ )
		{
			y[i] += A[ i*stride + k ] * x[k];
		}

		y[i] *= alpha;
	}
#endif
}

static void calcRowMajorResidual( Real* Adata, int stride, int T, int r, VECTOR& w, VECTOR& b, VECTOR& residOut )
{
	VECTOR bAct( b.size() );
	MatrixMult_RowMajor( r*T, w.size(), 1.0, Adata, stride, w.data(), bAct.data() );

	// subtract bAct from b
	residOut = b;
	residOut -= bAct;
}

static bool contains( vector<int>* v, int val )
{
	for( int i = 0; i < v->size(); i++ )
	{
		if( (*v)[i] == val ) return true;
	}
	return false;
}

static bool contains( set<int>& v, int val )
{
	return v.find( val ) != v.end();
}

template<typename T> int find_max( T* array, int size, T& maxVal ) {
	int maxIdx = 0;
	maxVal = array[0];
	for( int i = 1; i < size; i++ ) {
		if( array[i] > maxVal ) {
			maxVal = array[i];
			maxIdx = i;
		}
	}
	return maxIdx;
}

void GreedyCubop::gatherRandomUnusedPoints( vector<int>& usedPoints, vector<int>& candidatesOut, int maxCands )
{
	// Choose R random points that aren't in usedPoints by Rejection sampling

	// First create a map of used points
	set<int> usedPointsSet;
	for( int i = 0; i < usedPoints.size(); i++ )
		usedPointsSet.insert( usedPoints[i] );

	while( candidatesOut.size() < maxCands && usedPointsSet.size() < numTotalPoints() ) {
		int randPointId = _rand.randInt( numTotalPoints()-1 );

		// make sure it's not used, AND it wasn't made a candidate already
		if( !contains( usedPointsSet, randPointId ) ) {
			// Add to our list
			candidatesOut.push_back( randPointId );

			// Remember that it was used already
			usedPointsSet.insert( randPointId );
		}

		// TODO maybe
		// also, assuming usedPointsSet.size is near numTotalPoints, this could take a while. but that's unlikely in the cubop algorithm
	}
}

void GreedyCubop::gatherUnusedPoints( vector<int>& usedPoints, vector<int>& candidatesOut )
{
	// First create a map of used points
	set<int> usedPointsSet;
	for( int i = 0; i < usedPoints.size(); i++ )
		usedPointsSet.insert( usedPoints[i] );

	for( int pointId = 0; pointId < numTotalPoints(); pointId++ ) {
		if( !contains( usedPointsSet, pointId ) ) {
			candidatesOut.push_back( pointId );
			// no need to insert into our set map, since we'll never come to this point again
		}
	}
}

static void write_col_major_matrix( double* A, int m, int n, ostream& os )
{
	os << "[";
	for( int i = 0; i < m; i++ ) {
		for( int j = 0; j < n; j++ ) {
			os << A[ j*m + i ] << " ";
		}
		os << endl;
	}
	os << "]";
}


void GreedyCubop::evalPointColumn(
		int pointId,
		TrainingSet& trainingSet,
		vector<Real>& inverseForceNorms,
	   	VECTOR& columnOut )
{
	VECTOR g(_r);

	for( int t = 0; t < trainingSet.size(); t++ )
	{
		evalPointForceDensity( pointId, *trainingSet[t], g );

		// scale g by 1/norm(f_t) and copy into the column
		g *= inverseForceNorms[t];
		memcpy( &( columnOut.data()[ t*_r ] ), g.data(), _r * sizeof(Real) );
	}
}

//----------------------------------------
//  This normalizes each block-entry of the given b vector and
//  returns the inverse-norms (by which they were scaled).
//----------------------------------------
static void normalizeTrainingForces( int r, VECTOR& forces, vector<Real>& inverseForceNorms )
{
	assert( forces.size() % r == 0 );

	const int T = forces.size() / r;

	inverseForceNorms.resize( T );

	for( int t = 0; t < T; t++ )
	{
		Real bblockSqrSum = 0.0;

		// Do our own per-block norm
		for( int i = 0; i < r; i++ ) {
			Real e = forces( t*r + i );
			bblockSqrSum += e*e;
		}

		if( abs(bblockSqrSum) > 1e-8 )
		{
			inverseForceNorms[t] = 1.0 / sqrt( bblockSqrSum );

			// scale forces
			for( int i = 0; i < r; i++ ) {
				forces( t*r + i ) *= inverseForceNorms[t];
			}
		}
		else
		{
			cout << "R-mag-sq of training set " << t << " was near zero: " << bblockSqrSum << ". doing no scaling." << endl;
			inverseForceNorms[t] = 1.0;
		}
	}
}

void fill_range( vector<int>& vec, int start, int end ) {
	for( int i = start; i < end; i++ ) {
		vec.push_back(i);
	}
}

void sample_range( int start, int end, int numSamples, vector<int>& samplesOut ) {
	fill_range( samplesOut, start, end );
	random_shuffle( samplesOut.begin(), samplesOut.end() );
	samplesOut.resize( numSamples );
}

// analogous to matlab: out = from( subsetIndices )
template <typename T> void collect_subset( vector<T>& from, vector<int>& subsetIndices, vector<T>& out ) {
	for( int i = 0; i < subsetIndices.size(); i++ ) {
		out.push_back( from[ subsetIndices[i] ] );
	}
}

/**
 * OpenMP-friendly implementation of the argmax step.
 */
int GreedyCubop::findMostFitPoint(
		std::vector<int>& candidatePoints,
		VECTOR& residual,
		TrainingSet& trainingSet,
		vector<Real>& inverseForceNorms
		)
{
	const int T = trainingSet.size();
	const int nCands = candidatePoints.size();

	assert( residual.size() == _r * T );
	assert( inverseForceNorms.size() == T );

	Real residualNorm = residual.norm2();

	// Do this in parallel, so put all our candidates in a simple array
	// and each thread will fill out parts of this array
	Real* candDots = new Real[ nCands ];
	memset( candDots, 0, nCands * sizeof(Real) );

	// Loop complexity: O ( r * T * C) (C == numCandsPerIter)
#ifdef USE_OMP
#pragma omp parallel for schedule(static) default(shared)
#endif
	for( int i = 0; i < nCands; ++i ) {
		int candPointId = candidatePoints[i];

		// O(rT)
		VECTOR candCol( _r * T );
		evalPointColumn( candPointId, trainingSet, inverseForceNorms, candCol );

		Real candNorm = candCol.norm2();

		if( candNorm == 0.0 ) {
			// This candidate contributes nothing, so we shouldn't use it.
			// Just give it a very low value, and hopefully we'll get a positive one later on.
			// Otherwise...we'll end up adding this and it'll be a bad point.
			// Oh well - not much harm done.
			candDots[i] = -1e9;
		}
		else {
			// normalized dot prod
			candDots[i] = (candCol * residual) / candNorm / residualNorm;
		}
	}

	// now find the best one
	Real bestDot;
	int bestCandIdx = find_max<Real>( candDots, nCands, bestDot );

	// cout << "bestDot = " << bestDot << endl;

	delete candDots;

	return candidatePoints[bestCandIdx];
}

void GreedyCubop::selectCandidatePoints(
		vector<int>& usedPoints,
		int numCandidates,
		vector<int>& candidatesOut
		)
{
	if( numCandidates >= (numTotalPoints()-usedPoints.size()) ) {
		// just grab em all
		gatherUnusedPoints( usedPoints, candidatesOut );
	}
	else {
		// select a random subset of numCandidates-many unused points
		gatherRandomUnusedPoints( usedPoints, candidatesOut, numCandidates );
	}
}

GreedyCubop::GreedyCubop() :
	_rand( 42 )

{
}

void GreedyCubop::run(
		TrainingSet& trainingSet,
		VECTOR& trainingForces,	// r*T vector of ground-truth forces. 'b' in the paper.
		Real relErrTol,	// TOL
		int maxNumPoints,	// some sane limit, for overnight runs
		int numCandsPerIter,	// |C|
		int itersPerFullNNLS,	// r/2 in the paper
		int numSamplesPerSubtrain	// T_s
		)
{
	//----------------------------------------
	//  Convenience constants
	//----------------------------------------
	const int T = trainingSet.size();
	_r = trainingSet[0]->size();
	
	//----------------------------------------
	//  Calculate scalings for normalized training,
	//  and scale b.
	//----------------------------------------
	VECTOR b = trainingForces;
	vector<Real> inverseForceNorms;
	normalizeTrainingForces( _r, b, inverseForceNorms );

	//----------------------------------------
	//  Start training
	//----------------------------------------
	printf("Allocating matrix of %d x %d x %d bytes per Real = %d = %f MB. If this is too large, it could core-dump here.", _r*T, maxNumPoints, sizeof(Real), _r*T*maxNumPoints*sizeof(Real), _r*T*maxNumPoints*sizeof(Real) / 1024.0 / 1024.0 );
	printf("\n");
	MATRIX A( _r*T, maxNumPoints );
	A.clear();
	printf("Matrix allocation successful! No core-dump - OK.");
	printf("\n");

	vector<int> selectedPoints;
	VECTOR w;
	VECTOR residual( b );
	Real bNorm = b.norm2();
	cout << "bNorm = " << bNorm << " should be sqrt(T) = " << sqrt(T) << endl;
	Real residualNorm = bNorm;
	Real relErr = 1.0;

	MATRIX subsetA( _r*numSamplesPerSubtrain, maxNumPoints );
	subsetA.clear();

	NnlsSolver nnlser( _r*T, maxNumPoints );
	NnlsSolver subNNLS( _r*numSamplesPerSubtrain, maxNumPoints );

	//----------------------------------------
	//  Main loop
	//----------------------------------------

	printf("Entering cubop loop");
	printf("\n");

	while( relErr > relErrTol && selectedPoints.size() < maxNumPoints && selectedPoints.size() < numTotalPoints())
	{
		vector<int> subsetIndices( T );
		TrainingSet trainingSubset( T );
		vector<Real> inverseForceNormsSubset( T );

		// initially, we use all training samples
		fill_range( subsetIndices, 0, trainingSet.size() );
		copy( trainingSet.begin(), trainingSet.end(), trainingSubset.begin() );
		copy( inverseForceNorms.begin(), inverseForceNorms.end(), inverseForceNormsSubset.begin() );

		VECTOR subsetResidual( residual );
		VECTOR subsetB( b );

		for( int i = 0; i < itersPerFullNNLS && selectedPoints.size() < maxNumPoints; i++ ) {
			//----------------------------------------
			//  Select candidate points
			//----------------------------------------
			vector<int> candPointIds;
			selectCandidatePoints( selectedPoints, numCandsPerIter, candPointIds );

			//----------------------------------------
			//  Find the candidate that's most fit, ie. highest dot prodct with residual
			//  NOTE: this part can be easily paralelized! then we can do more key tets per iteration
			//----------------------------------------

			int bestPointId = findMostFitPoint( candPointIds, subsetResidual, trainingSubset, inverseForceNormsSubset );
			selectedPoints.push_back( bestPointId );
			// cout << "bestPointId = " << bestPointId << endl;

			// we'll need its column for all samples (not just the subset)
			VECTOR bestPointCol( _r*T );
			evalPointColumn( bestPointId, trainingSet, inverseForceNorms, bestPointCol );

			// Add this column to the A matrix
			A.setVector( bestPointCol, 0, selectedPoints.size()-1 );

			//----------------------------------------
			//	Now calculate a new residual
			//  Do NNLS with a new subset
			//----------------------------------------

			subsetIndices.clear();
			trainingSubset.clear();
			inverseForceNormsSubset.clear();

			// randomly choose new training subset
			sample_range( 0, T, numSamplesPerSubtrain, subsetIndices );
			collect_subset( trainingSet, subsetIndices, trainingSubset );
			collect_subset( inverseForceNorms, subsetIndices, inverseForceNormsSubset );

			// prep for NNLS for new subset
			subsetB.resizeAndWipe( numSamplesPerSubtrain * _r );
			subsetA.clear();
			for( int i = 0; i < subsetIndices.size(); i++ ) {
				int t = subsetIndices[i];
				for( int j = 0; j < _r; j++ ) {
					subsetA.copyRowFrom( A, t*_r+j, i*_r+j );
					subsetB( i*_r+j ) = b( t*_r+j );
				}
			}
			w.resizeAndWipe( selectedPoints.size() );

			// go!
			bool converged = subNNLS.solveRowMajor(
					subsetA.data(),
					subsetA.cols(),
					selectedPoints.size(),
					subsetB,
					w,
					residualNorm );

			subsetResidual.resizeAndWipe( numSamplesPerSubtrain * _r );
			calcRowMajorResidual( subsetA.data(), subsetA.cols(), numSamplesPerSubtrain, _r, w, subsetB, subsetResidual );

			// cout << "n = " << selectedPoints.size() << "s.  resid norm = " << residualNorm << " check = " << subsetResidual.norm2() << endl;
		}

		//----------------------------------------
		//  do NNLS with all samples now for an optimal output
		//----------------------------------------
		w.resizeAndWipe( selectedPoints.size() );
		bool converged = nnlser.solveRowMajor(
				A.data(),
				A.cols(),
				selectedPoints.size(),
				b,
				w,
				residualNorm );
		
		if( converged ) {
			//----------------------------------------
			//  Got a better solution. Update residual and stuff
			//----------------------------------------
			relErr = residualNorm / bNorm;
			calcRowMajorResidual( A.data(), A.cols(), T, _r, w, b, residual );

			// sanity checks
			// make sure we have the right residual calculated
			Real rnormErr = (residual.norm2() - residualNorm) / residualNorm;
			assert( rnormErr < 1e-6 );

			// Remove zero points
			// vector<int> nonzero_selectedPoints;
			// vector<Real> nonzero_w;
			// for(int i = 0; i < selectedPoints.size(); i++) {
			// 	if(w(i) > 1e-6) {
			// 		nonzero_selectedPoints.push_back(selectedPoints[i]);
			// 		nonzero_w.push_back(w(i));
			// 	}
			// }
			// selectedPoints = nonzero_selectedPoints;
			// w.resizeAndWipe(selectedPoints.size());
			// for(int i = 0; i < nonzero_w.size(); i++) {
			// 	w(i) = nonzero_w[i];
			// }

			//----------------------------------------
			//  Hand it off
			//----------------------------------------
			handleCubature( selectedPoints, w, relErr );

			/*
			char fnamebuf[128];
			sprintf( fnamebuf, "%03d tet relErr = %10.6f.cubature", selectedPoints.size(), relErr );
			write_cubature( outPrefix + string( fnamebuf ), selectedPoints, w );
			*/
		}
		else {
			cerr << "[ERROR] ** NNLS did not converge for selectedPoints.size() = " << selectedPoints.size() << endl;
		}
	}
}

void GreedyCubop::handleCubature( std::vector<int>& selectedPoints, VECTOR& weights, Real relErr )
{
	cout << "n = " << selectedPoints.size() << " \t relerr = " << (relErr*100) << "%" << endl;
}

static int nqp_iteration( MATRIX& Apos, MATRIX& Aneg4, VECTOR& b, VECTOR& x, Real convergenceTol = 1e-12 )
{
	assert( Apos.rows() == Apos.cols() );
	assert( Aneg4.rows() == Aneg4.cols() );
	assert( Apos.rows() == Aneg4.rows() );
	assert( b.size() == Apos.rows() );
	assert( x.size() == Apos.rows() );

	Real deltaNorm = 2*convergenceTol;
	int numIters = 0;
	VECTOR a(Apos.rows());
	VECTOR a2(Apos.rows());
	VECTOR c4(Apos.rows());
	VECTOR xOld(x.size());

	// Precompute some stuff

	VECTOR bNeg(b);
	bNeg *= -1;

	VECTOR bSquared(b);
	for( int i = 0; i < bSquared.size(); i++ ) bSquared(i) = b(i)*b(i);

	// Main iteration loop

	time_t lastShoutTime = time(NULL);

	while( deltaNorm > convergenceTol )
	{
		// Save for convergence comparison
		xOld = x;

		// Do the update
		// TODO - symmetric multiply? Apos/Aneg4 are symmetric!
		Apos.uppertriMultiplyInplace( x, a );
		Aneg4.uppertriMultiplyInplace( x, c4 );

		a2 = a;
		a2 *= 2;

#ifdef USE_OMP
#pragma omp parallel for schedule(static) default(shared)
#endif
		// TODO - use blas for this?
		for( int i = 0; i < x.size(); i++ )
		{
			x(i) *= (bNeg(i) + sqrt( bSquared(i) + a(i)*c4(i))) / (a2(i));
		}

		// Figure out delta norm = | x0 - x |
		xOld -= x;
		deltaNorm = xOld.norm2();
		numIters++;

		if( time(NULL) - lastShoutTime > 2 )
		{
			lastShoutTime = time(NULL);
			cout << "Took " << numIters << " so far.." << endl;

			// Taking too long..give some feedback
			Apos.write("TEMP_nqpApos.matrix");
			Aneg4.write("TEMP_nqpAneg4.matrix");
			b.write("TEMP_nqpB.matrix");
			x.write("TEMP_nqpX.matrix");
		}
	}

	return numIters;
}

void copy_submatrix( MATRIX& d, MATRIX& s )
{
	assert( s.rows() <= d.rows() );
	assert( s.cols() <= d.cols() );

#ifdef USE_OMP
#pragma omp parallel for schedule(static) default(shared)
#endif
	for( int i = 0; i < s.rows(); i++ )
	for( int j = 0; j < s.cols(); j++ )
		d(i,j) = s(i,j);
}

void GreedyCubop::runNQP(
			TrainingSet& trainingSet, // list of q vectors
			VECTOR& trainingForces,	// r*T vector of ground-truth forces. 'b' in the paper.
			Real relErrTol,	// TOL
			int maxNumPoints,	// some sane limit, for overnight runs
			int numCandsPerIter,	// |C|
			bool keepPosWeights
			)
{
	//----------------------------------------
	//  Convenience constants
	//----------------------------------------
	const int T = trainingSet.size();
	assert( T > 0 );
	_r = trainingSet[0]->size();
	
	//----------------------------------------
	//  Calculate scalings for normalized training,
	//  and scale b.
	//----------------------------------------
	VECTOR b = trainingForces;
	vector<Real> inverseForceNorms;
	normalizeTrainingForces( _r, b, inverseForceNorms );

	vector<int> selectedPoints;
	VECTOR residual( b );
	Real bNorm = b.norm2();
	cout << "bNorm = " << bNorm << " should be sqrt(T) = " << sqrt(T) << endl;

	//----------------------------------------
	//  Main loop
	//----------------------------------------

	printf("Entering cubop loop");
	printf("\n");
	VECTOR nqpB(0);
	MATRIX nqpApos(0,0);
	MATRIX nqpAneg4(0,0);
	VECTOR w(0);
	Real relErr = 2 * relErrTol;

	// For speed (in calculating the residual), we still want to store all training columns
	printf("Allocating matrix of %d x %d x %d bytes per Real = %d = %f MB. If this is too large, it could core-dump here.", _r*T, maxNumPoints, sizeof(Real), _r*T*maxNumPoints*sizeof(Real), _r*T*maxNumPoints*sizeof(Real) / 1024.0 / 1024.0 );
	printf("\n");
	MATRIX A( _r*T, maxNumPoints );
	A.clear();
	printf("Matrix allocation successful! No core-dump - OK.");
	printf("\n");

	VECTOR Aw( _r*T );

	while( relErr > relErrTol && selectedPoints.size() < maxNumPoints && selectedPoints.size() < numTotalPoints() )
	{
		//----------------------------------------
		//  Select candidate points for this iter
		//----------------------------------------
		vector<int> candPointIds;
		selectCandidatePoints( selectedPoints, numCandsPerIter, candPointIds );

		//----------------------------------------
		//  Find the candidate that's most fit, ie. highest dot prodct with residual
		//  NOTE: this part can be easily paralelized! then we can do more key tets per iteration
		//----------------------------------------
		int newPoint = findMostFitPoint( candPointIds, residual, trainingSet, inverseForceNorms );
		selectedPoints.push_back( newPoint );

		// We'll need its column for all samples (not just the subset)
		VECTOR newPointCol( _r*T );
		evalPointColumn( newPoint, trainingSet, inverseForceNorms, newPointCol );
		const int n = selectedPoints.size();

		// Add this column to the A matrix
		A.setVector( newPointCol, 0, n-1 );

		// Grow our NQP matrices by one

		MATRIX oldNqpApos = nqpApos;
		MATRIX oldNqpAneg = nqpAneg4;
		VECTOR oldNqpB = nqpB;
		VECTOR oldW = w;

		nqpApos.resizeAndWipe( n, n );
		nqpAneg4.resizeAndWipe( n, n );
		nqpB.resizeAndWipe( n );
		w.resizeAndWipe( n );

		// Copy old values to new buffers

		copy_submatrix( nqpApos, oldNqpApos );
		copy_submatrix( nqpAneg4, oldNqpAneg );
		memcpy( nqpB.data(), oldNqpB.data(), sizeof(Real)*oldNqpB.size() );
		memcpy( w.data(), oldW.data(), sizeof(Real)*oldW.size() );

		// Set new entries of A^T*A

		// A' * newCol
		VECTOR AtCol( n );
		A.subMatrixMultiplyInplace( newPointCol, AtCol, _r*T, n, true );

		// Set the new column in A'*A = nqpA
		for( int i = 0; i < n; i++ )
		{
			Real val = 2.0 * AtCol(i);
			if( val > 0 )
			{
				nqpApos( i, n-1 ) = val;
				nqpApos( n-1, i ) = val;
			}
			else
			{
				val *= -4;
				nqpAneg4( i, n-1 ) = val;
				nqpAneg4( n-1, i ) = val;
			}
		}

		// Update A^T*b
		nqpB( n-1 ) = b * newPointCol * -2.0;

		// Make sure we have no non-pos weights! The NQP algo relies on that.
		for( int i = 0; i < w.size(); i++ )
		{
			// Not sure which is better yet..
			if( keepPosWeights )
			{
				if( w(i) <= 0.0 ) w(i) = 1.0;
			}
			else
				w(i) = 1.0;
		}

		//----------------------------------------
		//	Now calculate a new residual
		//  Do NNLS with a new subset using NQP algo
		//----------------------------------------

		int numNqpIters = nqp_iteration( nqpApos, nqpAneg4, nqpB, w, 1e-8 );
		cout << SDUMP(numNqpIters) << "\t";

		//----------------------------------------
		//  Actually update resiudal and compute training error
		//----------------------------------------
		A.subMatrixMultiplyInplace( w, Aw, _r*T, n, false );

		residual = b;
		residual -= Aw;

		relErr = residual.norm2() / bNorm;	// TEMP

		//----------------------------------------
		//  Hand it off
		//----------------------------------------
		handleCubature( selectedPoints, w, relErr );
	}
}

