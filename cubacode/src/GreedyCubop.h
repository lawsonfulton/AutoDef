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

/**
 * Abstract base-class for the greedy, subset-training cubature optimization algorithm.
 * Extend this to do cubature optimization for custom discretizations.
 */

#ifndef __GREEDY_CUBOP_H__
#define __GREEDY_CUBOP_H__

#include <string>
#include <vector>
#include "VECTOR.h"
#include "TYPES.h"
#include "MERSENNETWISTER.h"

typedef std::vector<VECTOR*> TrainingSet;

class GreedyCubop
{
	public:

	GreedyCubop();

	void run(
			TrainingSet& trainingSet, // list of q vectors
			VECTOR& trainingForces,	// r*T vector of ground-truth forces. 'b' in the paper.
			Real relErrTol,	// TOL
			int maxNumPoints,	// some sane limit, for overnight runs
			int numCandsPerIter,	// |C|
			int itersPerFullNNLS,	// r/2 in the paper
			int numSamplesPerSubtrain	// T_s
			);

	// The algorithm using NQP instead of NNLS, so we don't need to do subset training.
	void runNQP(
			TrainingSet& trainingSet, // list of q vectors
			VECTOR& trainingForces,	// r*T vector of ground-truth forces. 'b' in the paper.
			Real relErrTol,	// TOL
			int maxNumPoints,	// some sane limit, for overnight runs
			int numCandsPerIter,	// |C|
			bool keepPosWeights
			);

	/**
	 * Call this to reproduce results
	 */
	void randomSeed( int seed ) {
		_rand.seed( seed );
	}

	private:

	MERSENNETWISTER _rand;

	// The 1/||f||'s for the original trainingForces vector.
	// Set by run()
	std::vector<Real> _inverseForceNorms;	

	// Reduced rank of model.
	// Set by run()
	int _r;

	int findMostFitPoint( std::vector<int>& candidatePoints, VECTOR& residual, TrainingSet& trainingSet, vector<Real>& inverseForceNorms );

	void gatherRandomUnusedPoints( std::vector<int>& usedPoints, std::vector<int>& candidatesOut, int maxCands );

	void gatherUnusedPoints( std::vector<int>& usedPoints, std::vector<int>& candidatesOut );

	protected:

	/**
	 * Return the total number of points we have to choose from.
	 * For example, for FEM piecewise-linear tetrahedra elements, this is just the number of tets.
	 */
	virtual int numTotalPoints() = 0;

	/**
	 * The main method that subclasses must implement. This should evaluate the reduced force density (or just the force, since
	 * the volume/area is constant anyway) at the given point.
	 *
	 *	pointId - an index in [0, getNumTotalPoints()). Implementations must map this index deterministically to a cubature point.
	 */
	virtual void evalPointForceDensity( int pointId, VECTOR& q, VECTOR& gOut ) = 0;

	/**
	 * At the end of each iteration (_not_ a sub-train iteration), when some cubature has been optimized, this will be called.
	 * Implementations should probably overwrite this and write the cubature out to file or something.
	 */
	virtual void handleCubature( std::vector<int>& selectedPoints, VECTOR& weights, Real relErr );

	/**
	 * This evaluates the column of g^i_t force density vectors at the given point, scaled by the inverse force norms.
	 * Subclasses probably don't need to override this. This just goes through every training sample,
	 * calling evalPointForceDensity (which subclasses must implement).
	 *
	 *	pointId - an index in [0, getNumTotalPoints()). Implementations must map this index deterministically to a cubature point.
	 */
	virtual void evalPointColumn( int pointId, TrainingSet& trainingSet, vector<Real>& inverseForceNorms, VECTOR& columnOut );

	/**
	 * Subclasses may want to override this, if there's a smarter way to choose candidate points.
	 */
	virtual void selectCandidatePoints( std::vector<int>& usedPoints, int numCandidates, std::vector<int>& candidatesOut );
};

#endif
