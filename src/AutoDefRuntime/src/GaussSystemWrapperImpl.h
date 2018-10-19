#ifndef GaussSystemWrapper_H
#define GaussSystemWrapper_H

#include "TypeDefs.h"

#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>
#include <AssemblerParallel.h>


using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

/* Tetrahedral finite elements */

//typedef physical entities I need

typedef PhysicalSystemFEM<double, NeohookeanTet> NeohookeanTets;

typedef World<double, 
                        std::tuple<PhysicalSystemParticleSingle<double> *, NeohookeanTets *>,
                        std::tuple<ForceSpringFEMParticle<double> *>,
                        std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
typedef TimeStepperEulerImplicitLinear<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>,
AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepper;

class GaussSystemWrapperImpl
{
public:
    GaussSystemWrapperImpl(const MatrixXd &V, const MatrixXi &T, double density, double poissons_ratio, double youngs_modulus);
    ~GaussSystemWrapperImpl();

    Vector3d getVertPos(int index);
    Eigen::Map<Eigen::VectorXd> getMappedq();

    SparseMatrix<double> getMassMatrixImpl();
    SparseMatrix<double> getStiffnessMatrixImpl();

    VectorXd getInternalForceVectorImpl();
    double getStrainEnergy();
     VectorXd getStrainEnergyPerElement();

    void getEnergyAndForcesForTets(const VectorXi &indices, VectorXd &energies, MatrixXd &forces);
    void getVertIndicesForTets(const VectorXi &tet_indices, VectorXi &vert_indices);

private:
    MyWorld m_world;
    NeohookeanTets *m_tets;

    int m_element_size;
    int m_num_elements;
};

#endif