#ifndef GaussSystemWrapper_H
#define GaussSystemWrapper_H

#include "TypeDefs.h"

class GaussSystemWrapperImpl; // Need to hide the implementation to speed up compile

class GaussSystemWrapper
{
public:
    GaussSystemWrapper(const MatrixXd &V, const MatrixXi &T, double density, double poissons_ratio, double youngs_modulus) {
        m_impl = new m_impl(V, T, density, poissons_ratio, youngs_modulus);
    }

    ~GaussSystemWrapper() {
        delete m_impl;
    }

    Vector3d getVertPos(int index) { return m_impl->getVertPos(); }
    Eigen::Map<Eigen::VectorXd> getMappedq() { return m_impl->getMappedq(); }

    SparseMatrix<double> getMassMatrix() { return m_impl->getMassMatrixImpl(); }
    SparseMatrix<double> getStiffnessMatrix() { return m_impl->getStiffnessMatrixImpl(); }

    VectorXd getInternalForceVector() { return m_impl->getInternalForceVectorImpl(); }
    double getStrainEnergy() { return m_impl->getStrainEnergy(); }
    VectorXd getStrainEnergyPerElement() { return m_impl->getStrainEnergyPerElement(); }

    void getEnergyAndForcesForTets(const VectorXi &indices, VectorXd &energies, MatrixXd &forces) { m_impl->getEnergyForTets(indices, energies, forces); }
    void getVertIndicesForTets(const VectorXi &tet_indices, VectorXi &vert_indices) { m_impl->getVertIndicesForTets(tet_indices, vert_indices); }


private:
    GaussSystemWrapperImpl *m_impl;
};


#endif