#include "GaussSystemWrapperImpl.h"

GaussSystemWrapperImpl::GaussSystemWrapperImpl(const MatrixXd &V, const MatrixXi &T, double density, double poissons_ratio, double youngs_modulus) {
    m_tets = new NeohookeanTets(V,T);

    m_element_size = T.cols();
    m_num_elements = T.rows();

    for(auto element: m_tets->getImpl().getElements()) {
        element->setDensity(density);
        element->setParameters(youngs_modulus, poissons_ratio);   
    }

    m_world.addSystem(m_tets);
    m_world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
}


Vector3d GaussSystemWrapperImpl::getVertPos(int index){
    return PosFEM<double>(&m_tets->getQ()[index], index, &m_tets->getImpl().getV())(m_world.getState());
}

Eigen::Map<Eigen::VectorXd> GaussSystemWrapperImpl::getMappedq(){
    return mapDOFEigen(m_tets->getQ(), m_world);
}


SparseMatrix<double> GaussSystemWrapperImpl::getMassMatrixImpl(){
    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > M_asm;
    getMassMatrix(M_asm, m_world);
    return *M_asm;
}

SparseMatrix<double> GaussSystemWrapperImpl::getStiffnessMatrixImpl(){
    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > K_asm;
    getStiffnessMatrix(K_asm, m_world);
    return *K_asm;
}


VectorXd GaussSystemWrapperImpl::getInternalForceVectorImpl(){
    AssemblerParallel<double, AssemblerEigenVector<double> > internal_force_asm;
    getInternalForceVector(internal_force_asm, *m_tets, m_world);
    return *internal_force_asm;
}

double GaussSystemWrapperImpl::getStrainEnergy(){
    return m_tets->getStrainEnergy(m_world.getState());
}

VectorXd GaussSystemWrapperImpl::getStrainEnergyPerElement(){
    return m_tets->getImpl().getStrainEnergyPerElement(m_world.getState());
}


void GaussSystemWrapperImpl::getEnergyAndForcesForTets(const VectorXi &indices, VectorXd &energies, MatrixXd &forces){
    int num_samples = indices.size();

    energies = VectorXd(num_samples);
    forces = MatrixXd(num_samples, m_element_size * 3);

    #pragma omp parallel for num_threads(4)
    for(int i = 0; i < num_samples; i++) {
        int tet_index = indices[i];

        energies[i] = m_tets->getImpl().getElement(tet_index)->getStrainEnergy(m_world.getState());

        VectorXd force(m_element_size * 3);
        m_tets->getImpl().getElement(tet_index)->getInternalForce(force, m_world.getState());
        forces.row(i) = force;
    }    
}

void GaussSystemWrapperImpl::getVertIndicesForTets(const VectorXi &tet_indices, VectorXi &vert_indices){
    vert_indices = VectorXi(tet_indices.size() * m_element_size);

    // TODO parallel?
    for (int i = 0; i < tet_indices.size(); i++) {
        int tet_index = tet_indices[i];
        for(int j = 0; j < m_element_size; j++) {
            int vert_x_index = m_tets->getImpl().getElement(tet_index)->getQDOFList()[j]->getGlobalId();
            for(int k = 0; k < 3; k++) {
                vert_indices[i * 3 + k] = vert_x_index + k;
            }
        }
    }
}
