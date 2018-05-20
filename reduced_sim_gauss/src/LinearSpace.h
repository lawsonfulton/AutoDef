#ifndef LinearSpace_H
#define LinearSpace_H

#include <iostream>
#include "TypeDefs.h"

template <typename MatrixType>
class LinearSpaceImpl
{
public:
    LinearSpaceImpl(const MatrixType &U);

    VectorXd encode(const VectorXd &q);
    VectorXd decode(const VectorXd &z);
    VectorXd sub_decode(const VectorXd &z);
    MatrixType jacobian(const VectorXd &z);
    VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q);
    VectorXd jacobian_vector_product(const VectorXd &z, const VectorXd &z_v);
    MatrixType outer_jacobian();
    MatrixType inner_jacobian(const VectorXd &z);
    MatrixType compute_reduced_mass_matrix(const MatrixType &M);
    double get_energy(const VectorXd &z);
    VectorXd get_energy_gradient(const VectorXd &z);
    void get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights);
    VectorXd cubature_vjp(const VectorXd &z, VectorXd &energies);
private:
    MatrixType m_U;
    MatrixType m_inner_jac;
};

#endif