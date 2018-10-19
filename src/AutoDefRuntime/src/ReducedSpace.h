#ifndef ReducedSpace_H
#define ReducedSpace_H

#include "TypeDefs.h"
#include "AutoEncoderSpace.h"
#include "LinearSpace.h"

template <typename ReducedSpaceImpl, typename MatrixType>
class ReducedSpace
{
public:
    template<typename ...Params> // TODO necessary?
    ReducedSpace(Params ...params) : m_impl(params...) {}
    // ReducedSpace(Params ...params);
    // ReducedSpace();

    VectorXd encode(const VectorXd &q);
    VectorXd decode(const VectorXd &z);
    VectorXd sub_decode(const VectorXd &z);
    // Must be a better way than this crazy decltyp
    MatrixType jacobian(const VectorXd &z);
    VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q);
    VectorXd jacobian_vector_product(const VectorXd &z, const VectorXd &z_v);
    MatrixType outer_jacobian();
    MatrixType inner_jacobian(const VectorXd &z);// TODO make this sparse for linear subspac;
    MatrixType compute_reduced_mass_matrix(const SparseMatrix<double> &M);
    double get_energy(const VectorXd &z);
    VectorXd get_energy_gradient(const VectorXd &z);
    void get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights);
    VectorXd cubature_vjp(const VectorXd &z, VectorXd &energies);

private:
    ReducedSpaceImpl m_impl;
};

typedef ReducedSpace<LinearSpaceImpl<MatrixXd>, MatrixXd> LinearSpace;
typedef ReducedSpace<LinearSpaceImpl<SparseMatrix<double>>, SparseMatrix<double>> SparseConstraintSpace;
typedef ReducedSpace<AutoEncoderSpaceImpl, MatrixXd> AutoencoderSpace;

#endif
