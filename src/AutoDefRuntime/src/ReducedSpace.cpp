#include "ReducedSpace.h"

template <typename ReducedSpaceImpl, typename MatrixType>
VectorXd ReducedSpace<ReducedSpaceImpl, MatrixType>::encode(const VectorXd &q) {
    return m_impl.encode(q);
}

template <typename ReducedSpaceImpl, typename MatrixType>
VectorXd ReducedSpace<ReducedSpaceImpl, MatrixType>::decode(const VectorXd &z) {
    return m_impl.decode(z);
}

template <typename ReducedSpaceImpl, typename MatrixType>
VectorXd ReducedSpace<ReducedSpaceImpl, MatrixType>::sub_decode(const VectorXd &z) {
    return m_impl.sub_decode(z);
}
// Must be a better way than this crazy decltyp
template <typename ReducedSpaceImpl, typename MatrixType>
MatrixType ReducedSpace<ReducedSpaceImpl, MatrixType>::jacobian(const VectorXd &z) {
    return m_impl.jacobian(z);
}

template <typename ReducedSpaceImpl, typename MatrixType>
VectorXd ReducedSpace<ReducedSpaceImpl, MatrixType>::jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {
    // d decode / d z * q
    return m_impl.jacobian_transpose_vector_product(z, q);
}

template <typename ReducedSpaceImpl, typename MatrixType>
VectorXd ReducedSpace<ReducedSpaceImpl, MatrixType>::jacobian_vector_product(const VectorXd &z, const VectorXd &z_v) {
    return m_impl.jacobian_vector_product(z, z_v);
}

template <typename ReducedSpaceImpl, typename MatrixType>
MatrixType ReducedSpace<ReducedSpaceImpl, MatrixType>::outer_jacobian() {
    return m_impl.outer_jacobian();
}

template <typename ReducedSpaceImpl, typename MatrixType>
MatrixType ReducedSpace<ReducedSpaceImpl, MatrixType>::inner_jacobian(const VectorXd &z) { // TODO make this sparse for linear subspace?
    return m_impl.inner_jacobian(z);
}

template <typename ReducedSpaceImpl, typename MatrixType>
MatrixType ReducedSpace<ReducedSpaceImpl, MatrixType>::compute_reduced_mass_matrix(const SparseMatrix<double> &M) {
    return m_impl.compute_reduced_mass_matrix(M);
}

template <typename ReducedSpaceImpl, typename MatrixType>
double ReducedSpace<ReducedSpaceImpl, MatrixType>::get_energy(const VectorXd &z) {
    return m_impl.get_energy(z);
}

template <typename ReducedSpaceImpl, typename MatrixType>
VectorXd ReducedSpace<ReducedSpaceImpl, MatrixType>::get_energy_gradient(const VectorXd &z) {
    return m_impl.get_energy_gradient(z);
}

template <typename ReducedSpaceImpl, typename MatrixType>
void ReducedSpace<ReducedSpaceImpl, MatrixType>::get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights) {
    return m_impl.get_cubature_indices_and_weights(z, indices, weights);
}

template <typename ReducedSpaceImpl, typename MatrixType>
VectorXd ReducedSpace<ReducedSpaceImpl, MatrixType>::cubature_vjp(const VectorXd &z, VectorXd &energies) {
    return m_impl.cubature_vjp(z, energies);
}

template class ReducedSpace<LinearSpaceImpl<MatrixXd>, MatrixXd>;
template class ReducedSpace<LinearSpaceImpl<SparseMatrix<double>>, SparseMatrix<double>>;
template class ReducedSpace<AutoEncoderSpaceImpl, MatrixXd>;
