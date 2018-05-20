#include "LinearSpace.h"

template<typename MatrixType>
LinearSpaceImpl<MatrixType>::LinearSpaceImpl(const MatrixType &U) : m_U(U) {
    std::cout<<"U rows: " << U.rows() << std::endl;
    std::cout<<"U cols: " << U.cols() << std::endl;

    m_inner_jac.resize(U.cols(), U.cols());
    m_inner_jac.setIdentity();
}

template<typename MatrixType>
VectorXd LinearSpaceImpl<MatrixType>::encode(const VectorXd &q) {
    return m_U.transpose() * q;
}

template<typename MatrixType>
VectorXd LinearSpaceImpl<MatrixType>::decode(const VectorXd &z) {
    return m_U * z;
}

template<typename MatrixType>
VectorXd LinearSpaceImpl<MatrixType>::sub_decode(const VectorXd &z) {
    return z;
}

template<typename MatrixType>
MatrixType LinearSpaceImpl<MatrixType>::jacobian(const VectorXd &z) { return m_U; }

template<typename MatrixType>
VectorXd LinearSpaceImpl<MatrixType>::jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) { // TODO: do this without copy?
    return q;
}

template<typename MatrixType>
VectorXd LinearSpaceImpl<MatrixType>::jacobian_vector_product(const VectorXd &z, const VectorXd &z_v) {
    return z_v;
}

template<typename MatrixType>
MatrixType LinearSpaceImpl<MatrixType>::outer_jacobian() {
    return m_U;
}

template<typename MatrixType>
MatrixType LinearSpaceImpl<MatrixType>::inner_jacobian(const VectorXd &z) {
    return m_inner_jac;
}

template<typename MatrixType>
MatrixType LinearSpaceImpl<MatrixType>::compute_reduced_mass_matrix(const MatrixType &M) {
    return m_U.transpose() * M * m_U;
}

template<typename MatrixType>
double LinearSpaceImpl<MatrixType>::get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}

template<typename MatrixType>
VectorXd LinearSpaceImpl<MatrixType>::get_energy_gradient(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}

template<typename MatrixType>
void LinearSpaceImpl<MatrixType>::get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights) {std::cout << "Reduced energy not implemented!" << std::endl;}

template<typename MatrixType>
VectorXd LinearSpaceImpl<MatrixType>::cubature_vjp(const VectorXd &z, VectorXd &energies) {std::cout << "Reduced energy not implemented!" << std::endl;}

template class LinearSpaceImpl<MatrixXd>;
template class LinearSpaceImpl<SparseMatrix<double>>;
