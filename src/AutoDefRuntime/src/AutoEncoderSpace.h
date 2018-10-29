#ifndef AutoEncoderSpaceImpl_H
#define AutoEncoderSpaceImpl_H

// Before change 58s to compile

// Autodef
#include "TypeDefs.h"
#include "AutoDefUtils.h"
// Third
#include <tensorflow/core/public/session.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

using json = nlohmann::json;
namespace tf = tensorflow;
namespace fs = boost::filesystem;

class AutoEncoderSpaceImpl
{
public:
    AutoEncoderSpaceImpl(fs::path tf_models_root, json integrator_config);
    VectorXd encode(const VectorXd &q);
    VectorXd sub_decode(const VectorXd &z);
    VectorXd decode(const VectorXd &z);
    MatrixXd jacobian(const VectorXd &z);
    MatrixXd outer_jacobian();
    MatrixXd inner_jacobian(const VectorXd &z);
    VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &sub_q);
    VectorXd jacobian_vector_product(const VectorXd &z, const VectorXd &z_v);
    MatrixXd compute_reduced_mass_matrix(const MatrixXd &M);
    double get_energy(const VectorXd &z);
    VectorXd get_energy_gradient(const VectorXd &z);
    void get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights);
    VectorXd cubature_vjp(const VectorXd &z, VectorXd &energies);
    void checkStatus(const tf::Status& status);

private:
    int m_enc_dim;
    int m_sub_q_size;

    MatrixXd m_U;

    tf::Session* m_decoder_session;
    tf::Session* m_decoder_jac_session;
    tf::Session* m_decoder_vjp_session;
    tf::Session* m_decoder_jvp_session;
    tf::Session* m_encoder_session;
    tf::Session* m_direct_energy_model_session;
    tf::Session* m_discrete_energy_model_session;
    tf::Session* m_cubature_vjp_session;
};

#endif