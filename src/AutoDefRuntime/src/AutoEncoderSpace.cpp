#include "AutoEncoderSpace.h"
#include "AutoDefUtils.h"

// Boost
#include <boost/filesystem.hpp>

// IGL
#include <igl/get_seconds.h>
#include <igl/readDMAT.h>

auto tf_dtype = tf::DT_DOUBLE;
typedef double tf_dtype_type;

AutoEncoderSpaceImpl::AutoEncoderSpaceImpl(fs::path tf_models_root, json integrator_config) {
    m_enc_dim = integrator_config["ae_encoded_dim"];
    m_sub_q_size = integrator_config["ae_decoded_dim"];

    fs::path decoder_path = tf_models_root / "decoder.pb";
    fs::path decoder_jac_path = tf_models_root / "decoder_jac.pb";
    fs::path decoder_vjp_path = tf_models_root / "decoder_vjp.pb";
    fs::path decoder_jvp_path = tf_models_root / "decoder_jvp.pb";
    fs::path encoder_path = tf_models_root / "encoder.pb";

    // Decoder
    tf::Status status = tf::NewSession(tf::SessionOptions(), &m_decoder_session);
    checkStatus(status);
    tf::GraphDef decoder_graph_def;
    status = ReadBinaryProto(tf::Env::Default(), decoder_path.string(), &decoder_graph_def);
    checkStatus(status);
    status = m_decoder_session->Create(decoder_graph_def);
    checkStatus(status);

    // Decoder Jacobian
    status = tf::NewSession(tf::SessionOptions(), &m_decoder_jac_session);
    checkStatus(status);
    tf::GraphDef decoder_jac_graph_def;
    status = ReadBinaryProto(tf::Env::Default(), decoder_jac_path.string(), &decoder_jac_graph_def);
    checkStatus(status);
    status = m_decoder_jac_session->Create(decoder_jac_graph_def);
    checkStatus(status);

    // Decoder vjp
    status = tf::NewSession(tf::SessionOptions(), &m_decoder_vjp_session);
    checkStatus(status);
    tf::GraphDef decoder_vjp_graph_def;
    status = ReadBinaryProto(tf::Env::Default(), decoder_vjp_path.string(), &decoder_vjp_graph_def);
    checkStatus(status);
    status = m_decoder_vjp_session->Create(decoder_vjp_graph_def);
    checkStatus(status);

    // // Decoder jvp
    status = tf::NewSession(tf::SessionOptions(), &m_decoder_jvp_session);
    checkStatus(status);
    tf::GraphDef decoder_jvp_graph_def;
    status = ReadBinaryProto(tf::Env::Default(), decoder_jvp_path.string(), &decoder_jvp_graph_def);
    checkStatus(status);
    status = m_decoder_jvp_session->Create(decoder_jvp_graph_def);
    checkStatus(status);


    // Encoder
    status = tf::NewSession(tf::SessionOptions(), &m_encoder_session);
    checkStatus(status);
    tf::GraphDef encoder_graph_def;
    status = ReadBinaryProto(tf::Env::Default(), encoder_path.string(), &encoder_graph_def);
    checkStatus(status);
    status = m_encoder_session->Create(encoder_graph_def);
    checkStatus(status);

    // Last layer
    fs::path U_path = tf_models_root / "../pca_results/ae_pca_components.dmat";
    igl::readDMAT(U_path.string(), m_U);

    // Currently disabled
    // if(integrator_config["use_reduced_energy"]) {
    //     fs::path energy_model_path = tf_models_root / "energy_model.pb";

    //     status = tf::NewSession(tf::SessionOptions(), &m_direct_energy_model_session);
    //     checkStatus(status);
    //     tf::GraphDef energy_model_graph_def;
    //     status = ReadBinaryProto(tf::Env::Default(), energy_model_path.string(), &energy_model_graph_def);
    //     checkStatus(status);
    //     status = m_direct_energy_model_session->Create(energy_model_graph_def);
    //     checkStatus(status);
    // }

    EnergyMethod energy_method = energy_method_from_integrator_config(integrator_config);

    if(energy_method == PRED_WEIGHTS_L1) {
        fs::path discrete_energy_model_path = tf_models_root / "l1_discrete_energy_model.pb";

        status = tf::NewSession(tf::SessionOptions(), &m_discrete_energy_model_session);
        checkStatus(status);
        tf::GraphDef discrete_energy_model_graph_def;
        status = ReadBinaryProto(tf::Env::Default(), discrete_energy_model_path.string(), &discrete_energy_model_graph_def);
        checkStatus(status);
        status = m_discrete_energy_model_session->Create(discrete_energy_model_graph_def);
        checkStatus(status);

        // Disabled since we don't currently need it.
        fs::path cubature_vjp_model_path = tf_models_root / "l1_discrete_energy_model_vjp.pb";

        status = tf::NewSession(tf::SessionOptions(), &m_cubature_vjp_session);
        checkStatus(status);
        tf::GraphDef cubature_vjp_graph_def;
        status = ReadBinaryProto(tf::Env::Default(), cubature_vjp_model_path.string(), &cubature_vjp_graph_def);
        checkStatus(status);
        status = m_cubature_vjp_session->Create(cubature_vjp_graph_def);
        checkStatus(status);
    }
    else if (energy_method == PRED_DIRECT) {
        fs::path energy_model_path = tf_models_root / "direct_energy_model.pb";

        status = tf::NewSession(tf::SessionOptions(), &m_direct_energy_model_session);
        checkStatus(status);
        tf::GraphDef energy_model_graph_def;
        status = ReadBinaryProto(tf::Env::Default(), energy_model_path.string(), &energy_model_graph_def);
        checkStatus(status);
        status = m_direct_energy_model_session->Create(energy_model_graph_def);
        checkStatus(status);
    }
}

VectorXd AutoEncoderSpaceImpl::encode(const VectorXd &q) {
    VectorXd sub_q = m_U.transpose() * q;

    tf::Tensor sub_q_tensor(tf_dtype, tf::TensorShape({1, sub_q.size()}));
    std::copy_n(sub_q.data(), sub_q.size(), sub_q_tensor.flat<tf_dtype_type>().data());

    std::vector<tf::Tensor> z_outputs;
    tf::Status status = m_encoder_session->Run({{"encoder_input:0", sub_q_tensor}},
                               {"output_node0:0"}, {}, &z_outputs);

    VectorXd res(m_enc_dim);
    std::copy_n(z_outputs[0].flat<tf_dtype_type>().data(), res.size(), res.data());

    return res;
}

VectorXd AutoEncoderSpaceImpl::sub_decode(const VectorXd &z) {
    tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim}));
    std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

    std::vector<tf::Tensor> sub_q_outputs;
    tf::Status status = m_decoder_session->Run({{"decoder_input:0", z_tensor}},
                               {"output_node0:0"}, {}, &sub_q_outputs);

    VectorXd res(m_U.cols());
    std::copy_n(sub_q_outputs[0].flat<tf_dtype_type>().data(), res.size(), res.data());

    return res;
}

VectorXd AutoEncoderSpaceImpl::decode(const VectorXd &z) {
    return m_U * sub_decode(z);
}

MatrixXd AutoEncoderSpaceImpl::jacobian(const VectorXd &z) {
    return m_U * inner_jacobian(z);
}

MatrixXd AutoEncoderSpaceImpl::outer_jacobian() {
    return m_U;
}

MatrixXd AutoEncoderSpaceImpl::inner_jacobian(const VectorXd &z) {
    // //Analytic with jvp 
    MatrixXd sub_jac(m_sub_q_size, z.size()); // just m_U.cols()?
    MatrixXd zs = MatrixXd::Identity(z.size(), z.size());

    // #pragma omp parallel for
    for(int i = 0; i < z.size(); i++) {
        sub_jac.col(i) = jacobian_vector_product(z, zs.row(i));
    }

    // // Analytic with vjp - (slower than FD probably)
    // // MatrixXd sub_jac(m_sub_q_size, z.size()); // just m_U.cols()?
    // // MatrixXd subqs = MatrixXd::Identity(m_sub_q_size, m_sub_q_size);

    // // for(int i = 0; i < m_sub_q_size; i++) {
    // //     sub_jac.row(i) = jacobian_transpose_vector_product(z, subqs.row(i));
    // // }

    return sub_jac;

}


VectorXd AutoEncoderSpaceImpl::jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &sub_q) { // TODO: do this without copy?
    tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); 
    std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

    tf::Tensor sub_q_tensor(tf_dtype, tf::TensorShape({1, sub_q.size()})); 
    std::copy_n(sub_q.data(), sub_q.size(), sub_q_tensor.flat<tf_dtype_type>().data());

    std::vector<tf::Tensor> vjp_outputs;
    tf::Status status = m_decoder_vjp_session->Run({{"decoder_input:0", z_tensor}, {"input_v:0", sub_q_tensor}},
                               {"vjp/dense_decode_layer_0/MatMul_grad/MatMul:0"}, {}, &vjp_outputs); // TODO get better names
    
    VectorXd res(m_enc_dim);
    std::copy_n(vjp_outputs[0].flat<tf_dtype_type>().data(), res.size(), res.data());

    return res;
}

VectorXd AutoEncoderSpaceImpl::jacobian_vector_product(const VectorXd &z, const VectorXd &z_v) {
    tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); 
    std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

    tf::Tensor z_v_tensor(tf_dtype, tf::TensorShape({1, z_v.size()})); 
    std::copy_n(z_v.data(), z_v.size(), z_v_tensor.flat<tf_dtype_type>().data());

    std::vector<tf::Tensor> jvp_outputs; 
    tf::Status status = m_decoder_jvp_session->Run({{"decoder_input:0", z_tensor}, {"input_z_v:0", z_v_tensor}},
                               {"jvp/gradients/decoder_output_layer/MatMul_grad/MatMul_grad/MatMul:0"}, {}, &jvp_outputs);

    VectorXd res(m_sub_q_size);
    std::copy_n(jvp_outputs[0].flat<tf_dtype_type>().data(), res.size(), res.data());

    return res;
}

MatrixXd AutoEncoderSpaceImpl::compute_reduced_mass_matrix(const MatrixXd &M) {
    return m_U.transpose() * M * m_U;
}

double AutoEncoderSpaceImpl::get_energy(const VectorXd &z) {
    tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); 
    std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

    std::vector<tf::Tensor> q_outputs;
    tf::Status status = m_direct_energy_model_session->Run({{"energy_model_input:0", z_tensor}},
                               {"output_node0:0"}, {}, &q_outputs);

    auto energy_tensor_mapped = q_outputs[0].tensor<tf_dtype_type, 2>();

    return energy_tensor_mapped(0,0);
}

VectorXd AutoEncoderSpaceImpl::get_energy_gradient(const VectorXd &z) {
    VectorXd energy_grad(z.size()); 
    double t = 5e-05;
    for(int i = 0; i < z.size(); i++) {
        VectorXd dz_pos(z);
        VectorXd dz_neg(z);
        dz_pos[i] += t;
        dz_neg[i] -= t;
        energy_grad[i] = (get_energy(dz_pos) - get_energy(dz_neg)) / (2.0 * t);
    }
    return energy_grad;
}

void AutoEncoderSpaceImpl::get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights) {
    double start_time = igl::get_seconds();
    tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim}));
    std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

    std::vector<tf::Tensor> energy_weight_outputs;
    tf::Status status = m_discrete_energy_model_session->Run({{"energy_model_input:0", z_tensor}},
                               {"output_node0:0"}, {}, &energy_weight_outputs);
    
    auto energy_weight_tensor_mapped = energy_weight_outputs[0].tensor<tf_dtype_type, 2>();
    double predict_time = igl::get_seconds() - start_time;

    double eps = 0.01;
    int n_tets_used = 0;

    std::vector<int> indices_vec;
    indices_vec.reserve(100);
    for(int i = 0; i < energy_weight_tensor_mapped.size(); i++) {
        double energy_weight_i = energy_weight_tensor_mapped(0, i);
        if(energy_weight_i > eps) {
            indices_vec.push_back(i);
            n_tets_used++;
        }
    }

    indices.resize(n_tets_used);
    weights.resize(n_tets_used);

    for(int i = 0; i < n_tets_used; i++) {
        int idx = indices_vec[i];
        indices[i] = idx;
        weights[i] = energy_weight_tensor_mapped(0, idx);
    }

    double copy_time = igl::get_seconds() - start_time - predict_time;
}

VectorXd AutoEncoderSpaceImpl::cubature_vjp(const VectorXd &z, VectorXd &energies) {
    tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim}));
    std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

    tf::Tensor energies_tensor(tf_dtype, tf::TensorShape({1, energies.size()}));
    std::copy_n(energies.data(), energies.size(), energies_tensor.flat<tf_dtype_type>().data());

    std::vector<tf::Tensor> vjp_outputs;
    tf::Status status = m_cubature_vjp_session->Run({{"energy_model_input:0", z_tensor}, {"input_v:0", energies_tensor}},
                               {"vjp/dense_decode_layer_0/MatMul_grad/MatMul:0"}, {}, &vjp_outputs);

    VectorXd res(m_enc_dim);
    std::copy_n(vjp_outputs[0].flat<tf_dtype_type>().data(), res.size(), res.data());
    return res;
}

void AutoEncoderSpaceImpl::checkStatus(const tf::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(1);
  }
}
