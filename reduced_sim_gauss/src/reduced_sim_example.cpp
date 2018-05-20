#include <vector>
#include <set>

// Tensorflow
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>


//#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>
#include <AssemblerParallel.h>

#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/get_seconds.h>
#include <igl/jet.h>
#include <igl/slice.h>
#include <igl/per_corner_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/material_colors.h>
#include <igl/snap_points.h>
#include <igl/centroid.h>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <boost/filesystem.hpp>

// Optimization
#include <LBFGS.h>

// JSON
#include <json.hpp>


#include <omp.h>

#include<Eigen/SparseCholesky>

using json = nlohmann::json;
namespace fs = boost::filesystem;

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::SparseVector;
using namespace LBFGSpp;

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


// Mesh
Eigen::MatrixXd V; // Verts
Eigen::MatrixXd N; // Normals
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices
int n_dof;
double finite_diff_eps;

// Mouse/Viewer state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
Eigen::RowVector3d mesh_pos;
bool is_dragging = false;
bool per_vertex_normals = false;
int dragged_vert = 0;
int current_frame = 0;

// Parameters
bool LOGGING_ENABLED = false;
bool PLAYBACK_SIM = false;
int log_save_freq = 5;
json sim_log;
json sim_playback_json;
json timestep_info; // Kind of hack but this gets reset at the beginning of each timestep so that we have a global structure to log to
fs::path log_dir;
fs::path surface_obj_dir;
fs::path pointer_obj_dir;
fs::path tet_obj_dir;
std::ofstream log_ofstream;

enum EnergyMethod {FULL, PCR, AN08, PRED_WEIGHTS_L1, PRED_DIRECT};


EnergyMethod energy_method_from_integrator_config(const json &integrator_config) {
    try {
        if(!integrator_config.at("use_reduced_energy")) {
            return FULL;
        }

        std::string energy_method = integrator_config.at("reduced_energy_method");    

        if(energy_method == "full") return FULL;
        if(energy_method == "pcr") return PCR;
        if(energy_method == "an08") return AN08;
        if(energy_method == "pred_weights_l1") return PRED_WEIGHTS_L1;
        if(energy_method == "pred_energy_direct") return PRED_DIRECT;
    } 
    catch (nlohmann::detail::out_of_range& e){} // Didn't exist
    std::cout << "Unkown energy method." << std::endl;
    exit(1);
}

std::vector<unsigned int> get_min_verts(int axis, bool get_max = false, double tol = 0.01) {
    int dim = axis; // x
    double min_x_val = get_max ?  V.col(dim).maxCoeff() : V.col(dim).minCoeff();
    std::vector<unsigned int> min_verts;

    for(unsigned int ii=0; ii<V.rows(); ++ii) {
        if(fabs(V(ii, dim) - min_x_val) < tol) {
            min_verts.push_back(ii);
        }
    }

    return min_verts;
}

SparseMatrix<double> construct_constraints_P(const MatrixXd &V, std::vector<unsigned int> &verts) {
    // Construct constraint projection matrix
    std::cout << "Constructing constraints..." << std::endl;
    std::sort(verts.begin(), verts.end());
    int q_size = V.rows() * 3; // n dof
    int n = q_size;
    int m = n - verts.size()*3;
    SparseMatrix<double> P(m, n);
    P.reserve(VectorXi::Constant(n, 1)); // Reserve enough space for 1 non-zero per column
    int min_vert_i = 0;
    int cur_col = 0;
    for(int i = 0; i < m; i+=3){
        while(verts[min_vert_i] * 3 == cur_col) { // Note * is for vert index -> flattened index
            cur_col += 3;
            min_vert_i++;
        }
        P.insert(i, cur_col) = 1.0;
        P.insert(i+1, cur_col+1) = 1.0;
        P.insert(i+2, cur_col+2) = 1.0;
        cur_col += 3;
    }
    P.makeCompressed();
    std::cout << "Done." << std::endl;
    // -- Done constructing P
    return P;
}


void reset_world (MyWorld &world) {
        auto q = mapStateEigen(world); // TODO is this necessary?
        q.setZero();
}


// -- My integrator
template <typename ReducedSpaceImpl>
class ReducedSpace
{
public:
    template<typename ...Params> // TODO necessary?
    ReducedSpace(Params ...params) : m_impl(params...) {}
    ReducedSpace() : m_impl() {}

    VectorXd encode(const VectorXd &q) {
        return m_impl.encode(q);
    }

    VectorXd decode(const VectorXd &z) {
        return m_impl.decode(z);
    }

    VectorXd sub_decode(const VectorXd &z) {
        return m_impl.sub_decode(z);
    }
    // Must be a better way than this crazy decltyp
    decltype(auto) jacobian(const VectorXd &z) {
        return m_impl.jacobian(z);
    }

    VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {
        // d decode / d z * q
        return m_impl.jacobian_transpose_vector_product(z, q);
    }

    VectorXd jacobian_vector_product(const VectorXd &z, const VectorXd &z_v) {
        return m_impl.jacobian_vector_product(z, z_v);
    }

    decltype(auto) outer_jacobian() {
        return m_impl.outer_jacobian();
    }

    decltype(auto) inner_jacobian(const VectorXd &z) { // TODO make this sparse for linear subspace?
        return m_impl.inner_jacobian(z);
    }

    decltype(auto) compute_reduced_mass_matrix(const SparseMatrix<double> &M) {
        return m_impl.compute_reduced_mass_matrix(M);
    }

    double get_energy(const VectorXd &z) {
        return m_impl.get_energy(z);
    }

    VectorXd get_energy_gradient(const VectorXd &z) {
        return m_impl.get_energy_gradient(z);
    }

    void get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights) {
        return m_impl.get_cubature_indices_and_weights(z, indices, weights);
    }

    VectorXd cubature_vjp(const VectorXd &z, VectorXd &energies) {
        return m_impl.cubature_vjp(z, energies);
    }

private:
    ReducedSpaceImpl m_impl;
};

template <typename MatrixType>
class LinearSpaceImpl
{
public:
    LinearSpaceImpl(const MatrixType &U) : m_U(U) {
        std::cout<<"U rows: " << U.rows() << std::endl;
        std::cout<<"U cols: " << U.cols() << std::endl;

        m_inner_jac.resize(U.cols(), U.cols());
        m_inner_jac.setIdentity();
    }

    VectorXd encode(const VectorXd &q) {
        return m_U.transpose() * q;
    }

    VectorXd decode(const VectorXd &z) {
        return m_U * z;
    }

    VectorXd sub_decode(const VectorXd &z) {
        return z;
    }

    MatrixType jacobian(const VectorXd &z) { return m_U; }

    VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) { // TODO: do this without copy?
        return q;
    }

    VectorXd jacobian_vector_product(const VectorXd &z, const VectorXd &z_v) {
        return z_v;
    }

    MatrixType outer_jacobian() {
        return m_U;
    }

    MatrixType inner_jacobian(const VectorXd &z) {
        return m_inner_jac;
    }

    MatrixType compute_reduced_mass_matrix(const MatrixType &M) {
        return m_U.transpose() * M * m_U;
    }

    double get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}
    VectorXd get_energy_gradient(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}
    void get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights) {std::cout << "Reduced energy not implemented!" << std::endl;}
    VectorXd cubature_vjp(const VectorXd &z, VectorXd &energies) {std::cout << "Reduced energy not implemented!" << std::endl;}
private:
    MatrixType m_U;
    MatrixType m_inner_jac;
};

namespace tf = tensorflow;
auto tf_dtype = tf::DT_DOUBLE;
typedef double tf_dtype_type;

class AutoEncoderSpaceImpl
{
public:
    AutoEncoderSpaceImpl(fs::path tf_models_root, json integrator_config) {
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

    VectorXd encode(const VectorXd &q) {
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

    VectorXd sub_decode(const VectorXd &z) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim}));
        std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

        std::vector<tf::Tensor> sub_q_outputs;
        tf::Status status = m_decoder_session->Run({{"decoder_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &sub_q_outputs);

        VectorXd res(m_U.cols());
        std::copy_n(sub_q_outputs[0].flat<tf_dtype_type>().data(), res.size(), res.data());

        return res;
    }

    VectorXd decode(const VectorXd &z) {
        return m_U * sub_decode(z);
    }

    MatrixXd jacobian(const VectorXd &z) {
        return m_U * inner_jacobian(z);
    }

    MatrixXd outer_jacobian() {
        return m_U;
    }

    MatrixXd inner_jacobian(const VectorXd &z) {
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

    
    VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &sub_q) { // TODO: do this without copy?
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

    VectorXd jacobian_vector_product(const VectorXd &z, const VectorXd &z_v) {
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

    MatrixXd compute_reduced_mass_matrix(const MatrixXd &M) {
        return m_U.transpose() * M * m_U;
    }

    double get_energy(const VectorXd &z) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); 
        std::copy_n(z.data(), z.size(), z_tensor.flat<tf_dtype_type>().data());

        std::vector<tf::Tensor> q_outputs;
        tf::Status status = m_direct_energy_model_session->Run({{"energy_model_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &q_outputs);

        auto energy_tensor_mapped = q_outputs[0].tensor<tf_dtype_type, 2>();

        return energy_tensor_mapped(0,0);
    }

    VectorXd get_energy_gradient(const VectorXd &z) {
        VectorXd energy_grad(z.size()); 
        double t = finite_diff_eps;
        for(int i = 0; i < z.size(); i++) {
            VectorXd dz_pos(z);
            VectorXd dz_neg(z);
            dz_pos[i] += t;
            dz_neg[i] -= t;
            energy_grad[i] = (get_energy(dz_pos) - get_energy(dz_neg)) / (2.0 * t);
        }
        return energy_grad;
    }

    void get_cubature_indices_and_weights(const VectorXd &z, VectorXi &indices, VectorXd &weights) {
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
        for(int i = 0; i < T.rows(); i++) {
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

    VectorXd cubature_vjp(const VectorXd &z, VectorXd &energies) {
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

    void checkStatus(const tf::Status& status) {
      if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        exit(1);
      }
    }

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

template <typename ReducedSpaceType>
class GPLCObjective
{
public:
    GPLCObjective(
        fs::path model_root,
        json integrator_config,
        VectorXd cur_z,
        VectorXd prev_z,
        MyWorld *world,
        NeohookeanTets *tets,
        ReducedSpaceType *reduced_space ) : 
            m_cur_z(cur_z),
            m_prev_z(prev_z),
            m_world(world),
            m_tets(tets),
            m_reduced_space(reduced_space)
    {
        m_cur_sub_q = sub_dec(cur_z);
        m_prev_sub_q = sub_dec(prev_z);

        m_h = integrator_config["timestep"];
        finite_diff_eps = integrator_config["finite_diff_eps"];
        // Construct mass matrix and external forces
        getMassMatrix(m_M_asm, *m_world);

        std::cout << "Constructing reduced mass matrix..." << std::endl;
        m_M = *m_M_asm;
        m_U = m_reduced_space->outer_jacobian();
        m_UTMU = m_U.transpose() * m_M * m_U;
        std::cout << "Done." << std::endl;

        VectorXd g(m_M.cols());
        for(int i=0; i < g.size(); i += 3) {
            g[i] = 0.0;
            g[i+1] = integrator_config["gravity"];
            g[i+2] = 0.0;
        }

        m_F_ext = m_M * g;
        m_UT_F_ext = m_U.transpose() * m_F_ext;
        m_interaction_force = SparseVector<double>(m_F_ext.size());

        m_energy_method = energy_method_from_integrator_config(integrator_config);

        if(m_energy_method == PCR){
            fs::path pca_components_path("pca_results/energy_pca_components.dmat");
            fs::path sample_tets_path("pca_results/energy_indices.dmat");

            MatrixXd U;
            igl::readDMAT((model_root / pca_components_path).string(), U);

            MatrixXi tet_indices_mat;
            igl::readDMAT((model_root / sample_tets_path).string(), tet_indices_mat);
            std::cout << "Done." << std::endl;

            std::cout << "Factoring reduced energy basis..." << std::endl;
            std::cout << "n tets: " <<  tet_indices_mat.rows() << std::endl;
            std::cout << "n pca: " <<  U.cols() << std::endl;
            VectorXi tet_indices = tet_indices_mat.col(0);
            m_energy_sample_tets.resize(tet_indices.size());
            VectorXi::Map(&m_energy_sample_tets[0], tet_indices.size()) = tet_indices;
            std::sort(m_energy_sample_tets.begin(), m_energy_sample_tets.end());
            for(int i=0; i < m_energy_sample_tets.size(); i++) {
                tet_indices_mat(i, 0) = m_energy_sample_tets[i];
            }
            m_cubature_indices = tet_indices_mat.col(0); // TODO fix these duplicate structures

            // Computing the sampled verts
            // figure out the verts
            std::set<int> vert_set;
            int n_sample_tets = m_energy_sample_tets.size();
            for(int i = 0; i < m_energy_sample_tets.size(); i++) {
                int tet_index = m_energy_sample_tets[i];
                for(int j = 0; j < 4; j++) {
                    int vert_index = m_tets->getImpl().getElement(tet_index)->getQDOFList()[j]->getGlobalId();
                    for(int k = 0; k < 3; k++) {
                        vert_set.insert(vert_index + k);
                    }
                }
            }
            // Put them in a vector sorted order
            m_cubature_vert_indices = VectorXi(vert_set.size());
            int i = 0;
            for(auto vi: vert_set) {
                m_cubature_vert_indices[i++] = vi;
                // std::cout << vi << ", " << std::endl;
            }
            MatrixXd denseU = m_U;  // Need to cast to dense to support sparse in full space
            m_U_sampled = igl::slice(denseU, m_cubature_vert_indices, 1);

            // Other stuff?
            m_energy_basis = U;
            m_energy_sampled_basis = igl::slice(m_energy_basis, tet_indices_mat, 1);
            m_summed_energy_basis = m_energy_basis.colwise().sum();
            MatrixXd S_barT_S_bar = m_energy_sampled_basis.transpose() * m_energy_sampled_basis;
            // m_energy_sampled_basis_qr = m_energy_sampled_basis.fullPivHouseholderQr();
            
            m_cubature_weights = m_energy_sampled_basis * S_barT_S_bar.ldlt().solve(m_summed_energy_basis); //U_bar(A^-1*u)

            m_neg_energy_sample_jac = SparseMatrix<double>(n_dof, m_energy_sample_tets.size());
            m_neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, T.cols() * 3)); // Reserve enough room for 4 verts (tet corners) per column
            std::cout << "Done. Will sample from " << m_energy_sample_tets.size() << " tets." << std::endl;
        }
        else if(m_energy_method == AN08) {
            fs::path energy_model_dir = model_root / "energy_model/an08/";
            fs::path indices_path = energy_model_dir / "indices.dmat";
            fs::path weights_path = energy_model_dir / "weights.dmat";

            Eigen::VectorXi Is;
            Eigen::VectorXd Ws;
            igl::readDMAT(indices_path.string(), Is); //TODO do I need to sort this?
            igl::readDMAT(weights_path.string(), Ws); 

            m_cubature_weights = Ws;
            m_cubature_indices = Is; // TODO remove duplicate structures!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            m_energy_sample_tets.resize(Is.size());
            VectorXi::Map(&m_energy_sample_tets[0], Is.size()) = Is;

            m_neg_energy_sample_jac = SparseMatrix<double>(n_dof, m_energy_sample_tets.size());
            m_neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, T.cols() * 3)); // Reserve enough room for 4 verts (tet corners) per column
            std::cout << "Done. Will sample from " << m_energy_sample_tets.size() << " tets." << std::endl;
        }

    }

    void update_zs(const VectorXd &cur_z) {
        m_prev_z = cur_z;
        m_prev_sub_q = m_cur_sub_q;
        m_cur_z = cur_z;
        m_cur_sub_q = sub_dec(m_cur_z);
    }

    void set_interaction_force(const SparseVector<double> &interaction_force) {
        m_interaction_force = interaction_force;
    }

    void set_cubature_indices_and_weights(const VectorXi &indices, const VectorXd &weights) {
        m_cubature_weights = weights;
        m_cubature_indices = indices;
    }

    // Just short helpers
    VectorXd dec(const VectorXd &z) { return m_reduced_space->decode(z); }
    VectorXd sub_dec(const VectorXd &z) { return m_reduced_space->sub_decode(z); }
    VectorXd enc(const VectorXd &q) { return m_reduced_space->encode(q); }
    VectorXd jtvp(const VectorXd &z, const VectorXd &q) { return m_reduced_space->jacobian_transpose_vector_product(z, q); }

    double current_reduced_energy_and_forces(double &energy, VectorXd &UT_forces) {
        int n_sample_tets = m_energy_sample_tets.size();
        
        VectorXd sampled_energy(n_sample_tets);

        int n_force_per_element = T.cols() * 3;
        MatrixXd element_forces(n_sample_tets, n_force_per_element);
        

        #pragma omp parallel for num_threads(4) //schedule(static,64)// TODO how to optimize this
        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            int tet_index = m_energy_sample_tets[i];

            // Energy
            sampled_energy[i] = m_tets->getImpl().getElement(tet_index)->getStrainEnergy(m_world->getState());

            //Forces
            VectorXd sampled_force(n_force_per_element);
            m_tets->getImpl().getElement(tet_index)->getInternalForce(sampled_force, m_world->getState());
            element_forces.row(i) = sampled_force;
        }

        // Constructing sparse matrix is not thread safe
        m_neg_energy_sample_jac.setZero(); // Doesn't clear reserved memory
        for(int i = 0; i < n_sample_tets; i++) {
            int tet_index = m_energy_sample_tets[i];
            for(int j = 0; j < 4; j++) {
                int vert_index = m_tets->getImpl().getElement(tet_index)->getQDOFList()[j]->getGlobalId();
                for(int k = 0; k < 3; k++) {
                    m_neg_energy_sample_jac.insert(vert_index + k, i) = element_forces(i, j*3 + k);
                }
            }
        }
        m_neg_energy_sample_jac.makeCompressed();

        // VectorXd alpha = m_energy_sampled_basis.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(sampled_energy);
        // VectorXd alpha = (m_energy_sampled_basis.transpose() * m_energy_sampled_basis).ldlt().solve(m_energy_sampled_basis.transpose() * sampled_energy);
        // VectorXd alpha = m_energy_sampled_basis_qr.solve(sampled_energy);
        // energy = m_summed_energy_basis.dot(alpha);
        energy = m_cubature_weights.dot(sampled_energy);

        UT_forces = m_U.transpose() * (m_neg_energy_sample_jac * m_cubature_weights);
    }

    double current_reduced_energy_and_forces_using_pred_weights_l1(const VectorXd &z, double &energy, VectorXd &UT_forces, double &prediction_time_s) {
        // double get_weights_start = igl::get_seconds();
        // VectorXd weights;
        // VectorXi indices;
        m_reduced_space->get_cubature_indices_and_weights(z, m_cubature_indices, m_cubature_weights);
        // prediction_time_s = igl::get_seconds() - get_weights_start;
        // TODO do I need to do this every step of optimization?


        int n_sample_tets = m_cubature_indices.size();
        
        int n_force_per_element = T.cols() * 3;
        MatrixXd element_forces(n_sample_tets, n_force_per_element);
        
        VectorXd energy_samp(n_sample_tets);
        // VectorXd full_energies = VectorXd::Zero(T.rows());

        #pragma omp parallel for num_threads(4) //schedule(static,64)// TODO how to optimize this
        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            int tet_index = m_cubature_indices[i];
            energy_samp[i] = m_tets->getImpl().getElement(tet_index)->getStrainEnergy(m_world->getState());

            //Forces
            VectorXd sampled_force(n_force_per_element);
            m_tets->getImpl().getElement(tet_index)->getInternalForce(sampled_force, m_world->getState());
            element_forces.row(i) = sampled_force;
        }

        // #pragma omp parallel for //schedule(static,64)// TODO how to optimize this
        // for(int i = 0; i < T.rows(); i++) { // TODO parallel
        //     full_energies[i] = m_tets->getImpl().getElement(i)->getStrainEnergy(m_world->getState());
        // }
        // m_weight_vjp = m_reduced_space->cubature_vjp(z, full_energies);

        // Constructing sparse matrix is not thread safe
        m_neg_energy_sample_jac = SparseMatrix<double>(n_dof, m_cubature_indices.size());
        m_neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, T.cols() * 3)); // Reserve enough room for 4 verts (tet corners) per column       

        for(int i = 0; i < n_sample_tets; i++) {
            int tet_index = m_cubature_indices[i];
            for(int j = 0; j < 4; j++) {
                int vert_index = m_tets->getImpl().getElement(tet_index)->getQDOFList()[j]->getGlobalId();
                for(int k = 0; k < 3; k++) {
                    m_neg_energy_sample_jac.insert(vert_index + k, i) = element_forces(i, j*3 + k);
                }
            }
        }
        m_neg_energy_sample_jac.makeCompressed();

        energy = energy_samp.transpose() * m_cubature_weights;
        UT_forces = m_U.transpose() * (m_neg_energy_sample_jac * m_cubature_weights);
    }


    double operator()(const VectorXd& new_z, VectorXd& grad)
    {
        // Update the tets with candidate configuration
        double obj_start_time = igl::get_seconds();
        VectorXd new_sub_q = sub_dec(new_z);
        double tf_time = igl::get_seconds() - obj_start_time;

        double decode_start_time = igl::get_seconds();
        Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(m_tets->getQ(), *m_world);
        
        if(m_energy_method == PCR || m_energy_method == AN08) {
            // We only need to update verts for the tets we actually sample
            VectorXd sampled_q = m_U_sampled * new_sub_q;
            for(int i = 0; i < sampled_q.size(); i++) {
                q[m_cubature_vert_indices[i]] = sampled_q[i]; // TODO this could probably be more efficient with a sparse operator?
            } 
        } else {
            q = m_U * new_sub_q;
        } 

        // TWISTING
        // auto max_verts = get_min_verts(1, true);
        // Eigen::AngleAxis<double> rot(0.05 * current_frame, Eigen::Vector3d(0.0,1.0,0.0));
        // for(int i = 0; i < max_verts.size(); i++) {
        //     int vi = max_verts[i];
        //     Eigen::Vector3d v = V.row(vi);
        //     Eigen::Vector3d new_q = rot * v - v + Eigen::Vector3d(0.0,1.0,0.0) * current_frame / 300.0;
        //     for(int j = 0; j < 3; j++) {
        //         q[vi * 3 + j] = new_q[j];
        //     }
        // }

        double decode_time = igl::get_seconds() - decode_start_time;
        
        // Compute objective
        double energy;
        VectorXd UT_internal_forces;
        double energy_forces_start_time = igl::get_seconds();
        double predict_weight_time = 0.0;
        if(m_energy_method == PCR || m_energy_method == AN08) {
            current_reduced_energy_and_forces(energy, UT_internal_forces);
        }
        else if(m_energy_method == PRED_WEIGHTS_L1) {
            current_reduced_energy_and_forces_using_pred_weights_l1(m_prev_z, energy, UT_internal_forces, predict_weight_time);  // Use prev z so weights are constant through search
        }
        else if(m_energy_method == PRED_DIRECT) {
            energy = m_reduced_space->get_energy(new_z);
        }
        else {
            energy = m_tets->getStrainEnergy(m_world->getState());
            getInternalForceVector(m_internal_force_asm, *m_tets, *m_world);
            UT_internal_forces = m_U.transpose() * *m_internal_force_asm; // TODO can we avoid copy here?
        }
        double energy_forces_time = igl::get_seconds() - energy_forces_start_time;

        double obj_and_grad_time_start = igl::get_seconds();
        VectorXd h_2_UT_external_forces =  m_h * m_h * (m_U.transpose() * m_interaction_force + m_UT_F_ext);

        VectorXd sub_q_tilde = new_sub_q - 2.0 * m_cur_sub_q + m_prev_sub_q;
        VectorXd UTMU_sub_q_tilde = m_UTMU * sub_q_tilde;
        // Old
        double obj_val = 0.5 * sub_q_tilde.transpose() * UTMU_sub_q_tilde
                            + m_h * m_h * energy
                            - sub_q_tilde.transpose() * h_2_UT_external_forces;

        // Compute gradient
        // **** TODO
        // Can I further reduce the force calculations by carrying through U?

        if(m_energy_method == PRED_DIRECT) {
            grad = jtvp(new_z,UTMU_sub_q_tilde - h_2_UT_external_forces) - m_h * m_h * m_reduced_space->get_energy_gradient(new_z);
        } else {
            grad = jtvp(new_z,UTMU_sub_q_tilde - m_h * m_h * UT_internal_forces - h_2_UT_external_forces);
        }

        double obj_and_grad_time = igl::get_seconds() - obj_and_grad_time_start;
        double obj_time = igl::get_seconds() - obj_start_time;

        if(LOGGING_ENABLED) {
            timestep_info["iteration_info"]["lbfgs_obj_vals"].push_back(obj_val);
            timestep_info["iteration_info"]["timing"]["tot_obj_time_s"].push_back(obj_time);
            timestep_info["iteration_info"]["timing"]["tf_time_s"].push_back(tf_time);
            timestep_info["iteration_info"]["timing"]["linear_decode_time_s"].push_back(decode_time);
            timestep_info["iteration_info"]["timing"]["energy_forces_time_s"].push_back(energy_forces_time);
            timestep_info["iteration_info"]["timing"]["predict_weight_time"].push_back(predict_weight_time);
            timestep_info["iteration_info"]["timing"]["obj_and_grad_eval_time_s"].push_back(obj_and_grad_time);
        }

        return obj_val;
    }

    VectorXi get_current_tets() {
        return m_cubature_indices;
    }

private:
    VectorXd m_cur_z;
    VectorXd m_prev_z;
    VectorXd m_cur_sub_q;
    VectorXd m_prev_sub_q;
    VectorXd m_F_ext;
    VectorXd m_UT_F_ext;

    SparseVector<double> m_interaction_force;

    SparseMatrix<double> m_M; // mass matrix
    // Below is a kind of hack way to get the matrix type used in the reduced space class
    decltype(std::declval<ReducedSpaceType>().outer_jacobian())  m_UTMU; // reduced mass matrix
    decltype(std::declval<ReducedSpaceType>().outer_jacobian())  m_U;
    MatrixXd m_U_sampled; // Always used in dense mode

    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > m_M_asm;
    AssemblerParallel<double, AssemblerEigenVector<double> > m_internal_force_asm;

    double m_h;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    EnergyMethod m_energy_method;
    ReducedSpaceType *m_reduced_space;

    std::vector<int> m_energy_sample_tets;

    MatrixXd m_energy_basis;
    VectorXd m_summed_energy_basis;
    MatrixXd m_energy_sampled_basis;
    Eigen::FullPivHouseholderQR<MatrixXd> m_energy_sampled_basis_qr;
    SparseMatrix<double> m_neg_energy_sample_jac;

    VectorXd m_cubature_weights;
    VectorXi m_cubature_indices;
    VectorXi m_cubature_vert_indices;
};

template <typename ReducedSpaceType>
class GPLCTimeStepper {
public:
    GPLCTimeStepper(fs::path model_root, json integrator_config, MyWorld *world, NeohookeanTets *tets, ReducedSpaceType *reduced_space) :
        m_world(world), m_tets(tets), m_reduced_space(reduced_space) {

        m_use_preconditioner = integrator_config["use_preconditioner"];
        m_is_full_space = integrator_config["reduced_space_type"] == "full";
       // LBFGSpp::LBFGSParam<DataType> param;
       // param.epsilon = 1e-4;
       // param.max_iterations = 1000;
       // param.past = 1;
       // param.m = 5;
       // param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
        m_lbfgs_param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;

        // Set up lbfgs params
        json lbfgs_config = integrator_config["lbfgs_config"];
        m_lbfgs_param.m = lbfgs_config["lbfgs_m"];
        m_lbfgs_param.epsilon = lbfgs_config["lbfgs_epsilon"]; //1e-8// TODO replace convergence test with abs difference
        m_lbfgs_param.max_iterations = lbfgs_config["lbfgs_max_iterations"];//500;//300;
        m_solver = new LBFGSSolver<double>(m_lbfgs_param);

        reset_zs_to_current_world();
        m_gplc_objective = new GPLCObjective<ReducedSpaceType>(model_root, integrator_config, m_cur_z, m_prev_z, world, tets, reduced_space);

        m_h = integrator_config["timestep"];
        m_U = reduced_space->outer_jacobian();
        decltype(std::declval<ReducedSpaceType>().outer_jacobian()) J = reduced_space->inner_jacobian(m_cur_z);

        
        m_energy_method = energy_method_from_integrator_config(integrator_config);

        if(m_use_preconditioner) {
            std::cout << "Constructing reduced mass and stiffness matrix..." << std::endl;
            double start_time = igl::get_seconds();
            getMassMatrix(m_M_asm, *m_world);
            getStiffnessMatrix(m_K_asm, *m_world);
            m_UTMU = m_U.transpose() * (*m_M_asm) * m_U;
            m_UTKU = m_U.transpose() * (*m_K_asm) * m_U;
            m_H = J.transpose() * m_UTMU * J - m_h * m_h * J.transpose() * m_UTKU * J;
            m_H_llt.compute(m_H);
            std::cout << "Took " << (igl::get_seconds() - start_time) << "s" << std::endl;
        }
        // m_H_llt = MatrixXd::Identity(m_H.rows(), m_H.cols()).llt();
    }

    ~GPLCTimeStepper() {
        delete m_solver;
        delete m_gplc_objective;
    }

    void step(const SparseVector<double> &interaction_force) {
        if(LOGGING_ENABLED) {
            timestep_info = json(); // Clear timestep info for this frame.
        }
        double start_time = igl::get_seconds();

        VectorXd z_param = m_cur_z; // Stores both the first guess and the final result
        double min_val_res;

        m_gplc_objective->set_interaction_force(interaction_force);
        m_gplc_objective->update_zs(m_cur_z);
        
        int activated_tets = T.rows();
        // if(m_energy_method == PRED_WEIGHTS_L1) {
        //     VectorXd weights;
        //     VectorXi indices;
        //     m_reduced_space->get_cubature_indices_and_weights(m_cur_z, indices, weights);
        //     m_gplc_objective->set_cubature_indices_and_weights(indices, weights);
        //     activated_tets = indices.size();
        // }

        // Conclusion
        // For the AE model at least, preconditioning with rest hessian is slower
        // but preconditioning with rest hessian with current J gives fairly small speed increas
        // TODO: determine if llt or ldlt is faster
        int niter;
        double precondition_compute_time = 0.0;
        if(m_use_preconditioner) {
            double precondition_start_time = igl::get_seconds();
            // TODO shouldn't do this for any linear space
            if(!m_is_full_space) { // Only update the hessian each time for nonlinear space. 
                decltype(std::declval<ReducedSpaceType>().outer_jacobian()) J = m_reduced_space->inner_jacobian(m_cur_z);
                m_H = J.transpose() * m_UTMU * J - m_h * m_h * J.transpose() * m_UTKU * J;
                m_H_llt.compute(m_H);//m_H.ldlt();
            }
            if(m_is_full_space){ // Currently doing a FULL hessian update for the full space..
                getMassMatrix(m_M_asm, *m_world);
                getStiffnessMatrix(m_K_asm, *m_world);
                m_UTMU = m_U.transpose() * (*m_M_asm) * m_U;
                m_UTKU = m_U.transpose() * (*m_K_asm) * m_U;
                m_H = m_UTMU - m_h * m_h * m_UTKU;
                m_H_llt.compute(m_H);
            }
            precondition_compute_time = igl::get_seconds() - precondition_start_time;

            niter = m_solver->minimizeWithPreconditioner(*m_gplc_objective, z_param, min_val_res, m_H_llt);   
        } else {
            niter = m_solver->minimize(*m_gplc_objective, z_param, min_val_res);   
        }
        double update_time = igl::get_seconds() - start_time;

        std::cout << niter << " iterations" << std::endl;
        std::cout << "objective val = " << min_val_res << std::endl;

        m_prev_z = m_cur_z;
        m_cur_z = z_param; // TODO: Use pointers to avoid copies

        // update_world_with_current_configuration(); This happens in each opt loop

        m_total_time += update_time;
        std::cout << "Timestep took: " << update_time << "s" << std::endl;
        std::cout << "Avg timestep: " << m_total_time / (double)current_frame << "s" << std::endl;
        // std::cout << "Current z: " << m_cur_z.transpose() << std::endl;
        std::cout << "Activated tets: " << activated_tets << std::endl;

        if(LOGGING_ENABLED) {
            m_total_tets += activated_tets;

            timestep_info["current_frame"] = current_frame; // since at end of timestep
            timestep_info["tot_step_time_s"] = update_time;
            timestep_info["precondition_time"] = precondition_compute_time;
            timestep_info["lbfgs_iterations"] = niter;
            timestep_info["avg_activated_tets"] = m_total_tets / (double) current_frame;

            timestep_info["mouse_info"]["dragged_pos"] = {dragged_pos[0], dragged_pos[1], dragged_pos[2]};
            timestep_info["mouse_info"]["is_dragging"] = is_dragging;

            sim_log["timesteps"].push_back(timestep_info);
            if(current_frame % log_save_freq == 0) {
                log_ofstream.seekp(0);
                log_ofstream << std::setw(2) << sim_log;
            }
        }


        return;
    }

    void update_world_with_current_configuration() {
        // TODO: is this the fastest way to do this?
        Eigen::Map<Eigen::VectorXd> gauss_map_q = mapDOFEigen(m_tets->getQ(), *m_world);
        VectorXd cur_q = m_reduced_space->decode(m_cur_z);
        for(int i=0; i < cur_q.size(); i++) {
            gauss_map_q[i] = cur_q[i];
        }
    }

    void reset_zs_to_current_world() {
        Eigen::Map<Eigen::VectorXd> gauss_map_q = mapDOFEigen(m_tets->getQ(), *m_world);
        m_prev_z = m_reduced_space->encode(gauss_map_q);
        m_cur_z = m_reduced_space->encode(gauss_map_q);
        update_world_with_current_configuration();
    }

    ReducedSpaceType* get_reduced_space() {
        return m_reduced_space;
    }

    VectorXd get_current_z() {
        return m_cur_z;
    }

    VectorXd get_current_q() {
        return m_reduced_space->decode(m_cur_z);
    }

    VectorXd get_current_V() {
        auto q = get_current_q();
        Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only

        return V + dV.transpose(); 
    }

    VectorXi get_current_tets() {
        return m_gplc_objective->get_current_tets();
    }

private:
    bool m_use_preconditioner;
    bool m_is_full_space;

    LBFGSParam<double> m_lbfgs_param;
    LBFGSSolver<double> *m_solver;
    GPLCObjective<ReducedSpaceType> *m_gplc_objective;

    VectorXd m_prev_z;
    VectorXd m_cur_z;

    double m_h;

    decltype(std::declval<ReducedSpaceType>().outer_jacobian()) m_H;
    typename std::conditional<
        std::is_same<decltype(std::declval<ReducedSpaceType>().outer_jacobian()), MatrixXd>::value,
        Eigen::LDLT<MatrixXd>,
        Eigen::SimplicialLDLT<SparseMatrix<double>>>::type m_H_llt;
    decltype(std::declval<ReducedSpaceType>().outer_jacobian()) m_H_inv;
    decltype(std::declval<ReducedSpaceType>().outer_jacobian()) m_UTKU;
    decltype(std::declval<ReducedSpaceType>().outer_jacobian()) m_UTMU;
    decltype(std::declval<ReducedSpaceType>().outer_jacobian()) m_U;

    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > m_M_asm;
    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > m_K_asm;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    ReducedSpaceType *m_reduced_space;
    EnergyMethod m_energy_method;

    double m_total_time = 0.0;
    int m_total_tets = 0; // Sum the number of tets activated each frame
};



SparseVector<double> compute_interaction_force(const Vector3d &dragged_pos, int dragged_vert, bool is_dragging, double spring_stiffness, NeohookeanTets *tets, const MyWorld &world) {
    SparseVector<double> force(n_dof);// = VectorXd::Zero(n_dof); // TODO make this a sparse vector?
    force.reserve(3);
    if(is_dragging) {
        Vector3d fem_attached_pos = PosFEM<double>(&tets->getQ()[dragged_vert], dragged_vert, &tets->getImpl().getV())(world.getState());
        Vector3d local_force = spring_stiffness * (dragged_pos - fem_attached_pos);

        for(int i=0; i < 3; i++) {
            force.insert(dragged_vert * 3 + i) = local_force[i];    
        }
    } 

    return force;
}

VectorXd get_von_mises_stresses(NeohookeanTets *tets, MyWorld &world) {
    // Compute cauchy stress
    int n_ele = tets->getImpl().getF().rows();

    Eigen::Matrix<double, 3,3> stress = MatrixXd::Zero(3,3);
    VectorXd vm_stresses(n_ele);

    std::cout << "Disabled cauchy stress for linear tets.." << std::endl;
    // for(unsigned int i=0; i < n_ele; ++i) {
    //     tets->getImpl().getElement(i)->getCauchyStress(stress, Vec3d(0,0,0), world.getState());

    //     double s11 = stress(0, 0);
    //     double s22 = stress(1, 1);
    //     double s33 = stress(2, 2);
    //     double s23 = stress(1, 2);
    //     double s31 = stress(2, 0);
    //     double s12 = stress(0, 1);
    //     vm_stresses[i] = sqrt(0.5 * (s11-s22)*(s11-s22) + (s33-s11)*(s33-s11) + 6.0 * (s23*s23 + s31*s31 + s12*s12));
    // }

    return vm_stresses;
}

std::string int_to_padded_str(int i) {
    std::stringstream frame_str;
    frame_str << std::setfill('0') << std::setw(5) << i;
    return frame_str.str();
}


// typedef ReducedSpace<IdentitySpaceImpl> IdentitySpace;
// typedef ReducedSpace<ConstraintSpaceImpl> ConstraintSpace;
typedef ReducedSpace<LinearSpaceImpl<MatrixXd>> LinearSpace;
typedef ReducedSpace<LinearSpaceImpl<SparseMatrix<double>>> SparseConstraintSpace;
typedef ReducedSpace<AutoEncoderSpaceImpl> AutoencoderSpace;

template <typename ReducedSpaceType>
void run_sim(ReducedSpaceType *reduced_space, const json &config, const fs::path &model_root) {
    // -- Setting up GAUSS
    MyWorld world;

    MatrixXi T_sampled;
    MatrixXi F_sampled;
    if(LOGGING_ENABLED) {
        if(config["integrator_config"]["use_reduced_energy"]) {
            fs::path sample_tets_path("pca_results/energy_indices.dmat");
            MatrixXi tet_indices_mat;
            igl::readDMAT((model_root / sample_tets_path).string(), tet_indices_mat);

            T_sampled = igl::slice(T, tet_indices_mat, 1);
            igl::boundary_facets(T_sampled, F_sampled);
        }
    }

    auto material_config = config["material_config"];
    auto integrator_config = config["integrator_config"];
    auto visualization_config = config["visualization_config"];

    NeohookeanTets *tets = new NeohookeanTets(V,T);
    n_dof = tets->getImpl().getV().rows() * 3;
    for(auto element: tets->getImpl().getElements()) {
        element->setDensity(material_config["density"]);
        element->setParameters(material_config["youngs_modulus"], material_config["poissons_ratio"]);   
    }

    world.addSystem(tets);
    fixDisplacementMin(world, tets);
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    reset_world(world);
    // --- Finished GAUSS set up
    

    // --- My integrator set up
    double spring_stiffness = visualization_config["interaction_spring_stiffness"];
    SparseVector<double> interaction_force(n_dof);// = VectorXd::Zero(n_dof);


    GPLCTimeStepper<ReducedSpaceType> gplc_stepper(model_root, integrator_config, &world, tets, reduced_space);

    bool show_stress = visualization_config["show_stress"];
    bool show_energy = false;
    bool show_tets = false;
    try {
        show_energy = visualization_config.at("show_energy");
    }
    catch (nlohmann::detail::out_of_range& e){} // Didn't exist

    if(show_energy && show_stress) {
        std::cout << "Can't show both energy and stress. Exiting." << std::endl;
        exit(1);
    }

    /** libigl display stuff **/
    igl::viewer::Viewer viewer;

    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
        // predefined colors
        const Eigen::RowVector3d orange(1.0,0.7,0.2);
        const Eigen::RowVector3d yellow(1.0,0.9,0.2);
        const Eigen::RowVector3d blue(0.2,0.3,0.8);
        const Eigen::RowVector3d green(0.2,0.6,0.3);
        // const Eigen::RowVector3d sea_green(70./255.,252./255.,167./255.);
        const Eigen::RowVector3d sea_green(229./255.,211./255.,91./255.);

        if(is_dragging) {
            // Eigen::MatrixXd part_pos(1, 3);
            // part_pos(0,0) = dragged_pos[0]; // TODO why is eigen so confusing.. I just want to make a matrix from vec
            // part_pos(0,1) = dragged_pos[1];
            // part_pos(0,2) = dragged_pos[2];

            // viewer.data.set_points(part_pos, orange);

            MatrixXi E(1,2);
            E(0,0) = 0;
            E(0,1) = 1;
            MatrixXd P(2,3);
            P.row(0) = dragged_pos;
            P.row(1) = mesh_pos;
            
            viewer.data.set_edges(P, E, sea_green);
        } else {
            Eigen::MatrixXd part_pos = MatrixXd::Zero(1,3);
            part_pos(0,0)=100000.0;
            // viewer.data.set_points(part_pos, sea_green);

            MatrixXi E(1,2);
            E(0,0) = 0;
            E(0,1) = 1;
            MatrixXd P(2,3);
            P.row(0) = Eigen::RowVector3d(1000.0,1000.0, 1000.0);
            P.row(1) = Eigen::RowVector3d(1000.0,1000.0, 1000.0);;
            viewer.data.set_edges(P, E, sea_green);
        }

        if(viewer.core.is_animating)
        {   
            auto q = mapDOFEigen(tets->getQ(), world);
            Eigen::MatrixXd newV = gplc_stepper.get_current_V();

            if(show_tets) {
                VectorXi tet_indices = gplc_stepper.get_current_tets();
                T_sampled = igl::slice(T, tet_indices, 1);
                igl::boundary_facets(T_sampled, F_sampled);

                viewer.data.clear();
                viewer.data.set_mesh(newV, F_sampled);
            } else {
                viewer.data.set_vertices(newV);
            }

            mesh_pos = newV.row(dragged_vert);

            if(per_vertex_normals) {
                igl::per_vertex_normals(V,F,N);
            } else {
                igl::per_corner_normals(V,F,40,N);
            }
            viewer.data.set_normals(N);

            // Play back mouse interaction from previous sim
            if(PLAYBACK_SIM) {
                json current_mouse_info = sim_playback_json["timesteps"][current_frame]["mouse_info"];
                dragged_pos = Eigen::RowVector3d(current_mouse_info["dragged_pos"][0], current_mouse_info["dragged_pos"][1], current_mouse_info["dragged_pos"][2]);

                bool was_dragging = is_dragging;
                is_dragging = current_mouse_info["is_dragging"];

                if(is_dragging && !was_dragging) { // Got a click
                    // Get closest point on mesh
                    MatrixXd C(1,3);
                    C.row(0) = dragged_pos;
                    MatrixXi I;
                    igl::snap_points(C, newV, I);

                    dragged_vert = I(0,0);
                }

                mesh_pos = newV.row(dragged_vert);
            }
            
            // Do the physics update
            interaction_force = compute_interaction_force(dragged_pos, dragged_vert, is_dragging, spring_stiffness, tets, world);
            gplc_stepper.step(interaction_force);

            if(LOGGING_ENABLED) {
                fs::path obj_filename(int_to_padded_str(current_frame) + "_surface_faces.obj");
                igl::writeOBJ((surface_obj_dir /obj_filename).string(), newV, F);

                if(config["integrator_config"]["use_reduced_energy"]) {
                    fs::path tet_obj_filename(int_to_padded_str(current_frame) + "_sample_tets.obj");

                    // Need to recompute sample tets each time if they are changing.
                    if(PRED_WEIGHTS_L1 == energy_method_from_integrator_config(config["integrator_config"])) {
                        VectorXd weights;
                        VectorXi indices;
                        gplc_stepper.get_reduced_space()->get_cubature_indices_and_weights(gplc_stepper.get_current_z(), indices, weights);

                        MatrixXi sample_tets = igl::slice(T, indices, 1);
                        igl::boundary_facets(sample_tets, F_sampled);

                        igl::writeOBJ((tet_obj_dir /tet_obj_filename).string(), newV, F_sampled);                    
                    } else {
                        igl::writeOBJ((tet_obj_dir /tet_obj_filename).string(), newV, F_sampled);                    
                    }
                }

                // Export the pointer line
                MatrixXi pointerF(1, 2);
                pointerF << 0, 1;
                MatrixXd pointerV(2,3);
                if(is_dragging) {
                    pointerV.row(0) = mesh_pos;
                    pointerV.row(1) = dragged_pos;
                } else {
                    pointerV << 10000, 10000, 10000, 10001, 10001, 10001; // Fix this and make it work
                }

                fs::path pointer_filename(int_to_padded_str(current_frame) + "_mouse_pointer.obj");
                igl::writeOBJ((pointer_obj_dir / pointer_filename).string(), pointerV, pointerF);
            }
            current_frame++;
        }

        if(show_stress) {
            // Do stress field viz
            VectorXd vm_stresses = get_von_mises_stresses(tets, world);
            // vm_stresses is per element... Need to compute avg value per vertex
            VectorXd vm_per_vert = VectorXd::Zero(V.rows());
            VectorXi neighbors_per_vert = VectorXi::Zero(V.rows());
            
            int t = 0;
            for(int i=0; i < T.rows(); i++) {
                for(int j=0; j < 4; j++) {
                    t++;
                    int vert_index = T(i,j);
                    vm_per_vert[vert_index] += vm_stresses[i];
                    neighbors_per_vert[vert_index]++;
                }
            }
            for(int i=0; i < vm_per_vert.size(); i++) {
                vm_per_vert[i] /= neighbors_per_vert[i];
            }
            VectorXd vm_per_face = VectorXd::Zero(F.rows());
            for(int i=0; i < vm_per_face.size(); i++) {
                vm_per_face[i] = (vm_per_vert[F(i,0)] + vm_per_vert[F(i,1)] + vm_per_vert[F(i,2)])/3.0;
            }
            // std::cout << vm_per_face.maxCoeff() << " " <<  vm_per_face.minCoeff() << std::endl;
            MatrixXd C;
            //VectorXd((vm_per_face.array() -  vm_per_face.minCoeff()) / vm_per_face.maxCoeff())
            igl::jet(vm_per_vert / 60.0, false, C);
            viewer.data.set_colors(C);
        }

        if(show_energy) {
            // Do stress field viz
            VectorXd energy_per_element = tets->getImpl().getStrainEnergyPerElement(world.getState());
            // energy_per_element is per element... Need to compute avg value per vertex
            VectorXd energy_per_vert = VectorXd::Zero(V.rows());
            VectorXi neighbors_per_vert = VectorXi::Zero(V.rows());
            
            int t = 0;
            for(int i=0; i < T.rows(); i++) {
                for(int j=0; j < 4; j++) {
                    t++;
                    int vert_index = T(i,j);
                    energy_per_vert[vert_index] += energy_per_element[i];
                    neighbors_per_vert[vert_index]++;
                }
            }
            for(int i=0; i < energy_per_vert.size(); i++) {
                energy_per_vert[i] /= neighbors_per_vert[i];
            }
            VectorXd energy_per_face = VectorXd::Zero(F.rows());
            for(int i=0; i < energy_per_face.size(); i++) {
                energy_per_face[i] = (energy_per_vert[F(i,0)] + energy_per_vert[F(i,1)] + energy_per_vert[F(i,2)])/3.0;
            }

            MatrixXd C;
            
            igl::jet(energy_per_face / 100.0, false, C);
            viewer.data.set_colors(C);
        }

        return false;
    };

    viewer.callback_key_pressed = [&](igl::viewer::Viewer &, unsigned int key, int mod)
    {
        switch(key)
        {
            case 'q':
            {
                log_ofstream.seekp(0);
                log_ofstream << std::setw(2) << sim_log;
                log_ofstream.close();
                exit(0);
                break;
            }
            case 'P':
            case 'p':
            {
                viewer.core.is_animating = !viewer.core.is_animating;
                break;
            }
            case 'r':
            case 'R':
            {
                reset_world(world);
                gplc_stepper.reset_zs_to_current_world();
                break;
            }
            case 's':
            case 'S':
            {
                show_stress = !show_stress;
                show_energy = false;
                break;
            }
            case 't':
            case 'T':
            {
                viewer.data.clear();
                viewer.data.set_mesh(V, F);
                show_tets = !show_tets;
                break;
            }
            case 'n':
            case 'N':
            {
                per_vertex_normals = !per_vertex_normals;
                break;
            }
            default:
            return false;
        }
        return true;
    };

    viewer.callback_mouse_down = [&](igl::viewer::Viewer&, int, int)->bool
    {   
        if(!PLAYBACK_SIM) {
            Eigen::MatrixXd curV = gplc_stepper.get_current_V(); 
            last_mouse = Eigen::RowVector3f(viewer.current_mouse_x,viewer.core.viewport(3)-viewer.current_mouse_y,0);
            
            // Find closest point on mesh to mouse position
            int fid;
            Eigen::Vector3f bary;
            if(igl::unproject_onto_mesh(
                last_mouse.head(2),
                viewer.core.view * viewer.core.model,
                viewer.core.proj, 
                viewer.core.viewport, 
                curV, F, 
                fid, bary))
            {
                long c;
                bary.maxCoeff(&c);
                dragged_pos = curV.row(F(fid,c)) + Eigen::RowVector3d(0.001,0.0,0.0); //Epsilon offset so we don't div by 0
                mesh_pos = curV.row(F(fid,c));
                dragged_vert = F(fid,c);
                std::cout << "Grabbed vertex: " << dragged_vert << std::endl;
                is_dragging = true;


                // forceSpring->getImpl().setStiffness(spring_stiffness);
                // auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
                // pinned_q = dragged_pos;//(dragged_pos).cast<double>(); // necessary?

                // fem_attached_pos = PosFEM<double>(&tets->getQ()[dragged_vert],dragged_vert, &tets->getImpl().getV());
                // forceSpring->getImpl().setPosition0(fem_attached_pos);

                return true;
            }
        }
        
        return false; // TODO false vs true??
    };

    viewer.callback_mouse_up = [&](igl::viewer::Viewer&, int, int)->bool
    {
        if(!PLAYBACK_SIM) {
            is_dragging = false;
        }
        return false;
    };

    viewer.callback_mouse_move = [&](igl::viewer::Viewer &, int,int)->bool
    {
        if(!PLAYBACK_SIM) {
            Eigen::MatrixXd curV = gplc_stepper.get_current_V();
            if(is_dragging) {
                Eigen::RowVector3f drag_mouse(
                    viewer.current_mouse_x,
                    viewer.core.viewport(3) - viewer.current_mouse_y,
                    last_mouse(2));

                Eigen::RowVector3f drag_scene,last_scene;

                igl::unproject(
                    drag_mouse,
                    viewer.core.view*viewer.core.model,
                    viewer.core.proj,
                    viewer.core.viewport,
                    drag_scene);
                igl::unproject(
                    last_mouse,
                    viewer.core.view*viewer.core.model,
                    viewer.core.proj,
                    viewer.core.viewport,
                    last_scene);

                dragged_pos += ((drag_scene-last_scene)*4.5).cast<double>(); //TODO why do I need to fine tune this
                mesh_pos = curV.row(dragged_vert);
                last_mouse = drag_mouse;
            }
        }
        return false;
    };

    viewer.data.set_mesh(V,F);
    igl::per_corner_normals(V,F,40,N);
    viewer.data.set_normals(N);

    viewer.core.show_lines = false;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = false;
    viewer.core.line_width = 2;

    viewer.core.background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
    viewer.core.shininess = 120.0;
    viewer.data.set_colors(Eigen::RowVector3d(igl::CYAN_DIFFUSE[0], igl::CYAN_DIFFUSE[1], igl::CYAN_DIFFUSE[2]));
    // viewer.data.uniform_colors(
    //   Eigen::Vector3d(igl::CYAN_AMBIENT[0], igl::CYAN_AMBIENT[1], igl::CYAN_AMBIENT[2]),
    //   Eigen::Vector3d(igl::CYAN_DIFFUSE[0], igl::CYAN_DIFFUSE[1], igl::CYAN_DIFFUSE[2]),
    //   Eigen::Vector3d(igl::CYAN_SPECULAR[0], igl::CYAN_SPECULAR[1], igl::CYAN_SPECULAR[2]));

    // viewer.launch();

    viewer.launch_init(true, false);    
      viewer.opengl.shader_mesh.free();

  {
    std::string mesh_vertex_shader_string =
R"(#version 150
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
in vec3 position;
in vec3 normal;
out vec3 position_eye;
out vec3 normal_eye;
in vec4 Ka;
in vec4 Kd;
in vec4 Ks;
in vec2 texcoord;
out vec2 texcoordi;
out vec4 Kai;
out vec4 Kdi;
out vec4 Ksi;

void main()
{
  position_eye = vec3 (view * model * vec4 (position, 1.0));
  normal_eye = vec3 (view * model * vec4 (normal, 0.0));
  normal_eye = normalize(normal_eye);
  gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * model * vec4(position, 1.0);
  Kai = Ka;
  Kdi = Kd;
  Ksi = Ks;
  texcoordi = texcoord;
})";

    std::string mesh_fragment_shader_string =
R"(#version 150
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 fixed_color;
in vec3 position_eye;
in vec3 normal_eye;
uniform vec3 light_position_world;
vec3 Ls = vec3 (1, 1, 1);
vec3 Ld = vec3 (1, 1, 1);
vec3 La = vec3 (1, 1, 1);
in vec4 Ksi;
in vec4 Kdi;
in vec4 Kai;
in vec2 texcoordi;
uniform sampler2D tex;
uniform float specular_exponent;
uniform float lighting_factor;
uniform float texture_factor;
out vec4 outColor;
void main()
{
vec3 Ia = La * vec3(Kai);    // ambient intensity

vec3 light_position_eye = vec3 (view * vec4 (light_position_world, 1.0));
vec3 vector_to_light_eye = light_position_eye - position_eye;
vec3 direction_to_light_eye = normalize (vector_to_light_eye);
float dot_prod = dot (direction_to_light_eye, normal_eye);
float clamped_dot_prod = max (dot_prod, 0.0);
vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity

vec3 reflection_eye = reflect (-direction_to_light_eye, normal_eye);
vec3 surface_to_viewer_eye = normalize (-position_eye);
float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
dot_prod_specular = float(abs(dot_prod)==dot_prod) * max (dot_prod_specular, 0.0);
float specular_factor = pow (dot_prod_specular, specular_exponent);
vec3 Kfi = 0.5*vec3(Ksi);
vec3 Lf = Ls;
float fresnel_exponent = 2*specular_exponent;
float fresnel_factor = 0;
{
  float NE = max( 0., dot( normal_eye, surface_to_viewer_eye));
  fresnel_factor = pow (max(sqrt(1. - NE*NE),0.0), fresnel_exponent);
}
vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
vec3 If = Lf * vec3(Kfi) * fresnel_factor;     // fresnel intensity
vec4 color = vec4(lighting_factor * (If + Is + Id) + Ia + 
  (1.0-lighting_factor) * vec3(Kdi),(Kai.a+Ksi.a+Kdi.a)/3);
outColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;
if (fixed_color != vec4(0.0)) outColor = fixed_color;
})";

  viewer.opengl.shader_mesh.init(
      mesh_vertex_shader_string,
      mesh_fragment_shader_string, 
      "outColor");
  }


  viewer.launch_rendering(true);
  viewer.launch_shut();

}

const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

VectorXd tf_JTJ(const VectorXd &z, const VectorXd &z_v, tf::Session* m_decoder_JTJ_session) {
    tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, z_v.size()})); 
    auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
    for(int i =0; i < z.size(); i++) {
        z_tensor_mapped(0, i) = z[i];
    } // TODO map with proper function

    tf::Tensor z_v_tensor(tf_dtype, tf::TensorShape({1, z_v.size()}));  
    auto z_v_tensor_mapped = z_v_tensor.tensor<tf_dtype_type, 2>();
    for(int i =0; i < z_v.size(); i++) {
        z_v_tensor_mapped(0, i) = z_v[i];
    } // TODO map with proper function

    std::vector<tf::Tensor> JTJ_outputs; 
    tf::Status status = m_decoder_JTJ_session->Run({{"decoder_input:0", z_tensor}, {"input_z_v:0", z_v_tensor}},
                               {"JTJ/dense_decode_layer_0/MatMul_grad/MatMul:0"}, {}, &JTJ_outputs); // TODO get better names
    
    auto JTJ_tensor_mapped = JTJ_outputs[0].tensor<tf_dtype_type, 2>();
    
    VectorXd res(z_v.size()); // TODO generalize
    for(int i = 0; i < res.rows(); i++) {
        res[i] = JTJ_tensor_mapped(0,i);    
    }
    return res;
}

void compare_jac_speeds(AutoencoderSpace &reduced_space) {
    fs::path tf_models_root("../../models/x-final/tf_models/");
    fs::path decoder_JTJ_path = tf_models_root / "decoder_JTJ.pb";
    tf::Session* m_decoder_JTJ_session;
    tf::NewSession(tf::SessionOptions(), &m_decoder_JTJ_session);
    
    tf::GraphDef decoder_JTJ_graph_def;
    ReadBinaryProto(tf::Env::Default(), decoder_JTJ_path.string(), &decoder_JTJ_graph_def);
    
    m_decoder_JTJ_session->Create(decoder_JTJ_graph_def);
    



    for(int j = 100; j < 10001; j *= 10) {
        int n_its = j;
        
        VectorXd q = VectorXd::Zero(reduced_space.outer_jacobian().rows());
        VectorXd z = reduced_space.encode(q);
        VectorXd vq = MatrixXd::Ones(reduced_space.outer_jacobian().cols(), 1);
        VectorXd vz = MatrixXd::Ones(z.size(), 1);
    
        
        double start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            MatrixXd full_jac = reduced_space.inner_jacobian(z);
            VectorXd Jv = full_jac * vz;    
        }
        std::cout << n_its << " its of fd jac product took " << igl::get_seconds() - start << std::endl;

        start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            VectorXd Jv = reduced_space.jacobian_vector_product(z, vz);    
        }
        std::cout << n_its << " its of jvp took " << igl::get_seconds() - start << std::endl;

        start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            MatrixXd full_jac = reduced_space.inner_jacobian(z);
            VectorXd Jv = full_jac.transpose() * vq;    
        }
        std::cout << n_its << " its of fd jacT product took " << igl::get_seconds() - start << std::endl;

        start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            VectorXd Jv = reduced_space.jacobian_transpose_vector_product(z, vq);    
        }
        std::cout << n_its << " its of vjp took " << igl::get_seconds() - start << std::endl;

        MatrixXd U = reduced_space.outer_jacobian();
        MatrixXd UTU = U.transpose() * U;
        start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            MatrixXd full_jac = reduced_space.inner_jacobian(z);
            VectorXd JUTUv = full_jac.transpose() * UTU * full_jac * vz;    
        }
        std::cout << n_its << " its of fd JUTUJv product took " << igl::get_seconds() - start << std::endl;

        start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            VectorXd JUTUv = reduced_space.jacobian_transpose_vector_product(z, UTU * reduced_space.jacobian_vector_product(z, vz));    
        }
        std::cout << n_its << " its of vjp JUTUJv took " << igl::get_seconds() - start << std::endl;

        start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            VectorXd JUTUv = reduced_space.jacobian_transpose_vector_product(z, reduced_space.jacobian_vector_product(z, vz));    
        }
        std::cout << n_its << " its of vjp JTJv took " << igl::get_seconds() - start << std::endl;

        start = igl::get_seconds();
        for(int i = 0; i < n_its; i++) {
            VectorXd JUTUv = tf_JTJ(z, vz, m_decoder_JTJ_session);    
        }
        std::cout << n_its << " its of Tensorflow JTJv took " << igl::get_seconds() - start << std::endl;

    }

}

int main(int argc, char **argv) {
    
    // Load the configuration file
    if(argc < 2) {
        std::cout << "Expected model root." << std::endl;
    }

    fs::path model_root(argv[1]);
    fs::path sim_config("sim_config.json");

    if(argc == 3) {
        fs::path sim_recording_path(argv[2]);
        std::ifstream recording_fin(sim_recording_path.string());

        recording_fin >> sim_playback_json;
        PLAYBACK_SIM = true;
    }


    std::ifstream fin((model_root / sim_config).string());
    json config;
    fin >> config;

    auto integrator_config = config["integrator_config"];
    std::string reduced_space_string = integrator_config["reduced_space_type"];

    LOGGING_ENABLED = config["logging_enabled"];
    if(LOGGING_ENABLED) {
        sim_log["sim_config"] = config;
        sim_log["timesteps"] = {};
        fs::path sim_log_path("simulation_logs/");
        log_dir = model_root / fs::path("./simulation_logs/" + currentDateTime() + "/");
        surface_obj_dir = log_dir / fs::path("objs/surface/");
        pointer_obj_dir = log_dir / fs::path("objs/pointer/");
        tet_obj_dir = log_dir / fs::path("objs/sampled_tets/");
        if(!boost::filesystem::exists(model_root / sim_log_path)){
            boost::filesystem::create_directory(model_root / sim_log_path);
        }

        if(!boost::filesystem::exists(log_dir)){
            boost::filesystem::create_directory(log_dir);
        }

        if(!boost::filesystem::exists(surface_obj_dir)){
            boost::filesystem::create_directories(surface_obj_dir);
        }

        if(!boost::filesystem::exists(pointer_obj_dir)){
            boost::filesystem::create_directories(pointer_obj_dir);
        }

        if(!boost::filesystem::exists(tet_obj_dir) && integrator_config["use_reduced_energy"]){
            boost::filesystem::create_directories(tet_obj_dir);
        }

        // boost::filesystem::create_directories(log_dir);
        fs::path log_file("sim_stats.json");

        log_ofstream = std::ofstream((log_dir / log_file).string());
    }

    // Load the mesh here
    igl::readMESH((model_root / "tets.mesh").string(), V, T, F);
    igl::boundary_facets(T,F);
    // Center mesh
    Eigen::RowVector3d centroid;
    igl::centroid(V, F, centroid);
    V = V.rowwise() - centroid;

    if(reduced_space_string == "linear") {
        std::string pca_dim(std::to_string((int)integrator_config["pca_dim"]));
        fs::path pca_components_path("pca_results/pca_components_" + pca_dim + ".dmat");

        MatrixXd U;
        igl::readDMAT((model_root / pca_components_path).string(), U);
        LinearSpace reduced_space(U);

        run_sim<LinearSpace>(&reduced_space, config, model_root);
    }
    else if(reduced_space_string == "autoencoder") {
        fs::path tf_models_root(model_root / "tf_models/");
        AutoencoderSpace reduced_space(tf_models_root, integrator_config);
        // compare_jac_speeds(reduced_space);
        run_sim<AutoencoderSpace>(&reduced_space, config, model_root);  
    }
    else if(reduced_space_string == "full") {
        int fixed_axis = 1;
        try {
            fixed_axis = config["visualization_config"].at("full_space_constrained_axis");
        } 
        catch (nlohmann::detail::out_of_range& e){
            std::cout << "full_space_constrained_axis field not found in visualization_config" << std::endl;
            exit(1);
        }

        auto min_verts = get_min_verts(fixed_axis);
        // For twisting
        // auto max_verts = get_min_verts(fixed_axis, true);
        // min_verts.insert(min_verts.end(), max_verts.begin(), max_verts.end());

        SparseMatrix<double> P = construct_constraints_P(V, min_verts); // Constrain on X axis
        SparseConstraintSpace reduced_space(P.transpose());

        run_sim<SparseConstraintSpace>(&reduced_space, config, model_root);
    }
    else {
        std::cout << "Not yet implemented." << std::endl;
        return 1;
    }
    
    return EXIT_SUCCESS;
}

