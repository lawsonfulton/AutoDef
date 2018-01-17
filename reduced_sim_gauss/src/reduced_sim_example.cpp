#include <vector>

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
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/get_seconds.h>
#include <igl/jet.h>
#include <igl/slice.h>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>

#include <boost/filesystem.hpp>

// Optimization
#include <LBFGS.h>

// JSON
#include <json.hpp>

// Tensorflow
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include <omp.h>

using json = nlohmann::json;
namespace fs = boost::filesystem;

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
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
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices
int n_dof;
double finite_diff_eps;

// Mouse/Viewer state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
bool is_dragging = false;
int dragged_vert = 0;
int current_frame = 0;

// Parameters
bool saving_training_data = false;
std::string output_dir = "output_data/";


void save_displacements_DMAT(const std::string path, MyWorld &world, NeohookeanTets *tets) { // TODO: Add mouse position data to ouput
    auto q = mapDOFEigen(tets->getQ(), world);
    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
    Eigen::MatrixXd displacements = dV.transpose();

    igl::writeDMAT(path, displacements, false); // Don't use ascii
}

void save_base_configurations_DMAT(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    std::stringstream verts_filename, faces_filename;
    verts_filename << output_dir << "base_verts.dmat";
    faces_filename << output_dir << "base_faces.dmat";
    
    igl::writeDMAT(verts_filename.str(), V, false); // Don't use ascii
    igl::writeDMAT(faces_filename.str(), F, false); // Don't use ascii
}

// Todo put this in utilities
Eigen::MatrixXd getCurrentVertPositions(MyWorld &world, NeohookeanTets *tets) {
    // Eigen::Map<Eigen::MatrixXd> q(mapStateEigen<0>(world).data(), V.cols(), V.rows()); // Get displacements only
    auto q = mapDOFEigen(tets->getQ(), world);
    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only

    return V + dV.transpose(); 
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

    inline VectorXd encode(const VectorXd &q) {
        return m_impl.encode(q);
    }

    inline VectorXd decode(const VectorXd &z) {
        return m_impl.decode(z);
    }

    inline VectorXd sub_decode(const VectorXd &z) {
        return m_impl.sub_decode(z);
    }

    inline MatrixXd jacobian(const VectorXd &z) {
        return m_impl.jacobian(z);
    }

    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {
        // d decode / d z * q
        return m_impl.jacobian_transpose_vector_product(z, q);
    }

    inline MatrixXd outer_jacobian() {
        return m_impl.outer_jacobian();
    }

    inline MatrixXd inner_jacobian(const VectorXd &z) { // TODO make this sparse for linear subspace?
        return m_impl.inner_jacobian(z);
    }

    inline MatrixXd compute_reduced_mass_matrix(const MatrixXd &M) {
        return m_impl.compute_reduced_mass_matrix(M);
    }

    inline double get_energy(const VectorXd &z) {
        return m_impl.get_energy(z);
    }

private:
    ReducedSpaceImpl m_impl;
};

// class IdentitySpaceImpl
// {
// public:
//     IdentitySpaceImpl(int n) {
//         m_I.resize(n,n);
//         m_I.setIdentity();
//     }

//     inline VectorXd encode(const VectorXd &q) {return q;}
//     inline VectorXd decode(const VectorXd &z) {return z;}
//     inline VectorXd sub_decode(const VectorXd &z) {return z;}
//     inline MatrixXd jacobian(const VectorXd &z) {std::cout << "Not implemented!" << std::endl;}
//     inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {return q;}
//     inline MatrixXd compute_reduced_mass_matrix(const MatrixXd &M) {std::cout << "Not implemented!" << std::endl;}
//     inline double get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}

// private:
//     SparseMatrix<double> m_I;
// };

// class ConstraintSpaceImpl
// {
// public:
//     ConstraintSpaceImpl(const SparseMatrix<double> &P) {
//         m_P = P;
//     }

//     inline VectorXd encode(const VectorXd &q) {return m_P * q;}
//     inline VectorXd decode(const VectorXd &z) {return m_P.transpose() * z;}
//     inline VectorXd sub_decode(const VectorXd &z) {return z;}
//     inline MatrixXd jacobian(const VectorXd &z) {std::cout << "Not implemented!" << std::endl;}
//     inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {return m_P * q;}
//     inline MatrixXd compute_reduced_mass_matrix(const MatrixXd &M) {std::cout << "Not implemented!" << std::endl;}
//     inline double get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}
//     inline double get_energy_discrete(const VectorXd &z, MyWorld *world, NeohookeanTets *tets) {std::cout << "Reduced energy not implemented!" << std::endl;}

// private:
//     SparseMatrix<double> m_P;
// };

class LinearSpaceImpl
{
public:
    LinearSpaceImpl(const MatrixXd &U) : m_U(U) {
        std::cout<<"U rows: " << U.rows() << std::endl;
        std::cout<<"U cols: " << U.cols() << std::endl;

        m_inner_jac = MatrixXd::Identity(U.cols(), U.cols());
    }

    inline VectorXd encode(const VectorXd &q) {
        return m_U.transpose() * q;
    }

    inline VectorXd decode(const VectorXd &z) {
        return m_U * z;
    }

    VectorXd sub_decode(const VectorXd &z) {
        return z;
    }

    inline MatrixXd jacobian(const VectorXd &z) { return m_U; }

    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) { // TODO: do this without copy?
        return m_U.transpose() * q;
    }

    inline MatrixXd outer_jacobian() {
        return m_U;
    }

    inline MatrixXd inner_jacobian(const VectorXd &z) {
        return m_inner_jac;
    }

    inline MatrixXd compute_reduced_mass_matrix(const MatrixXd &M) {
        return m_U.transpose() * M * m_U;
    }

    inline double get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}
    inline double get_energy_discrete(const VectorXd &z, MyWorld *world, NeohookeanTets *tets) {std::cout << "Reduced energy not implemented!" << std::endl;}
private:
    MatrixXd m_U;
    MatrixXd m_inner_jac;
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

        //     status = tf::NewSession(tf::SessionOptions(), &m_energy_model_session);
        //     checkStatus(status);
        //     tf::GraphDef energy_model_graph_def;
        //     status = ReadBinaryProto(tf::Env::Default(), energy_model_path.string(), &energy_model_graph_def);
        //     checkStatus(status);
        //     status = m_energy_model_session->Create(energy_model_graph_def);
        //     checkStatus(status);
        // }


        // if(integrator_config["use_discrete_reduced_energy"]) {
        //     fs::path discrete_energy_model_path = tf_models_root / "discrete_energy_model.pb";

        //     status = tf::NewSession(tf::SessionOptions(), &m_discrete_energy_model_session);
        //     checkStatus(status);
        //     tf::GraphDef discrete_energy_model_graph_def;
        //     status = ReadBinaryProto(tf::Env::Default(), discrete_energy_model_path.string(), &discrete_energy_model_graph_def);
        //     checkStatus(status);
        //     status = m_discrete_energy_model_session->Create(discrete_energy_model_graph_def);
        //     checkStatus(status);
        // }
        // -- Testing

        // tf::Tensor z_tensor(tf::DT_FLOAT, tf::TensorShape({1, 3}));
        // auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        // z_tensor_mapped(0, 0) = 0.0;
        // z_tensor_mapped(0, 1) = 0.0;
        // z_tensor_mapped(0, 2) = 0.0;

        // tf::Tensor q_tensor(tf::DT_FLOAT, tf::TensorShape({1, 1908}));
        // auto q_tensor_mapped = q_tensor.tensor<tf_dtype_type, 2>();
        // for(int i =0; i < 1908; i++) {
        //     q_tensor_mapped(0, i) = 0.0;
        // }

        // // std::vector<std::pair<tf::string, tf::Tensor>>integrator_config["energy_model_config"]["enabled"] input_tensors = {{"x", x}, {"y", y}};
        // std::vector<tf::Tensor> q_outputs;
        // status = m_decoder_session->Run({{"decoder_input:0", z_tensor}},
        //                            {"output_node0:0"}, {}, &q_outputs);
        // checkStatus(status);

        // std::vector<tf::Tensor> z_outputs;
        // status = m_encoder_session->Run({{"encoder_input:0", q_tensor}},
        //                            {"output_node0:0"}, {}, &z_outputs);
        // checkStatus(status);

        // tf::Tensor q_output = q_outputs[0];
        // std::cout << "Success: " << q_output.flat<tf_dtype_type>() << "!" << std::endl;
        // tf::Tensor z_output = z_outputs[0];
        // std::cout << "Success: " << z_output.flat<tf_dtype_type>() << "!" << std::endl;
        // std::cout << "Jac: " << jacobian(Vector3d(0.0,0.0,0.0)) << std::endl;
    }

    inline VectorXd encode(const VectorXd &q) {
        VectorXd sub_q = m_U.transpose() * q;

        tf::Tensor sub_q_tensor(tf_dtype, tf::TensorShape({1, sub_q.size()}));
        auto sub_q_tensor_mapped = sub_q_tensor.tensor<tf_dtype_type, 2>();
        for(int i =0; i < sub_q.size(); i++) {
            sub_q_tensor_mapped(0, i) = sub_q[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> z_outputs;
        tf::Status status = m_encoder_session->Run({{"encoder_input:0", sub_q_tensor}},
                                   {"output_node0:0"}, {}, &z_outputs);

        auto z_tensor_mapped = z_outputs[0].tensor<tf_dtype_type, 2>();
        VectorXd res(m_enc_dim); // TODO generalize and below in decode
        for(int i = 0; i < res.size(); i++) {
            res[i] = z_tensor_mapped(0,i);
        }

        return res;
    }

    VectorXd sub_decode(const VectorXd &z) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
        auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        for(int i = 0; i < z.size(); i++) {
            z_tensor_mapped(0, i) = z[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> sub_q_outputs;
        tf::Status status = m_decoder_session->Run({{"decoder_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &sub_q_outputs);

        auto sub_q_tensor_mapped = sub_q_outputs[0].tensor<tf_dtype_type, 2>();
        VectorXd res(m_U.cols());
        for(int i = 0; i < res.size(); i++) {
            res[i] = sub_q_tensor_mapped(0,i);
        }

        return res;
    }

    inline VectorXd decode(const VectorXd &z) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
        auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        for(int i = 0; i < z.size(); i++) {
            z_tensor_mapped(0, i) = z[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> sub_q_outputs;
        tf::Status status = m_decoder_session->Run({{"decoder_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &sub_q_outputs);

        auto sub_q_tensor_mapped = sub_q_outputs[0].tensor<tf_dtype_type, 2>();
        VectorXd res(m_U.cols());
        for(int i = 0; i < res.size(); i++) {
            res[i] = sub_q_tensor_mapped(0,i);
        }

        return m_U * res;
    }

    inline MatrixXd jacobian(const VectorXd &z) {
        return m_U * inner_jacobian(z);

        // Analytical
        // tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
        // auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        // for(int i =0; i < z.size(); i++) {
        //     z_tensor_mapped(0, i) = z[i];
        // } // TODO map with proper function

        // std::vector<tf::Tensor> jac_outputs;
        // tf::Status status = m_decoder_jac_session->Run({{"decoder_input:0", z_tensor}},
        //                            {"TensorArrayStack/TensorArrayGatherV3:0"}, {}, &jac_outputs); // TODO get better names

        // auto jac_tensor_mapped = jac_outputs[0].tensor<tf_dtype_type, 3>();
        
        // MatrixXd res(n_dof, m_enc_dim); // TODO generalize
        // for(int i = 0; i < res.rows(); i++) {
        //     for(int j = 0; j < res.cols(); j++) {
        //         res(i,j) = jac_tensor_mapped(0,i,j);
        //     }
        // }

        // return res;
    }

    inline MatrixXd outer_jacobian() {
        return m_U;
    }

    inline MatrixXd inner_jacobian(const VectorXd &z) {
        MatrixXd sub_jac(m_sub_q_size, z.size()); // just m_U.cols()?

        // Finite differences gradient
        double t = finite_diff_eps;
        for(int i = 0; i < z.size(); i++) {
            VectorXd dz_pos(z);
            VectorXd dz_neg(z);
            dz_pos[i] += t;
            dz_neg[i] -= t;
            sub_jac.col(i) = (sub_decode(dz_pos) - sub_decode(dz_neg)) / (2.0 * t);
        }

        return sub_jac;
    }

    // Using finite differences
    // TODO I should be able to do this as a single pass right? put all the inputs into one tensor
    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) { // TODO: do this without copy?
        VectorXd res(z.size());
        VectorXd sub_q = m_U.transpose() * q;
        //Finite differences gradient
        double t = finite_diff_eps;//0.0005;
        for(int i = 0; i < z.size(); i++) {
            VectorXd dz_pos(z);
            VectorXd dz_neg(z);
            dz_pos[i] += t;
            dz_neg[i] -= t;
            // std::cout << sub_q.size() << ", " << decode(dz_pos).size() << std::endl;
            res[i] = sub_q.dot((sub_decode(dz_pos) - sub_decode(dz_neg)) / (2.0 * t));
        }

        return res;
    }

    inline MatrixXd compute_reduced_mass_matrix(const MatrixXd &M) {
        return m_U.transpose() * M * m_U;
    }

    inline double get_energy(const VectorXd &z) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
        auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        for(int i =0; i < z.size(); i++) {
            z_tensor_mapped(0, i) = z[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> q_outputs;
        tf::Status status = m_energy_model_session->Run({{"energy_model_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &q_outputs);

        auto energy_tensor_mapped = q_outputs[0].tensor<tf_dtype_type, 2>();

        return energy_tensor_mapped(0,0) * 1000.0; // TODO keep track of this normalizaiton factor
    }

    inline double get_energy_discrete(const VectorXd &z, MyWorld *world, NeohookeanTets *tets) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
        auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        for(int i =0; i < z.size(); i++) {
            z_tensor_mapped(0, i) = z[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> energy_weight_outputs;
        tf::Status status = m_discrete_energy_model_session->Run({{"energy_model_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &energy_weight_outputs);

        auto energy_weight_tensor_mapped = energy_weight_outputs[0].tensor<tf_dtype_type, 2>();


        int n_tets = tets->getImpl().getNumElements();
        double eps = 0.01;
        double scale = 1000.0; // TODO link this to training data
        double summed_energy = 0.0;
        int n_tets_used = 0;
        for(int i = 0; i < n_tets; i++) {
            double energy_weight_i = energy_weight_tensor_mapped(0, i);
            // std::cout << energy_weight_i << ", ";
            if(energy_weight_i > eps) {
                auto element = tets->getImpl().getElement(i);
                summed_energy = energy_weight_i * element->getStrainEnergy(world->getState()); //TODO is the scale right? Is getting the state slow?
                n_tets_used++;
            }
        }
        // std::cout << std::endl;
        // std::cout << "n_tets_used: " << n_tets_used << std::endl;

        return summed_energy;
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
    tf::Session* m_encoder_session;
    tf::Session* m_energy_model_session;
    tf::Session* m_discrete_energy_model_session;
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

        m_use_reduced_energy = integrator_config["use_reduced_energy"];
        m_h = integrator_config["timestep"];
        finite_diff_eps = integrator_config["finite_diff_eps"];
        // Construct mass matrix and external forces
        AssemblerEigenSparseMatrix<double> M_asm;
        getMassMatrix(M_asm, *m_world);

        std::cout << "Constructing reduced mass matrix..." << std::endl;
        m_M = *M_asm;
        m_UTMU = m_reduced_space->compute_reduced_mass_matrix(m_M);
        std::cout << "Done." << std::endl;

        VectorXd g(m_M.cols());
        for(int i=0; i < g.size(); i += 3) {
            g[i] = 0.0;
            g[i+1] = integrator_config["gravity"];
            g[i+2] = 0.0;
        }

        m_F_ext = m_M * g;
        m_interaction_force = VectorXd::Zero(m_F_ext.size());

        if(integrator_config["use_reduced_energy"]) {
            std::cout << "Loading reduced energy basis..." << std::endl;
            fs::path pca_components_path("pca_results/energy_pca_components.dmat");
            fs::path sample_tets_path("pca_results/energy_indices.dmat");

            MatrixXd U;
            igl::readDMAT((model_root / pca_components_path).string(), U);

            MatrixXi tet_indices_mat;
            igl::readDMAT((model_root / sample_tets_path).string(), tet_indices_mat);\
            std::cout << "Done." << std::endl;

            std::cout << "Factoring reduced energy basis..." << std::endl;
            
            VectorXi tet_indices = tet_indices_mat.col(0);
            m_energy_sample_tets.resize(tet_indices.size());
            VectorXi::Map(&m_energy_sample_tets[0], tet_indices.size()) = tet_indices;
            std::sort(m_energy_sample_tets.begin(), m_energy_sample_tets.end());
            for(int i=0; i < m_energy_sample_tets.size(); i++) {
                tet_indices_mat(i, 0) = m_energy_sample_tets[i];
            }

            m_energy_basis = U;
            m_energy_sampled_basis = igl::slice(m_energy_basis, tet_indices_mat, 1);
            m_summed_energy_basis = m_energy_basis.colwise().sum();
            m_energy_sampled_basis_qr = m_energy_sampled_basis.fullPivHouseholderQr();
            std::cout << "Done." << std::endl;

            std::cout << "Constructing reduced force factor..." << std::endl;
            MatrixXd A = m_energy_sampled_basis.transpose() * m_energy_sampled_basis;
            m_summed_force_fact = m_energy_sampled_basis * A.ldlt().solve(m_summed_energy_basis); //U_bar(A^-1*u)
            std::cout << "Done." << std::endl;
        }

    }

    void set_prev_zs(const VectorXd &cur_z, const VectorXd &prev_z) {
        m_cur_z = cur_z;
        m_prev_z = prev_z;
    }

    void set_interaction_force(const VectorXd &interaction_force) {
        m_interaction_force = interaction_force;
    }

    // Just short helpers
    inline VectorXd dec(const VectorXd &z) { return m_reduced_space->decode(z); }
    inline VectorXd sub_dec(const VectorXd &z) { return m_reduced_space->sub_decode(z); }
    inline VectorXd enc(const VectorXd &q) { return m_reduced_space->encode(q); }
    inline VectorXd jtvp(const VectorXd &z, const VectorXd &q) { return m_reduced_space->jacobian_transpose_vector_product(z, q); }

    double current_reduced_energy_and_forces(double &energy, VectorXd &forces) {
        int n_sample_tets = m_energy_sample_tets.size();
        
        VectorXd sampled_energy(n_sample_tets);

        MatrixXd element_forces(n_sample_tets, 12);
        SparseMatrix<double> neg_energy_sample_jac(n_dof, n_sample_tets);
        neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, 12)); // Reserve enough room for 4 verts (tet corners) per column

        #pragma omp parallel for num_threads(4) //schedule(static,64)// TODO how to optimize this
        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            int tet_index = m_energy_sample_tets[i];

            // Energy
            sampled_energy[i] = m_tets->getImpl().getElement(tet_index)->getStrainEnergy(m_world->getState());

            //Forces
            VectorXd sampled_force(12);
            m_tets->getImpl().getElement(tet_index)->getInternalForce(sampled_force, m_world->getState());
            element_forces.row(i) = sampled_force;
        }

        // Constructing sparse matrix is not thread safe
        for(int i = 0; i < n_sample_tets; i++) {
            int tet_index = m_energy_sample_tets[i];
            for(int j = 0; j < 4; j++) {
                int vert_index = m_tets->getImpl().getElement(tet_index)->getQDOFList()[j]->getGlobalId();
                for(int k = 0; k < 3; k++) {
                    neg_energy_sample_jac.insert(vert_index + k, i) = element_forces(i, j*3 + k);
                }
            }
        }
        neg_energy_sample_jac.makeCompressed();

        // VectorXd alpha = m_energy_sampled_basis.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(sampled_energy);
        // VectorXd alpha = (m_energy_sampled_basis.transpose() * m_energy_sampled_basis).ldlt().solve(m_energy_sampled_basis.transpose() * sampled_energy);
        VectorXd alpha = m_energy_sampled_basis_qr.solve(sampled_energy);
        energy = m_summed_energy_basis.dot(alpha);

        forces = neg_energy_sample_jac * m_summed_force_fact;
    }


    double operator()(const VectorXd& new_z, VectorXd& grad)
    {
        // Optimizations
        // Slowest part by far is get reduced forces %55
        // Sparse matrix is some of it but not much

        // nn decode is like ~4%
        // Copy constructor? 8%
        // FullPivHouseholder solve -> determine if the fuull piv vs other method makes a deff

        // std::cout << "z: <";
        // for(int i = 0; i < new_z.size(); i++) {
        //     std::cout << new_z[i];
        //     if(i != new_z.size() - 1) {
        //         std::cout << ", ";
        //     }
        // }
        // std::cout << ">" << std::endl;
        MatrixXd U = m_reduced_space->outer_jacobian(); // U for both linear and ae space

        // Update the tets with candidate configuration
        Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(m_tets->getQ(), *m_world);
        VectorXd new_sub_q = sub_dec(new_z);
        VectorXd new_q = U * new_sub_q;
        for(int i=0; i < q.size(); i++) {
            // ******************************************
            // TODO also I really only have to update the tets that I'm sampling, so I can avoid a full decoder here right??
            // ******************************************
            q[i] = new_q[i]; // TODO is this the fastest way to do this?
        }
        
        // Compute objective
        double energy;
        VectorXd internal_forces;
        if(m_use_reduced_energy) {
            current_reduced_energy_and_forces(energy, internal_forces);
        }
        else {
            energy = m_tets->getStrainEnergy(m_world->getState());
            AssemblerEigenVector<double> internal_force_asm;
            getInternalForceVector(internal_force_asm, *m_tets, *m_world);
            internal_forces = *internal_force_asm; // TODO can we avoid copy here?
        }

        // *****
        // TODO reduce this term
        // *****
        VectorXd h_2_external_forces = m_h * m_h * (m_F_ext + m_interaction_force); // Same with the interaction forces. I can precompute th gravity and quickly reduce interaction
        VectorXd h_2_UT_external_forces =  U.transpose() * h_2_external_forces;
        
        // MatrixXd inner_jac = m_reduced_space->inner_jacobian(new_z);

        VectorXd sub_q_tilde = new_sub_q - 2.0 * sub_dec(m_cur_z) + sub_dec(m_prev_z);
        VectorXd UTMU_sub_q_tilde = m_UTMU * sub_q_tilde;
        double obj_val = 0.5 * sub_q_tilde.transpose() * UTMU_sub_q_tilde
                            + m_h * m_h * energy
                            - sub_q_tilde.transpose() * h_2_UT_external_forces;

        // Compute gradient
        // **** TODO
        // Can I further reduce the force calculations by carrying through U?
        MatrixXd J = m_reduced_space->inner_jacobian(new_z); // should be identity for linear
        grad = J.transpose() * (UTMU_sub_q_tilde - m_h * m_h * U.transpose() * internal_forces - h_2_UT_external_forces);
        return obj_val;

        // MatrixXd J = m_reduced_space->jacobian(new_z);
        // static int cur_frame = 0;
        // std::stringstream jac_filename;
        // jac_filename << "jacs/" << "jac_" << cur_frame++ << ".dmat";
        // igl::writeDMAT(jac_filename.str(), J, false); // Don't use ascii
        // VectorXd z_tilde = new_z - 2.0 * m_cur_z + m_prev_z;
        // VectorXd J_z_tilde = J * z_tilde;
        // VectorXd M_J_z_tilde = m_M * J_z_tilde;
        // double obj_val = 0.5 * J_z_tilde.transpose() * M_J_z_tilde + m_h * m_h * energy - J_z_tilde.transpose() * external_forces_h2;
        // std::cout << (q_tilde.transpose() * M_q_tilde) << ", " << (J_z_tilde.transpose() * M_J_z_tilde) << std::endl;

        // std::cout << m_reduced_space->jacobian(new_z) << std::endl;
        // for(int i = 900; i < 1000; i++) {
        //     std::cout << q_tilde[i] << ", " << J_z_tilde[i] << std::endl;
        // }
        // std::cout << std::endl;
    }

private:
    VectorXd m_cur_z;
    VectorXd m_prev_z;
    VectorXd m_F_ext;
    VectorXd m_interaction_force;

    SparseMatrix<double> m_M; // mass matrix
    MatrixXd m_UTMU; // reduced mass matrix

    double m_h;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    bool m_use_reduced_energy;
    ReducedSpaceType *m_reduced_space;

    std::vector<int> m_energy_sample_tets;
    MatrixXd m_energy_basis;
    VectorXd m_summed_energy_basis;
    MatrixXd m_energy_sampled_basis;
    Eigen::FullPivHouseholderQR<MatrixXd> m_energy_sampled_basis_qr;

    MatrixXd m_summed_force_fact;
};

template <typename ReducedSpaceType>
class GPLCTimeStepper {
public:
    GPLCTimeStepper(fs::path model_root, json integrator_config, MyWorld *world, NeohookeanTets *tets, ReducedSpaceType *reduced_space) :
        m_world(world), m_tets(tets), m_reduced_space(reduced_space) {

       // LBFGSpp::LBFGSParam<DataType> param;
       // param.epsilon = 1e-4;
       // param.max_iterations = 1000;
       // param.past = 1;
       // param.m = 5;
       // param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
        m_lbfgs_param.m = 5;
        m_lbfgs_param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;

        // Set up lbfgs params
        json lbfgs_config = integrator_config["lbfgs_config"];
        m_lbfgs_param.epsilon = lbfgs_config["lbfgs_epsilon"]; //1e-8// TODO replace convergence test with abs difference
        m_lbfgs_param.max_iterations = lbfgs_config["lbfgs_max_iterations"];//500;//300;
        m_solver = new LBFGSSolver<double>(m_lbfgs_param);

        reset_zs_to_current_world();
        m_gplc_objective = new GPLCObjective<ReducedSpaceType>(model_root, integrator_config, m_cur_z, m_prev_z, world, tets, reduced_space);

        m_h = integrator_config["timestep"];
        MatrixXd U = reduced_space->outer_jacobian();
        MatrixXd J = reduced_space->inner_jacobian(m_cur_z);

        AssemblerEigenSparseMatrix<double> M_asm;
        AssemblerEigenSparseMatrix<double> K_asm;
        getMassMatrix(M_asm, *m_world);
        getStiffnessMatrix(K_asm, *m_world);

        std::cout << "Constructing reduced mass and stiffness matrix..." << std::endl;
        m_UTMU = U.transpose() * (*M_asm) * U;
        m_UTKU = U.transpose() * (*K_asm) * U;
        m_H = J.transpose() * m_UTMU * J - m_h * m_h * J.transpose() * m_UTKU * J;
        m_H_llt = m_H.ldlt();
        // m_H_llt = MatrixXd::Identity(m_H.rows(), m_H.cols()).llt();
    }

    ~GPLCTimeStepper() {
        delete m_solver;
        delete m_gplc_objective;
    }

    void step(const VectorXd &interaction_force) {
        double start_time = igl::get_seconds();

        VectorXd z_param = m_cur_z; // Stores both the first guess and the final result
        double min_val_res;

        m_gplc_objective->set_interaction_force(interaction_force);
        m_gplc_objective->set_prev_zs(m_cur_z, m_prev_z);
        

        // Conclusion
        // For the AE model at least, preconditioning with rest hessian is slower
        // but preconditioning with rest hessian with current J gives fairly small speed increas
        // TODO: determine if llt or ldlt is faster
        // int niter = m_solver->minimize(*m_gplc_objective, z_param, min_val_res);   
        MatrixXd J = m_reduced_space->inner_jacobian(m_cur_z);
        m_H = J.transpose() * m_UTMU * J - m_h * m_h * J.transpose() * m_UTKU * J;
        m_H_llt = m_H_llt.compute(m_H);//m_H.ldlt();
        int niter = m_solver->minimizeWithPreconditioner(*m_gplc_objective, z_param, min_val_res, m_H_llt);   
        
        std::cout << niter << " iterations" << std::endl;
        std::cout << "objective val = " << min_val_res << std::endl;

        m_prev_z = m_cur_z;
        m_cur_z = z_param; // TODO: Use pointers to avoid copies

        update_world_with_current_configuration();

        double update_time = igl::get_seconds() - start_time;
        m_total_time += update_time;
        m_current_frame++;
        std::cout << "Timestep took: " << update_time << "s" << std::endl;
        std::cout << "Avg timestep: " << m_total_time / (double)m_current_frame << "s" << std::endl;
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

    // void test_gradient() {
    //     auto q_map = mapDOFEigen(m_tets->getQ(), *m_world);
    //     VectorXd q(m_reduced_space->encode(q_map));

    //     GPLCObjective<ReducedSpaceType> fun(q, q, 0.001, m_world, m_tets);
        
    //     VectorXd fun_grad(q.size());
    //     double fun_val = fun(q, fun_grad);

    //     VectorXd finite_diff_grad(q.size());
    //     VectorXd empty(q.size());
    //     double t = 0.000001;
    //     for(int i = 0; i < q.size(); i++) {
    //         VectorXd dq_pos(q);
    //         VectorXd dq_neg(q);
    //         dq_pos[i] += t;
    //         dq_neg[i] -= t;

    //         finite_diff_grad[i] = (fun(dq_pos, empty) - fun(dq_neg, empty)) / (2.0 * t);
    //     }

    //     VectorXd diff = fun_grad - finite_diff_grad;
    //     std::cout << "q size: " << q.size() << std::endl;
    //     std::cout << "Function val: " << fun_val << std::endl;
    //     std::cout << "Gradient diff: " << diff.norm() << std::endl;
    //     assert(diff.norm() < 1e-4);
    // }

private:
    LBFGSParam<double> m_lbfgs_param;
    LBFGSSolver<double> *m_solver;
    GPLCObjective<ReducedSpaceType> *m_gplc_objective;

    VectorXd m_prev_z;
    VectorXd m_cur_z;

    double m_h;

    MatrixXd m_H;
    Eigen::LDLT<MatrixXd> m_H_llt;
    MatrixXd m_H_inv;
    MatrixXd m_UTKU;
    MatrixXd m_UTMU;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    ReducedSpaceType *m_reduced_space;

    int m_current_frame = 0;
    double m_total_time = 0.0;
};

SparseMatrix<double> construct_constraints_P(NeohookeanTets *tets) {
    // Construct constraint projection matrix
    // Fixes all vertices with minimum x coordinate to 0
    int dim = 0; // x
    auto min_x_val = tets->getImpl().getV()(0,dim);
    std::vector<unsigned int> min_verts;

    for(unsigned int ii=0; ii<tets->getImpl().getV().rows(); ++ii) {
        if(tets->getImpl().getV()(ii,dim) < min_x_val) {
            min_x_val = tets->getImpl().getV()(ii,dim);
            min_verts.clear();
            min_verts.push_back(ii);
        } else if(fabs(tets->getImpl().getV()(ii,dim) - min_x_val) < 1e-5) {
            min_verts.push_back(ii);
        }
    }

    std::sort(min_verts.begin(), min_verts.end());
    int n = tets->getImpl().getV().rows() * 3;
    int m = n - min_verts.size()*3;
    SparseMatrix<double> P(m, n);
    P.reserve(VectorXi::Constant(n, 1)); // Reserve enough space for 1 non-zero per column
    int min_vert_i = 0;
    int cur_col = 0;
    for(int i = 0; i < m; i+=3){
        while(min_verts[min_vert_i] * 3 == cur_col) { // Note * is for vert index -> flattened index
            cur_col += 3;
            min_vert_i++;
        }
        P.insert(i, cur_col) = 1.0;
        P.insert(i+1, cur_col+1) = 1.0;
        P.insert(i+2, cur_col+2) = 1.0;
        cur_col += 3;
    }
    P.makeCompressed();
    // std::cout << P << std::endl;
    // -- Done constructing P
    return P;
}

VectorXd compute_interaction_force(const Vector3d &dragged_pos, int dragged_vert, bool is_dragging, double spring_stiffness, NeohookeanTets *tets, const MyWorld &world) {
    VectorXd force = VectorXd::Zero(n_dof); // TODO make this a sparse vector?

    if(is_dragging) {
        Vector3d fem_attached_pos = PosFEM<double>(&tets->getQ()[dragged_vert], dragged_vert, &tets->getImpl().getV())(world.getState());
        Vector3d local_force = spring_stiffness * (dragged_pos - fem_attached_pos);

        for(int i=0; i < 3; i++) {
            force[dragged_vert * 3 + i] = local_force[i];    
        }
    } 

    return force;
}

VectorXd get_von_mises_stresses(NeohookeanTets *tets, MyWorld &world) {
    // Compute cauchy stress
    int n_ele = tets->getImpl().getF().rows();

    Eigen::Matrix<double, 3,3> stress = MatrixXd::Zero(3,3);
    VectorXd vm_stresses(n_ele);

    for(unsigned int i=0; i < n_ele; ++i) {
        tets->getImpl().getElement(i)->getCauchyStress(stress, Vec3d(0,0,0), world.getState());

        double s11 = stress(0, 0);
        double s22 = stress(1, 1);
        double s33 = stress(2, 2);
        double s23 = stress(1, 2);
        double s31 = stress(2, 0);
        double s12 = stress(0, 1);
        vm_stresses[i] = sqrt(0.5 * (s11-s22)*(s11-s22) + (s33-s11)*(s33-s11) + 6.0 * (s23*s23 + s31*s31 + s12*s12));
    }

    return vm_stresses;
}



// typedef ReducedSpace<IdentitySpaceImpl> IdentitySpace;
// typedef ReducedSpace<ConstraintSpaceImpl> ConstraintSpace;
typedef ReducedSpace<LinearSpaceImpl> LinearSpace;
typedef ReducedSpace<AutoEncoderSpaceImpl> AutoencoderSpace;

template <typename ReducedSpaceType>
void run_sim(ReducedSpaceType *reduced_space, const json &config, const fs::path &model_root) {
    // -- Setting up GAUSS
    MyWorld world;
    igl::readMESH((model_root / "tets.mesh").string(), V, T, F);

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
    VectorXd interaction_force = VectorXd::Zero(n_dof);


    GPLCTimeStepper<ReducedSpaceType> gplc_stepper(model_root, integrator_config, &world, tets, reduced_space);

    bool show_stress = visualization_config["show_stress"];

    /** libigl display stuff **/
    igl::viewer::Viewer viewer;

    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
        // predefined colors
        const Eigen::RowVector3d orange(1.0,0.7,0.2);
        const Eigen::RowVector3d yellow(1.0,0.9,0.2);
        const Eigen::RowVector3d blue(0.2,0.3,0.8);
        const Eigen::RowVector3d green(0.2,0.6,0.3);

        if(is_dragging) {
            Eigen::MatrixXd part_pos(1, 3);
            part_pos(0,0) = dragged_pos[0]; // TODO why is eigen so confusing.. I just want to make a matrix from vec
            part_pos(0,1) = dragged_pos[1];
            part_pos(0,2) = dragged_pos[2];

            viewer.data.set_points(part_pos, orange);
        } else {
            Eigen::MatrixXd part_pos = MatrixXd::Zero(1,3);
            part_pos(0,0)=100000.0;
            viewer.data.set_points(part_pos, orange);
        }

        if(viewer.core.is_animating)
        {   
            // Save Current configuration
            std::stringstream filename;
            filename << output_dir << "displacements_" << current_frame << ".dmat";
            if(saving_training_data) {
                save_displacements_DMAT(filename.str(), world, tets);
            }

            // stepper.step(world);
            auto q = mapDOFEigen(tets->getQ(), world);
            // std::cout << "Potential = " << tets->getStrainEnergy(world.getState()) << std::endl;

            Eigen::MatrixXd newV = getCurrentVertPositions(world, tets); 
            // std::cout<< newV.block(0,0,10,3) << std::endl;
            viewer.data.set_vertices(newV);
            viewer.data.compute_normals();
            current_frame++;

            interaction_force = compute_interaction_force(dragged_pos, dragged_vert, is_dragging, spring_stiffness, tets, world);
            gplc_stepper.step(interaction_force);
            // test_gradient(world, tets);
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
            std::cout << vm_per_face.maxCoeff() << " " <<  vm_per_face.minCoeff() << std::endl;
            MatrixXd C;
            //VectorXd((vm_per_face.array() -  vm_per_face.minCoeff()) / vm_per_face.maxCoeff())
            igl::jet(vm_per_face / 60.0, false, C);
            viewer.data.set_colors(C);
        }

        return false;
    };

    viewer.callback_key_pressed = [&](igl::viewer::Viewer &, unsigned int key, int mod)
    {
        switch(key)
        {
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
            default:
            return false;
        }
        return true;
    };

    viewer.callback_mouse_down = [&](igl::viewer::Viewer&, int, int)->bool
    {   
        Eigen::MatrixXd curV = getCurrentVertPositions(world, tets); 
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
            dragged_vert = F(fid,c);
            is_dragging = true;


            // forceSpring->getImpl().setStiffness(spring_stiffness);
            // auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
            // pinned_q = dragged_pos;//(dragged_pos).cast<double>(); // necessary?

            // fem_attached_pos = PosFEM<double>(&tets->getQ()[dragged_vert],dragged_vert, &tets->getImpl().getV());
            // forceSpring->getImpl().setPosition0(fem_attached_pos);

            return true;
        }
        
        return false; // TODO false vs true??
    };

    viewer.callback_mouse_up = [&](igl::viewer::Viewer&, int, int)->bool
    {
        is_dragging = false;
        return false;
    };

    viewer.callback_mouse_move = [&](igl::viewer::Viewer &, int,int)->bool
    {
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

            last_mouse = drag_mouse;
        }

        return false;
    };

    viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;

    viewer.launch();
}

int main(int argc, char **argv) {
    
    // Load the configuration file
    fs::path model_root(argv[1]);
    fs::path sim_config("sim_config.json");

    std::ifstream fin((model_root / sim_config).string());
    json config;
    fin >> config;

    auto integrator_config = config["integrator_config"];
    std::string reduced_space_string = integrator_config["reduced_space_type"];

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
        run_sim<AutoencoderSpace>(&reduced_space, config, model_root);  
    }
    else if(reduced_space_string == "full") {
        std::cout << "Not yet implemented." << std::endl;
        // ConstraintSpace reduced_space(construct_constraints_P(tets));
        // run_sim<ConstraintSpace>(&reduced_space, config, model_root);
    }
    else {
        std::cout << "Not yet implemented." << std::endl;
        return 1;
    }
    
    return EXIT_SUCCESS;
}

