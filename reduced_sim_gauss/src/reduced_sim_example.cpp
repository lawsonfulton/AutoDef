#include <vector>

//#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>

#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/get_seconds.h>
#include <igl/jet.h>

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

using json = nlohmann::json;
namespace fs = boost::filesystem;

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::VectorXi;
using Eigen::MatrixXd;
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
typedef TimeStepperEulerImplictLinear<double, AssemblerEigenSparseMatrix<double>,
AssemblerEigenVector<double> > MyTimeStepper;

// Mesh
Eigen::MatrixXd V; // Verts
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices
int n_dof;

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

    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {
        // d decode / d z * q
        return m_impl.jacobian_transpose_vector_product(z, q);
    }

    inline double get_energy(const VectorXd &z) {
        return m_impl.get_energy(z);
    }

private:
    ReducedSpaceImpl m_impl;
};

class IdentitySpaceImpl
{
public:
    IdentitySpaceImpl(int n) {
        m_I.resize(n,n);
        m_I.setIdentity();
    }

    inline VectorXd encode(const VectorXd &q) {return q;}
    inline VectorXd decode(const VectorXd &z) {return z;}
    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {return q;}
    inline double get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}

private:
    SparseMatrix<double> m_I;
};

class ConstraintSpaceImpl
{
public:
    ConstraintSpaceImpl(const SparseMatrix<double> &P) {
        m_P = P;
    }

    inline VectorXd encode(const VectorXd &q) {return m_P * q;}
    inline VectorXd decode(const VectorXd &z) {return m_P.transpose() * z;}
    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) {return m_P * q;}
    inline double get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}

private:
    SparseMatrix<double> m_P;
};

class LinearSpaceImpl
{
public:
    LinearSpaceImpl(const MatrixXd &U) : m_U(U) {
        std::cout<<"U rows: " << U.rows() << std::endl;
        std::cout<<"U cols: " << U.cols() << std::endl;
    }

    inline VectorXd encode(const VectorXd &q) {
        return m_U * q;
    }

    inline VectorXd decode(const VectorXd &z) {
        return m_U.transpose() * z;
    }

    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) { // TODO: do this without copy?
        return m_U * q;
    }

    inline double get_energy(const VectorXd &z) {std::cout << "Reduced energy not implemented!" << std::endl;}

private:
    MatrixXd m_U;
};

namespace tf = tensorflow;
auto tf_dtype = tf::DT_DOUBLE;
typedef double tf_dtype_type;

class AutoEncoderSpaceImpl
{
public:
    AutoEncoderSpaceImpl(fs::path tf_models_root, json integrator_config) {
        m_enc_dim = integrator_config["ae_encoded_dim"];

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


        if(integrator_config["use_reduced_energy"]) {
            fs::path energy_model_path = tf_models_root / "energy_model.pb";

            status = tf::NewSession(tf::SessionOptions(), &m_energy_model_session);
            checkStatus(status);
            tf::GraphDef energy_model_graph_def;
            status = ReadBinaryProto(tf::Env::Default(), energy_model_path.string(), &energy_model_graph_def);
            checkStatus(status);
            status = m_energy_model_session->Create(energy_model_graph_def);
            checkStatus(status);
        }
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
        tf::Tensor q_tensor(tf_dtype, tf::TensorShape({1, n_dof}));
        auto q_tensor_mapped = q_tensor.tensor<tf_dtype_type, 2>();
        for(int i =0; i < q.size(); i++) {
            q_tensor_mapped(0, i) = q[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> z_outputs;
        tf::Status status = m_encoder_session->Run({{"encoder_input:0", q_tensor}},
                                   {"output_node0:0"}, {}, &z_outputs);

        auto z_tensor_mapped = z_outputs[0].tensor<tf_dtype_type, 2>();
        VectorXd res(m_enc_dim); // TODO generalize and below in decode
        for(int i = 0; i < res.size(); i++) {
            res[i] = z_tensor_mapped(0,i);
        }

        return res;
    }

    inline VectorXd decode(const VectorXd &z) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
        auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        for(int i =0; i < z.size(); i++) {
            z_tensor_mapped(0, i) = (float)z[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> q_outputs;
        tf::Status status = m_decoder_session->Run({{"decoder_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &q_outputs);

        auto q_tensor_mapped = q_outputs[0].tensor<tf_dtype_type, 2>();
        VectorXd res(n_dof);
        for(int i = 0; i < res.size(); i++) {
            res[i] = q_tensor_mapped(0,i);
        }

        return res;
    }

    // Using finite differences
    // TODO I should be able to do this as a single pass right? put all the inputs into one tensor
    inline VectorXd jacobian_transpose_vector_product(const VectorXd &z, const VectorXd &q) { // TODO: do this without copy?
        VectorXd res(z.size());

        //Finite differences gradient
        double t = 0.0005;
        for(int i = 0; i < z.size(); i++) {
            VectorXd dz_pos(z);
            VectorXd dz_neg(z);
            dz_pos[i] += t;
            dz_neg[i] -= t;
            res[i] = q.dot((decode(dz_pos) - decode(dz_neg)) / (2.0 * t));
        }

        return res;
    }

    // Analytical
    // inline VectorXd jacobian_transpose_vector_product(const VectorXd &z) { // TODO: do this without copy?
    //     tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
    //     auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
    //     for(int i =0; i < z.size(); i++) {
    //         z_tensor_mapped(0, i) = (float)z[i];
    //     } // TODO map with proper function

    //     std::vector<tf::Tensor> jac_outputs;
    //     tf::Status status = m_decoder_jac_session->Run({{"decoder_input:0", z_tensor}},
    //                                {"TensorArrayStack/TensorArrayGatherV3:0"}, {}, &jac_outputs); // TODO get better names

    //     auto jac_tensor_mapped = jac_outputs[0].tensor<tf_dtype_type, m_enc_dim>();
        
    //     MatrixXd res(n_dof, m_enc_dim); // TODO generalize
    //     for(int i = 0; i < res.rows(); i++) {
    //         for(int j = 0; j < res.cols(); j++) {
    //             res(i,j) = jac_tensor_mapped(0,i,j);
    //         }
    //     }

    //     return res;
    // }

    inline double get_energy(const VectorXd &z) {
        tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, m_enc_dim})); // TODO generalize
        auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
        for(int i =0; i < z.size(); i++) {
            z_tensor_mapped(0, i) = (float)z[i];
        } // TODO map with proper function

        std::vector<tf::Tensor> q_outputs;
        tf::Status status = m_energy_model_session->Run({{"energy_model_input:0", z_tensor}},
                                   {"output_node0:0"}, {}, &q_outputs);

        auto energy_tensor_mapped = q_outputs[0].tensor<tf_dtype_type, 2>();

        return energy_tensor_mapped(0,0) * 1000.0; // TODO keep track of this normalizaiton factor
    }

    void checkStatus(const tf::Status& status) {
      if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        exit(1);
      }
    }

private:
    int m_enc_dim;

    tf::Session* m_decoder_session;
    tf::Session* m_decoder_jac_session;
    tf::Session* m_encoder_session;
    tf::Session* m_energy_model_session;
};

template <typename ReducedSpaceType>
class GPLCObjective
{
public:
    GPLCObjective(
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
        // Construct mass matrix and external forces
        AssemblerEigenSparseMatrix<double> M_asm;
        getMassMatrix(M_asm, *m_world);

        m_M = *M_asm;
        VectorXd g(m_M.cols());
        for(int i=0; i < g.size(); i += 3) {
            g[i] = 0.0;
            g[i+1] = integrator_config["gravity"];
            g[i+2] = 0.0;
        }

        m_F_ext = m_M * g;
        m_interaction_force = VectorXd::Zero(m_F_ext.size());
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
    inline VectorXd enc(const VectorXd &q) { return m_reduced_space->encode(q); }
    inline VectorXd jtvp(const VectorXd &z, const VectorXd &q) { return m_reduced_space->jacobian_transpose_vector_product(z, q); }

    double objective(const VectorXd &new_z) { // TODO delete when done with finite differnece
        // -- Compute GPLCObjective objective
        double energy, reduced_energy;
        // if(m_use_reduced_energy) {
            reduced_energy = m_reduced_space->get_energy(new_z);
        // }
        // else {
            Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(m_tets->getQ(), *m_world);
            VectorXd new_q = dec(new_z);
            for(int i=0; i < q.size(); i++) {
                q[i] = new_q[i]; // TODO is this the fastest way to do this?
            }
            // energy = m_tets->getStrainEnergy(m_world->getState());
        // }


        // std::cout << "Energy: " << energy << ", Reduced Energy: " << reduced_energy << std::endl;
        // std::cout << m_use_reduced_energy << std::endl;
        if(m_use_reduced_energy) {
            energy = reduced_energy;
        }
        VectorXd A = new_q - 2.0 * dec(m_cur_z) + dec(m_prev_z);
        return 0.5 * A.transpose() * m_M * A + m_h * m_h * energy - m_h * m_h * A.transpose() * (m_F_ext + m_interaction_force);
    }

    double operator()(const VectorXd& new_z, VectorXd& grad)
    {
        // Update the tets with candidate configuration
        // Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(m_tets->getQ(), *m_world);
        // VectorXd new_q = dec(new_z);
        // for(int i=0; i < q.size(); i++) {
        //     q[i] = new_q[i]; // TODO is this the fastest way to do this?
        // }

        // -- Compute GPLCObjective objective
        double obj_val = 0.0;
        // VectorXd A = new_q - 2.0 * dec(m_cur_z) + dec(m_prev_z); // TODO: avoid decodes here by saving the qs aswell

        obj_val = objective(new_z);
        // double energy;
        // if(m_use_reduced_energy) {
        //     energy = m_reduced_space->get_energy(z);
        // }
        // else {
        //     energy = m_tets->getStrainEnergy(m_world->getState());
        // }

        // obj_val = 0.5 * A.transpose() * m_M * A + m_h * m_h * energy - m_h * m_h * A.transpose() * (m_F_ext + m_interaction_force);

        if(m_use_reduced_energy) {
            // Finite differences gradient
            double t = 0.0005;
            for(int i = 0; i < new_z.size(); i++) {
                VectorXd dq_pos(new_z);
                VectorXd dq_neg(new_z);
                dq_pos[i] += t;
                dq_neg[i] -= t;
                grad[i] = (objective(dq_pos) - obj_val) / (1.0 * t);
            }
        } else {
            VectorXd new_q = dec(new_z);
            VectorXd A = new_q - 2.0 * dec(m_cur_z) + dec(m_prev_z);
            // -- Compute gradient
            AssemblerEigenVector<double> internal_force; //maybe?
            ASSEMBLEVECINIT(internal_force, new_q.size());
            m_tets->getInternalForce(internal_force, m_world->getState());
            ASSEMBLEEND(internal_force);

            grad = jtvp(new_z, m_M * A - m_h * m_h * (*internal_force) - m_h * m_h * (m_F_ext + m_interaction_force));
        }

        // std::cout << "Objective: " << obj_val << std::endl;
        // std::cout << "grad: " << grad << std::endl;
        // std::cout << "z: " << new_z << std::endl;
        return obj_val;
    }
private:
    VectorXd m_cur_z;
    VectorXd m_prev_z;
    VectorXd m_F_ext;
    VectorXd m_interaction_force;

    SparseMatrix<double> m_M; // mass matrix

    double m_h;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    bool m_use_reduced_energy;
    ReducedSpaceType *m_reduced_space;
};

template <typename ReducedSpaceType>
class GPLCTimeStepper {
public:
    GPLCTimeStepper(json integrator_config, MyWorld *world, NeohookeanTets *tets, ReducedSpaceType *reduced_space) :
        m_world(world), m_tets(tets), m_reduced_space(reduced_space) {
        // Set up lbfgs params
        json lbfgs_config = integrator_config["lbfgs_config"];
        m_lbfgs_param.epsilon = lbfgs_config["lbfgs_epsilon"]; //1e-8// TODO replace convergence test with abs difference
        //m_lbfgs_param.delta = 1e-3;
        m_lbfgs_param.max_iterations = lbfgs_config["lbfgs_max_iterations"];//500;//300;
        m_solver = new LBFGSSolver<double>(m_lbfgs_param);

        reset_zs_to_current_world();
        m_gplc_objective = new GPLCObjective<ReducedSpaceType>(integrator_config, m_cur_z, m_prev_z, world, tets, reduced_space);
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
        int niter = m_solver->minimize(*m_gplc_objective, z_param, min_val_res);

        std::cout << niter << " iterations" << std::endl;
        std::cout << "objective val = " << min_val_res << std::endl;

        m_prev_z = m_cur_z;
        m_cur_z = z_param; // TODO: Use pointers to avoid copies

        update_world_with_current_configuration();

        std::cout << "Timestep took: " << igl::get_seconds() - start_time << "s" << std::endl;
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

    void test_gradient() {
        auto q_map = mapDOFEigen(m_tets->getQ(), *m_world);
        VectorXd q(m_reduced_space->encode(q_map));

        GPLCObjective<ReducedSpaceType> fun(q, q, 0.001, m_world, m_tets);
        
        VectorXd fun_grad(q.size());
        double fun_val = fun(q, fun_grad);

        VectorXd finite_diff_grad(q.size());
        VectorXd empty(q.size());
        double t = 0.000001;
        for(int i = 0; i < q.size(); i++) {
            VectorXd dq_pos(q);
            VectorXd dq_neg(q);
            dq_pos[i] += t;
            dq_neg[i] -= t;

            finite_diff_grad[i] = (fun(dq_pos, empty) - fun(dq_neg, empty)) / (2.0 * t);
        }

        VectorXd diff = fun_grad - finite_diff_grad;
        std::cout << "q size: " << q.size() << std::endl;
        std::cout << "Function val: " << fun_val << std::endl;
        std::cout << "Gradient diff: " << diff.norm() << std::endl;
        assert(diff.norm() < 1e-4);
    }

private:
    LBFGSParam<double> m_lbfgs_param;
    LBFGSSolver<double> *m_solver;
    GPLCObjective<ReducedSpaceType> *m_gplc_objective;

    VectorXd m_prev_z;
    VectorXd m_cur_z;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    ReducedSpaceType *m_reduced_space;
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



typedef ReducedSpace<IdentitySpaceImpl> IdentitySpace;
typedef ReducedSpace<ConstraintSpaceImpl> ConstraintSpace;
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


    GPLCTimeStepper<ReducedSpaceType> gplc_stepper(integrator_config, &world, tets, reduced_space);

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

