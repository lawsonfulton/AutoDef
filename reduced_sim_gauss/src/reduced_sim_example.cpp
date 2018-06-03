#include <vector>
#include <set>

// Autodef
#include "TypeDefs.h"
#include "AutoDefUtils.h"
#include "ReducedSpace.h"

#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>
#include <AssemblerParallel.h>



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


#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/unproject.h>
#include <igl/get_seconds.h>
#include <igl/jet.h>
#include <igl/slice.h>
#include <igl/per_corner_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/material_colors.h>
#include <igl/snap_points.h>
#include <igl/centroid.h>
#include <igl/LinSpaced.h>

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


using namespace LBFGSpp;


// Mesh
Eigen::MatrixXd V; // Verts
Eigen::MatrixXd N; // Normals
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices
Eigen::MatrixXi T_sampled;
Eigen::MatrixXi F_sampled;
int n_dof;

// Mouse/Viewer state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
Eigen::RowVector3d mesh_pos;
bool is_dragging = false;
bool per_vertex_normals = false;
int dragged_vert = 0;
int current_frame = 0;

// Parameters
int print_freq = 40;
bool LOGGING_ENABLED = false;
bool PLAYBACK_SIM = false;
int log_save_freq = 5;
json sim_log;
json sim_playback_json;
json timestep_info; // Kind of hack but this gets reset at the beginning of each timestep so that we have a global structure to log to
fs::path log_dir;
fs::path surface_obj_dir;
fs::path evaluation_ply_dir;
fs::path pointer_obj_dir;
fs::path tet_obj_dir;
std::ofstream log_ofstream;

const Eigen::RowVector3d sea_green(229./255.,211./255.,91./255.);


std::vector<int> get_min_verts(int axis, bool get_max = false, double tol = 0.001) {
    int dim = axis; // x
    double min_x_val = get_max ?  V.col(dim).maxCoeff() : V.col(dim).minCoeff();
    std::vector<int> min_verts;

    for(int ii=0; ii<V.rows(); ++ii) {
        if(fabs(V(ii, dim) - min_x_val) < tol) {
            min_verts.push_back(ii);
        }
    }

    return min_verts;
}

// SparseMatrix<double> construct_constraints_P(const MatrixXd &V, std::vector<unsigned int> &verts) {
//     // Construct constraint projection matrix
//     std::cout << "Constructing constraints..." << std::endl;
//     std::sort(verts.begin(), verts.end());
//     int q_size = V.rows() * 3; // n dof
//     int n = q_size;
//     int m = n - verts.size()*3;
//     SparseMatrix<double> P(m, n);
//     P.reserve(VectorXi::Constant(n, 1)); // Reserve enough space for 1 non-zero per column
//     int min_vert_i = 0;
//     int cur_col = 0;
//     for(int i = 0; i < m; i+=3){
//         while(verts[min_vert_i] * 3 == cur_col) { // Note * is for vert index -> flattened index
//             cur_col += 3;
//             min_vert_i++;
//         }
//         P.insert(i, cur_col) = 1.0;
//         P.insert(i+1, cur_col+1) = 1.0;
//         P.insert(i+2, cur_col+2) = 1.0;
//         cur_col += 3;
//     }
//     P.makeCompressed();
//     std::cout << "Done." << std::endl;
//     // -- Done constructing P
//     return P;
// }

Eigen::SparseMatrix<double> construct_constraints_P(const MatrixXd &V, std::vector<int> &indices) {
    
    std::vector<Eigen::Triplet<double> > triplets;
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXi sortedIndices = VectorXi::Map(indices.data(), indices.size());
    std::sort(sortedIndices.data(), sortedIndices.data()+indices.size());
    
    //build a projection matrix P which projects fixed points out of a physical syste
    int fIndex = 0;
    
    //total number of DOFS in system
    
    unsigned int n = V.rows() * 3;
    unsigned int m = n - 3*indices.size();
    
    P.resize(m,n);
    
    //number of unconstrained DOFs
    unsigned int rowIndex =0;
    for(unsigned int vIndex = 0; vIndex < V.rows(); vIndex++) {
        
        while((vIndex < V.rows()) && (fIndex < sortedIndices.rows()) &&(vIndex == sortedIndices[fIndex])) {
            fIndex++;
            vIndex++;
        }
        
        if(vIndex == V.rows())
            break;
        
        //add triplet into matrix
        triplets.push_back(Eigen::Triplet<double>(rowIndex,  3*vIndex,1));
        triplets.push_back(Eigen::Triplet<double>(rowIndex+1,  3*vIndex+1, 1));
        triplets.push_back(Eigen::Triplet<double>(rowIndex+2,  3*vIndex+2, 1));
        
        rowIndex+=3;
    }
    
    P.setFromTriplets(triplets.begin(), triplets.end());
    
    //build the matrix and  return
    return P;
}


void reset_world (MyWorld &world) {
        auto q = mapStateEigen(world); // TODO is this necessary?
        q.setZero();
}

void exit_gracefully() {
    log_ofstream.seekp(0);
    log_ofstream << std::setw(2) << sim_log;
    log_ofstream.close();
    exit(0);
}

// -- My integrator



template <typename ReducedSpaceType, typename MatrixType>
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
        m_save_ply_every_evaluation = get_json_value(integrator_config, "save_ply_every_evaluation", false);
        m_cur_sub_q = sub_dec(cur_z);
        m_prev_sub_q = sub_dec(prev_z);

        m_h = integrator_config["timestep"];
        // Construct mass matrix and external forces
        getMassMatrix(m_M_asm, *m_world);

        std::cout << "Constructing reduced mass matrix..." << std::endl;
        m_M = *m_M_asm;

        m_U = m_reduced_space->outer_jacobian();
        m_UTMU = m_U.transpose() * m_M * m_U;

        // Check if m_UTMU is identity (mass pca)
        if(integrator_config["reduced_space_type"] != "full") { // Only check for non-full (dense) spaces
            if(MatrixXd(m_UTMU).isIdentity(1e-6)) {
                m_UTMU_is_identity = true;
                std::cout << "UTMU is Identity!!! Removing from calculations." << std::endl;
            } else {
                std::cout << "UTMU is not Identity!!!" << std::endl;
            }
        }

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
        m_use_partial_decode =  (m_energy_method == PCR || m_energy_method == AN08 || m_energy_method == NEW_PCR) && get_json_value(integrator_config, "use_partial_decode", true);

        if(m_energy_method == PCR){
            fs::path pca_components_path("pca_results/energy_pca_components.dmat");
            fs::path sample_tets_path("pca_results/energy_indices.dmat");

            MatrixXd U;
            igl::readDMAT((model_root / pca_components_path).string(), U);

            Eigen::VectorXi Is;
            igl::readDMAT((model_root /sample_tets_path).string(), Is); //TODO do I need to sort this?
            m_cubature_indices = Is;

            // Other stuff?
            m_energy_basis = U;
            m_energy_sampled_basis = igl::slice(m_energy_basis, m_cubature_indices, 1);
            m_summed_energy_basis = m_energy_basis.colwise().sum();
            MatrixXd S_barT_S_bar = m_energy_sampled_basis.transpose() * m_energy_sampled_basis;
            // m_energy_sampled_basis_qr = m_energy_sampled_basis.fullPivHouseholderQr();
            
            m_cubature_weights = m_energy_sampled_basis * S_barT_S_bar.ldlt().solve(m_summed_energy_basis); //U_bar(A^-1*u)

            m_neg_energy_sample_jac = SparseMatrix<double>(n_dof, m_cubature_indices.size());
            m_neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, T.cols() * 3)); // Reserve enough room for 4 verts (tet corners) per column
            std::cout << "Done. Will sample from " << m_cubature_indices.size() << " tets." << std::endl;
        }
        else if(m_energy_method == AN08 || m_energy_method == NEW_PCR) {
            fs::path energy_model_dir;
            
            if(m_energy_method == AN08) energy_model_dir = model_root / "energy_model/an08/";
            if(m_energy_method == NEW_PCR) energy_model_dir = model_root / "energy_model/new_pcr/";

            fs::path indices_path = energy_model_dir / "indices.dmat";
            fs::path weights_path = energy_model_dir / "weights.dmat";

            Eigen::VectorXi Is;
            Eigen::VectorXd Ws;
            igl::readDMAT(indices_path.string(), Is); //TODO do I need to sort this?
            igl::readDMAT(weights_path.string(), Ws); 

            m_cubature_weights = Ws;
            m_cubature_indices = Is;

            m_neg_energy_sample_jac = SparseMatrix<double>(n_dof, m_cubature_indices.size());
            m_neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, T.cols() * 3)); // Reserve enough room for 4 verts (tet corners) per column
            std::cout << "Done. Will sample from " << m_cubature_indices.size() << " tets." << std::endl;
        } else if(m_energy_method == AN08_ALL) {
            fs::path energy_model_dir = model_root / "energy_model/an08/";
            load_all_an08_indices_and_weights(energy_model_dir, m_all_cubature_indices, m_all_cubature_weights);

            m_cubature_weights = m_all_cubature_weights.back();
            m_cubature_indices = m_all_cubature_indices.back();

            m_neg_energy_sample_jac = SparseMatrix<double>(n_dof, m_cubature_indices.size());
            m_neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, T.cols() * 3)); // Reserve enough room for 4 verts (tet corners) per column
        }

        // Computing the sampled verts
        if (m_energy_method == AN08 || m_energy_method == PCR || m_energy_method == NEW_PCR) {
            // figure out the verts
            std::set<int> vert_set;
            int n_sample_tets = m_cubature_indices.size();
            for(int i = 0; i < m_cubature_indices.size(); i++) {
                int tet_index = m_cubature_indices[i];
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
        int n_sample_tets = m_cubature_indices.size();
        
        VectorXd sampled_energy(n_sample_tets);

        int n_force_per_element = T.cols() * 3;
        MatrixXd element_forces(n_sample_tets, n_force_per_element);
        

        #pragma omp parallel for num_threads(4) //schedule(static,64)// TODO how to optimize this
        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            int tet_index = m_cubature_indices[i];

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
            int tet_index = m_cubature_indices[i];
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


    double operator()(const VectorXd& new_z, VectorXd& grad, const int cur_iteration, const int cur_line_search_iteration)
    {
        // Update the tets with candidate configuration
        double obj_start_time = igl::get_seconds();
        VectorXd new_sub_q = sub_dec(new_z);
        double tf_time = igl::get_seconds() - obj_start_time;

        double decode_start_time = igl::get_seconds();
        Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(m_tets->getQ(), *m_world);
        
        if(m_use_partial_decode) {
            // We only need to update verts for the tets we actually sample
            VectorXd sampled_q = m_U_sampled * new_sub_q;
            for(int i = 0; i < sampled_q.size(); i++) {
                q[m_cubature_vert_indices[i]] = sampled_q[i]; // TODO this could probably be more efficient with a sparse operator?
            } 
        } else {
            q = m_U * new_sub_q;
        } 

        // TWISTING
        // auto max_verts = get_min_verts(0, true);
        // Eigen::AngleAxis<double> rot(0.05 * current_frame, Eigen::Vector3d(1.0,0.0,0.0));
        // for(int i = 0; i < max_verts.size(); i++) {
        //     int vi = max_verts[i];
        //     Eigen::Vector3d v = V.row(vi);
        //     Eigen::Vector3d new_q = rot * v - v;// + Eigen::Vector3d(1.0,0.0,0.0) * current_frame / 300.0;
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
        if(m_energy_method == PCR || m_energy_method == AN08 || m_energy_method == NEW_PCR || m_energy_method == AN08_ALL) {
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
        // Full
        // double obj_val = 0.5 * sub_q_tilde.transpose() * UTMU_sub_q_tilde
        //                     + m_h * m_h * energy
        //                     - sub_q_tilde.transpose() * h_2_UT_external_forces;

        // grad = jtvp(new_z, UTMU_sub_q_tilde - m_h * m_h * UT_internal_forces - h_2_UT_external_forces);

        // No momentum
        double obj_val =   + m_h * m_h * energy
                            - sub_q_tilde.transpose() * h_2_UT_external_forces;
        grad = jtvp(new_z, - m_h * m_h * UT_internal_forces - h_2_UT_external_forces);

        // Compute gradient
        // **** TODO
        // Can I further reduce the force calculations by carrying through U?

        // if(m_energy_method == PRED_DIRECT) {
        //     grad = jtvp(new_z,UTMU_sub_q_tilde - h_2_UT_external_forces) - m_h * m_h * m_reduced_space->get_energy_gradient(new_z);
        // } else {
        //     grad = jtvp(new_z,UTMU_sub_q_tilde - m_h * m_h * UT_internal_forces - h_2_UT_external_forces);
        // }

        double obj_and_grad_time = igl::get_seconds() - obj_and_grad_time_start;
        double obj_time = igl::get_seconds() - obj_start_time;

        if(LOGGING_ENABLED) {
            timestep_info["iteration_info"]["lbfgs_iteration"].push_back(cur_iteration);
            timestep_info["iteration_info"]["lbfgs_line_iteration"].push_back(cur_line_search_iteration);
            timestep_info["iteration_info"]["lbfgs_obj_vals"].push_back(obj_val);
            timestep_info["iteration_info"]["timing"]["tot_obj_time_s"].push_back(obj_time);
            timestep_info["iteration_info"]["timing"]["tf_time_s"].push_back(tf_time);
            timestep_info["iteration_info"]["timing"]["linear_decode_time_s"].push_back(decode_time);
            timestep_info["iteration_info"]["timing"]["energy_forces_time_s"].push_back(energy_forces_time);
            timestep_info["iteration_info"]["timing"]["predict_weight_time"].push_back(predict_weight_time);
            timestep_info["iteration_info"]["timing"]["obj_and_grad_eval_time_s"].push_back(obj_and_grad_time);

            if(m_save_ply_every_evaluation) {
                VectorXd q = m_reduced_space->decode(new_z);
                Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only

                // PLY
                // fs::path ply_filename = evaluation_ply_dir / (ZeroPadNumber(cur_iteration) + "_" + ZeroPadNumber(cur_line_search_iteration, 2) + ".ply");
                // igl::writePLY(ply_filename.string(), V + dV.transpose(), F, false);

                fs::path ply_filename = evaluation_ply_dir / (ZeroPadNumber(cur_iteration) + "_" + ZeroPadNumber(cur_line_search_iteration, 2) + ".ply");
                MatrixXd new_verts = V + dV.transpose();
                igl::writeOBJ(ply_filename.string(), new_verts, F);
            }
        }

        return obj_val;
    }

    VectorXi get_current_tets() {
        return m_cubature_indices;
    }

    void switch_to_next_tets(int i) { // This is an ugly dirty hack to get extra tets working for an08
        m_cubature_weights = m_all_cubature_weights[m_all_cubature_weights.size() - i - 1];
        m_cubature_indices = m_all_cubature_indices[m_all_cubature_weights.size() - i - 1];

        m_neg_energy_sample_jac = SparseMatrix<double>(n_dof, m_cubature_indices.size());
        m_neg_energy_sample_jac.reserve(VectorXi::Constant(n_dof, T.cols() * 3)); // Reserve enough room for 4 verts (tet corners) per column

        std::cout << "Now sampling from " << m_cubature_indices.size() << " tets" << std::endl;
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
    MatrixType  m_UTMU; // reduced mass matrix
    MatrixType  m_U;
    MatrixXd m_U_sampled; // Always used in dense mode

    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > m_M_asm;
    AssemblerParallel<double, AssemblerEigenVector<double> > m_internal_force_asm;

    double m_h;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    EnergyMethod m_energy_method;
    ReducedSpaceType *m_reduced_space;
    bool m_use_partial_decode = false;
    bool m_save_ply_every_evaluation = false;
    bool m_UTMU_is_identity = false;

    MatrixXd m_energy_basis;
    VectorXd m_summed_energy_basis;
    MatrixXd m_energy_sampled_basis;
    Eigen::FullPivHouseholderQR<MatrixXd> m_energy_sampled_basis_qr;
    SparseMatrix<double> m_neg_energy_sample_jac;

    VectorXd m_cubature_weights;
    VectorXi m_cubature_indices;
    VectorXi m_cubature_vert_indices;

    std::vector<VectorXd> m_all_cubature_weights;
    std::vector<VectorXi> m_all_cubature_indices;
};

template <typename ReducedSpaceType, typename MatrixType>
class GPLCTimeStepper {
public:
    GPLCTimeStepper(fs::path model_root, json integrator_config, MyWorld *world, NeohookeanTets *tets, ReducedSpaceType *reduced_space) :
        m_world(world), m_tets(tets), m_reduced_space(reduced_space) {

        // Get parameters
        m_energy_method = energy_method_from_integrator_config(integrator_config);
        m_use_preconditioner = integrator_config["use_preconditioner"];
        m_is_full_space = integrator_config["reduced_space_type"] == "full";
        m_h = integrator_config["timestep"];
        m_U = reduced_space->outer_jacobian();


        // Set up lbfgs params
        json lbfgs_config = integrator_config["lbfgs_config"];
        m_lbfgs_param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
        m_lbfgs_param.m = lbfgs_config["lbfgs_m"];
        m_lbfgs_param.epsilon = lbfgs_config["lbfgs_epsilon"]; //1e-8// TODO replace convergence test with abs difference
        m_lbfgs_param.max_iterations = lbfgs_config["lbfgs_max_iterations"];//500;//300;
       // param.epsilon = 1e-4;
       // param.max_iterations = 1000;
       // param.past = 1;
       // param.m = 5;
       // param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
        m_solver = new LBFGSSolver<double>(m_lbfgs_param);


        // Load start pose if needed
        int start_pose_from_training_data = get_json_value(integrator_config, "start_pose_from_training_data", -1);
        if(start_pose_from_training_data != -1) {
            MatrixXd starting_data;
            igl::readDMAT((model_root / ("training_data/training/displacements_" + std::to_string(start_pose_from_training_data) + ".dmat")).string(), starting_data);
            starting_data.transposeInPlace();
            m_starting_pose = Eigen::Map<VectorXd>(starting_data.data(), starting_data.size());
        } else {
            m_starting_pose = VectorXd::Zero(V.size());
        }
        reset_zs();


        // Set up preconditioner
        if(m_use_preconditioner) {
            MatrixType J = reduced_space->inner_jacobian(m_cur_z);
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

        // Finally initialize the BFGS objective
        m_gplc_objective = new GPLCObjective<ReducedSpaceType, MatrixType>(model_root, integrator_config, m_cur_z, m_prev_z, world, tets, reduced_space);
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
        
        // int activated_tets = T.rows();
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
                MatrixType J = m_reduced_space->inner_jacobian(m_cur_z);
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


        

        m_prev_z = m_cur_z;
        m_cur_z = z_param; // TODO: Use pointers to avoid copies
        m_gplc_objective->update_zs(m_cur_z);

        // update_world_with_current_configuration(); This happens in each opt loop

        m_total_time += update_time;
        m_total_time_since_last += update_time;
        m_tot_its_since_last += niter;

        if(current_frame % print_freq == 0) {
            std::cout << std::endl;
            std::cout << "Averages for last " << print_freq << " frames:" << std::endl;

            std::cout << "L-BFGS Iterations: " << m_tot_its_since_last / (double)print_freq << std::endl;
            std::cout << "Step time(s): " << m_total_time_since_last / (double)print_freq << "s" << std::endl;
            std::cout << "Step time(Hz): " << print_freq / (double)m_total_time_since_last << "s" << std::endl;
            std::cout << "Total step time avg: " << m_total_time / (double)current_frame << "s" << std::endl;
            // std::cout << "objective val = " << min_val_res << std::endl;
            // std::cout << "Current z: " << m_cur_z.transpose() << std::endl;
            // std::cout << "Activated tets: " << activated_tets << std::endl;

            m_total_time_since_last = 0.0;
            m_tot_its_since_last = 0;
        }

        if(LOGGING_ENABLED) {
            if(current_frame % print_freq == 0) {
                std::cout << "LOGGING_ENABLED" << std::endl; // just a warning
            }
            // m_total_tets += activated_tets;

            timestep_info["current_frame"] = current_frame; // since at end of timestep
            timestep_info["tot_step_time_s"] = update_time;
            timestep_info["precondition_time"] = precondition_compute_time;
            timestep_info["lbfgs_iterations"] = niter;
            // timestep_info["avg_activated_tets"] = m_total_tets / (double) current_frame;

            timestep_info["mouse_info"]["dragged_pos"] = {dragged_pos[0], dragged_pos[1], dragged_pos[2]};
            timestep_info["mouse_info"]["dragged_vert"] = dragged_vert;
            timestep_info["mouse_info"]["is_dragging"] = is_dragging;

            sim_log["timesteps"].push_back(timestep_info);
            if(current_frame % log_save_freq == 0) {
                log_ofstream.seekp(0);
                log_ofstream << std::setw(2) << sim_log; // TODO 
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

    void reset_zs() {
        m_prev_z = m_reduced_space->encode(m_starting_pose);
        m_cur_z = m_prev_z;
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

    MatrixXd get_current_V() {
        VectorXd q = get_current_q();
        Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
        return V + dV.transpose();    
    }

    VectorXi get_current_tets() {
        return m_gplc_objective->get_current_tets();
    }

    void switch_to_next_tets(int i) {
        m_gplc_objective->switch_to_next_tets(i);
    }

    VectorXd get_current_vert_pos(int vert_index) {
        VectorXd sub_q = m_reduced_space->sub_decode(m_cur_z);

        return V.row(vert_index).transpose() + m_U.block(vert_index * 3, 0, 3, m_U.cols()) * sub_q;
    }

private:
    bool m_use_preconditioner;
    bool m_is_full_space;

    LBFGSParam<double> m_lbfgs_param;
    LBFGSSolver<double> *m_solver;
    GPLCObjective<ReducedSpaceType, MatrixType> *m_gplc_objective;

    VectorXd m_starting_pose;

    VectorXd m_prev_z;
    VectorXd m_cur_z;

    double m_h;

    MatrixType m_H;
    typename std::conditional<
        std::is_same<MatrixType, MatrixXd>::value,
        Eigen::LDLT<MatrixXd>,
        Eigen::SimplicialLDLT<SparseMatrix<double>>>::type m_H_llt;
    MatrixType m_H_inv;
    MatrixType m_UTKU;
    MatrixType m_UTMU;
    MatrixType m_U;

    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > m_M_asm;
    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > m_K_asm;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    ReducedSpaceType *m_reduced_space;
    EnergyMethod m_energy_method;

    double m_total_time = 0.0;
    double m_total_time_since_last = 0.0;
    int m_tot_its_since_last = 0;
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

// VectorXd get_von_mises_stresses(NeohookeanTets *tets, MyWorld &world) {
//     // Compute cauchy stress
//     int n_ele = tets->getImpl().getF().rows();

//     Eigen::Matrix<double, 3,3> stress = MatrixXd::Zero(3,3);
//     VectorXd vm_stresses(n_ele);

//     std::cout << "Disabled cauchy stress for linear tets.." << std::endl;
//     // for(unsigned int i=0; i < n_ele; ++i) {
//     //     tets->getImpl().getElement(i)->getCauchyStress(stress, Vec3d(0,0,0), world.getState());

//     //     double s11 = stress(0, 0);
//     //     double s22 = stress(1, 1);
//     //     double s33 = stress(2, 2);
//     //     double s23 = stress(1, 2);
//     //     double s31 = stress(2, 0);
//     //     double s12 = stress(0, 1);
//     //     vm_stresses[i] = sqrt(0.5 * (s11-s22)*(s11-s22) + (s33-s11)*(s33-s11) + 6.0 * (s23*s23 + s31*s31 + s12*s12));
//     // }

//     return vm_stresses;
// }

std::string int_to_padded_str(int i) {
    std::stringstream frame_str;
    frame_str << std::setfill('0') << std::setw(5) << i;
    return frame_str.str();
}


// typedef ReducedSpace<IdentitySpaceImpl> IdentitySpace;
// typedef ReducedSpace<ConstraintSpaceImpl> ConstraintSpace;


template <typename ReducedSpaceType, typename MatrixType>
void run_sim(ReducedSpaceType *reduced_space, const json &config, const fs::path &model_root) {
    // -- Setting up GAUSS
    MyWorld world;

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
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    reset_world(world);
    // --- Finished GAUSS set up
    

    // --- My integrator set up
    double spring_stiffness = visualization_config["interaction_spring_stiffness"];
    SparseVector<double> interaction_force(n_dof);// = VectorXd::Zero(n_dof);


    GPLCTimeStepper<ReducedSpaceType, MatrixType> gplc_stepper(model_root, integrator_config, &world, tets, reduced_space);

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

    bool do_gpu_decode = integrator_config["reduced_space_type"] != "full" && get_json_value(visualization_config, "gpu_decode", false);
    int max_frames = get_json_value(visualization_config, "max_frames", 0);

    /** libigl display stuff **/
    igl::viewer::Viewer viewer;    

    viewer.data.set_mesh(V,F);
    igl::per_corner_normals(V,F,40,N);
    viewer.data.set_normals(N);

    viewer.core.show_lines = false;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = false;
    viewer.core.line_width = 2;
    viewer.core.animation_max_fps = 1000.0;
    viewer.core.background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
    viewer.core.shininess = 120.0;
    viewer.data.set_colors(Eigen::RowVector3d(igl::CYAN_DIFFUSE[0], igl::CYAN_DIFFUSE[1], igl::CYAN_DIFFUSE[2]));

    viewer.launch_init(true, false);    
    viewer.opengl.shader_mesh.free();

    std::string mesh_vertex_shader_string;
    std::string mesh_fragment_shader_string;

    Eigen::Matrix< float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> U;
    Eigen::Matrix< float,Eigen::Dynamic,1> I = igl::LinSpaced< Eigen::Matrix< float,Eigen::Dynamic,1> >(V.rows(),0,V.rows()-1);
    Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor> tex;
    int n = 0;
    int m = 0;
    int s = 0;
    if(do_gpu_decode) {
        MatrixXd Ud = reduced_space->outer_jacobian();
        U = Ud.cast <float> ();
        // std::cout<< U << std::endl;
        n = V.rows();
        m = U.cols();
        s = ceil(sqrt(n*m));
        assert((U.rows() == V.rows()*3) && "#U should be 3*#V");
        // I = igl::LinSpaced< Eigen::Matrix< float,Eigen::Dynamic,1> >(V.rows(),0,V.rows()-1);
        assert(s*s > n*m);
        printf("%d %d %d\n",n,m,s);
        tex = Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor>::Zero(s*s,3);
        for(int j = 0;j<m;j++) {
            for(int i = 0;i<n;i++) {
                for(int c = 0;c<3;c++) {
                    tex(i*m+j,c) = U(i*3 + c, j);///U(i+c*n,j);
                }
            }
        }
    }

    mesh_vertex_shader_string =
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

in float id;
uniform int n;
uniform int m;
uniform int s;
uniform bool do_gpu_decode;
uniform float q[512];
uniform sampler2D tex;

void main()
{
  vec3 displacement = vec3(0,0,0);
  if(do_gpu_decode) {
      for(int j = 0;j < m; j++)
      {
        int index = int(id)*m+j;
        int si = index % s;
        int sj = int((index - si)/s);
        displacement = displacement + texelFetch(tex,ivec2(si,sj),0).xyz*q[j];
      }
  }
  vec3 deformed = position + displacement;

  position_eye = vec3 (view * model * vec4 (deformed, 1.0));
  gl_Position = proj * vec4 (position_eye, 1.0);
  Kai = Ka;
  Kdi = Kd;
  Ksi = Ks;
})";

    mesh_fragment_shader_string =
R"(#version 150
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 fixed_color;
in vec3 position_eye;
uniform vec3 light_position_world;
vec3 Ls = vec3 (1, 1, 1);
vec3 Ld = vec3 (1, 1, 1);
vec3 La = vec3 (1, 1, 1);
in vec4 Ksi;
in vec4 Kdi;
in vec4 Kai;
uniform float specular_exponent;
uniform float lighting_factor;
out vec4 outColor;
void main()
{
  vec3 xTangent = dFdx(position_eye);
  vec3 yTangent = dFdy(position_eye);
  vec3 normal_eye = normalize( cross( xTangent, yTangent ) );

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
outColor = color;
if (fixed_color != vec4(0.0)) outColor = fixed_color;
})";
    
    viewer.opengl.shader_mesh.init( // This needs to come after shader and before buffer setup
          mesh_vertex_shader_string,
          mesh_fragment_shader_string, 
          "outColor");
    
    ///////////////////////////////////////////////////////////////////
    // Send texture and vertex attributes to GPU
    ///////////////////////////////////////////////////////////////////
    {
        GLuint prog_id = viewer.opengl.shader_mesh.program_shader;
        glUseProgram(prog_id);
        GLuint VAO = viewer.opengl.vao_mesh;
        glBindVertexArray(VAO);
        GLuint IBO;
        glGenBuffers(1, &IBO);
        glBindBuffer(GL_ARRAY_BUFFER, IBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*I.size(), I.data(), GL_STATIC_DRAW);
        GLint iid = glGetAttribLocation(prog_id, "id");
        glVertexAttribPointer(
          iid, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(GLfloat), (GLvoid*)0);
        glEnableVertexAttribArray(iid);
        glBindVertexArray(0);
        glActiveTexture(GL_TEXTURE0);
        //glGenTextures(1, &v.opengl.vbo_tex);
        glBindTexture(GL_TEXTURE_2D, viewer.opengl.vbo_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        // 8650×8650 texture was roughly the max I could still get 60 fps, 8700²
        // already dropped to 1fps
        //
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, s,s, 0, GL_RGB, GL_FLOAT, tex.data());
    }


    double cur_fps = 0.0;
    double last_time = igl::get_seconds();
    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
        if(do_gpu_decode) {
            /////////////////////////////////////////////////////////
            // Send uniforms to shader
            /////////////////////////////////////////////////////////
            GLuint prog_id = viewer.opengl.shader_mesh.program_shader;
            glUseProgram(prog_id);
            GLint n_loc = glGetUniformLocation(prog_id,"n");
            glUniform1i(n_loc,n);
            GLint m_loc = glGetUniformLocation(prog_id,"m");
            glUniform1i(m_loc,m);
            GLint s_loc = glGetUniformLocation(prog_id,"s");
            glUniform1i(s_loc,s);
            GLint do_gpu_decode_loc = glGetUniformLocation(prog_id,"do_gpu_decode");
            glUniform1i(do_gpu_decode_loc, do_gpu_decode);
        
            VectorXd sub_qd = reduced_space->sub_decode(gplc_stepper.get_current_z());
            VectorXf sub_q = sub_qd.cast<float>();
            GLint q_loc = glGetUniformLocation(prog_id,"q");
            glUniform1fv(q_loc,U.cols(),sub_q.data());
            
            // Do this now so that we can stop texture from being loaded by viewer
            if (viewer.data.dirty)
            {
              viewer.opengl.set_data(viewer.data, viewer.core.invert_normals);
              viewer.data.dirty = igl::viewer::ViewerData::DIRTY_NONE;
            }
            viewer.opengl.dirty &= ~igl::viewer::ViewerData::DIRTY_TEXTURE;
        }

        

        mesh_pos = gplc_stepper.get_current_vert_pos(dragged_vert);
        if(is_dragging) {
            // Eigen::MatrixXd part_pos(1, 3);
            // part_pos(0,0) = dragged_pos[0]; // TODO why is eigen so confusing.. I just want to make a matrix from vec
            // part_pos(0,1) = dragged_pos[1];
            // part_pos(0,2) = dragged_pos[2];

            // viewer.data.set_points(part_pos, orange);

            MatrixXi E(1,2);
            E(0,0) = 0;
            E(0,1) = 1;
            MatrixXd P(2,3);using Eigen::VectorXd;
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
            if(!do_gpu_decode) {
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

                if(per_vertex_normals) {
                    igl::per_vertex_normals(V,F,N);
                } else {
                    igl::per_corner_normals(V,F,40,N);
                }
                viewer.data.set_normals(N);
            }

            // Play back mouse interaction from previous sim
            if(PLAYBACK_SIM) {
                // Eigen::MatrixXd newV = gplc_stepper.get_current_V(); // TODO we can avoid doing this
                json current_mouse_info = sim_playback_json["timesteps"][current_frame]["mouse_info"];
                dragged_pos = Eigen::RowVector3d(current_mouse_info["dragged_pos"][0], current_mouse_info["dragged_pos"][1], current_mouse_info["dragged_pos"][2]);

                bool was_dragging = is_dragging;
                is_dragging = current_mouse_info["is_dragging"];

                if(is_dragging && !was_dragging) { // Got a click
                    dragged_vert = current_mouse_info["dragged_vert"];
                    // // Get closest point on mesh
                    // MatrixXd C(1,3);
                    // C.row(0) = dragged_pos;
                    // MatrixXi I;
                    // igl::snap_points(C, newV, I);

                    // dragged_vert = I(0,0);
                }
            }
            
            // Do the physics update
            interaction_force = compute_interaction_force(dragged_pos, dragged_vert, is_dragging, spring_stiffness, tets, world);
            gplc_stepper.step(interaction_force);

            if(LOGGING_ENABLED) {
                // Only do this work if we are saving objs
                if(get_json_value(config, "save_objs", false)) {
                    Eigen::MatrixXd newV = gplc_stepper.get_current_V(); // TODO we can avoid doing this
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

                            T_sampled = igl::slice(T, gplc_stepper.get_current_tets(), 1);
                            igl::boundary_facets(T_sampled, F_sampled);
                            igl::writeOBJ((tet_obj_dir /tet_obj_filename).string(), newV, F_sampled);                    
                        }
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

            if(current_frame % print_freq == 0) {   
                double cur_time = igl::get_seconds();
                cur_fps = print_freq / (cur_time - last_time);
                last_time = cur_time;
                std::cout << "Step time + rendering(Hz): "<< cur_fps << std::endl;
            }

            if(max_frames != 0 && current_frame + 1 >= max_frames) {
                exit_gracefully();
            }
            current_frame++;
        }

        // if(show_stress) {
        //     // Do stress field viz
        //     VectorXd vm_stresses = get_von_mises_stresses(tets, world);
        //     // vm_stresses is per element... Need to compute avg value per vertex
        //     VectorXd vm_per_vert = VectorXd::Zero(V.rows());
        //     VectorXi neighbors_per_vert = VectorXi::Zero(V.rows());
            
        //     int t = 0;
        //     for(int i=0; i < T.rows(); i++) {
        //         for(int j=0; j < 4; j++) {
        //             t++;
        //             int vert_index = T(i,j);
        //             vm_per_vert[vert_index] += vm_stresses[i];
        //             neighbors_per_vert[vert_index]++;
        //         }
        //     }
        //     for(int i=0; i < vm_per_vert.size(); i++) {
        //         vm_per_vert[i] /= neighbors_per_vert[i];
        //     }
        //     VectorXd vm_per_face = VectorXd::Zero(F.rows());
        //     for(int i=0; i < vm_per_face.size(); i++) {
        //         vm_per_face[i] = (vm_per_vert[F(i,0)] + vm_per_vert[F(i,1)] + vm_per_vert[F(i,2)])/3.0;
        //     }
        //     // std::cout << vm_per_face.maxCoeff() << " " <<  vm_per_face.minCoeff() << std::endl;
        //     MatrixXd C;
        //     //VectorXd((vm_per_face.array() -  vm_per_face.minCoeff()) / vm_per_face.maxCoeff())
        //     igl::jet(vm_per_vert / 60.0, false, C);
        //     viewer.data.set_colors(C);
        // }

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

    int index_counter = 0;
    viewer.callback_key_pressed = [&](igl::viewer::Viewer &, unsigned int key, int mod)
    {
        switch(key)
        {
            case 'q':
            {
                exit_gracefully();
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
                gplc_stepper.reset_zs();
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
            case '-':
            {
                gplc_stepper.switch_to_next_tets(++index_counter);
                break;
            }
            case '=':
            case '+':
            {
                gplc_stepper.switch_to_next_tets(++index_counter);
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
            // Eigen::MatrixXd curV = gplc_stepper.get_current_V();
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
        }
        return false;
    };

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



int main(int argc, char **argv) {
    
    // Load the configuration file
    if(argc < 2) {
        std::cout << "Expected model root." << std::endl;
    }

    fs::path model_root(argv[1]);
    fs::path sim_config("sim_config.json");

    if(argc >= 3) {
        fs::path sim_recording_path(argv[2]);
        std::ifstream recording_fin(sim_recording_path.string());

        recording_fin >> sim_playback_json;
        PLAYBACK_SIM = true;
    }

    std::string log_string = "";
    if(argc == 4) {
        log_string = argv[3];
    } else {
        log_string = currentDateTime();
    }


    std::ifstream fin((model_root / sim_config).string());
    json config;
    fin >> config;

    auto integrator_config = config["integrator_config"];
    std::string reduced_space_string = integrator_config["reduced_space_type"];

    print_freq = get_json_value(config["visualization_config"], "print_every_n_frames", print_freq);

    LOGGING_ENABLED = config["logging_enabled"];
    if(LOGGING_ENABLED) {
        sim_log["sim_config"] = config;
        sim_log["timesteps"] = {};
        fs::path sim_log_path("simulation_logs/");
        log_dir = model_root / fs::path("./simulation_logs/" + log_string + "/");
        surface_obj_dir = log_dir / fs::path("objs/surface/");
        pointer_obj_dir = log_dir / fs::path("objs/pointer/");
        tet_obj_dir = log_dir / fs::path("objs/sampled_tets/");
        evaluation_ply_dir = log_dir / fs::path("plys/evaluations/");
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

        if(!boost::filesystem::exists(evaluation_ply_dir) && integrator_config["save_ply_every_evaluation"]){
            boost::filesystem::create_directories(evaluation_ply_dir);
        }

        // boost::filesystem::create_directories(log_dir);
        fs::path log_file("sim_stats.json");

        log_ofstream = std::ofstream((log_dir / log_file).string());
    }

    // Load the mesh here
    std::string alternative_mesh_path = get_json_value(config, "alternative_full_space_mesh", std::string(""));
    if(alternative_mesh_path != "") {
        igl::readMESH((model_root / alternative_mesh_path).string(), V, T, F);
    } else {
        igl::readMESH((model_root / "tets.mesh").string(), V, T, F);
    }

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

        run_sim<LinearSpace, MatrixXd>(&reduced_space, config, model_root);
    }
    else if(reduced_space_string == "autoencoder") {
        fs::path tf_models_root(model_root / "tf_models/");
        AutoencoderSpace reduced_space(tf_models_root, integrator_config);
        // compare_jac_speeds(reduced_space);
        run_sim<AutoencoderSpace, MatrixXd>(&reduced_space, config, model_root);  
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

        run_sim<SparseConstraintSpace, SparseMatrix<double>>(&reduced_space, config, model_root);
    }
    else {
        std::cout << "Not yet implemented." << std::endl;
        return 1;
    }
    
    return EXIT_SUCCESS;

}

