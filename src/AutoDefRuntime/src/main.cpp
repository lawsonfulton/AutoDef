#define EIGEN_USE_MKL_ALL
// #define MKL_DIRECT_CALL
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif

#include <vector>
#include <set>
#include <thread>
#include <future>

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

typedef PhysicalSystemFEM<double, NeohookeanHFixedTet> NeohookeanTets;

typedef World<double, 
                        std::tuple<PhysicalSystemParticleSingle<double> *, NeohookeanTets *>,
                        std::tuple<ForceSpringFEMParticle<double> *>,
                        std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
typedef TimeStepperEulerImplicitLinear<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>,
AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepper;

//#include <igl/png/render_to_png_async.h>
//#include <igl/png/render_to_png.h>
#include <igl/png/writePNG.h>

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
#include <igl/slice_into.h>
#include <igl/unique.h>
#include <igl/per_corner_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/circumradius.h>
#include <igl/material_colors.h>
#include <igl/snap_points.h>
#include <igl/centroid.h>
#include <igl/LinSpaced.h>
#include <igl/vertex_triangle_adjacency.h>

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
#include <nlohmann/json.hpp>


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

std::vector<int> fixed_verts;

// Mouse/Viewer state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
Eigen::RowVector3d dragged_mesh_pos;
int vis_vert_id = 0;
double spring_grab_radius = 0.03; 
bool is_dragging = false;
bool per_vertex_normals = false;
std::vector<int> dragged_verts;
int current_frame = 0;

// Parameters
int print_freq = 40;
bool NO_WAIT = false;
bool LOGGING_ENABLED = false;
bool PLAYBACK_SIM = false;
bool SAVE_TRAINING_DATA = false;
bool SAVE_PNGS = false;
int log_save_freq = 5;
json sim_log;
json sim_playback_json;
json timestep_info; // Kind of hack but this gets reset at the beginning of each timestep so that we have a global structure to log to
fs::path log_dir;
fs::path surface_obj_dir;
fs::path iteration_obj_dir;
fs::path pointer_obj_dir;
fs::path tet_obj_dir;
fs::path save_training_data_dir;
fs::path save_pngs_dir;
std::ofstream log_ofstream;

// Constants
const int PARDISO_DOF_CUTOFF = 7000;
const Eigen::RowVector3d sea_green(229./255.,211./255.,91./255.);

#define DO_TIMING
int total_evals = 0;
double tf_decode_time_tot = 0.0;
double tf_vjp_time_tot = 0.0;
double cuba_decode_time_tot = 0.0;
double cuba_eval_time_tot = 0.0;
double obj_grad_time_tot_no_jvp = 0.0;
double precon_time_tot = 0.0;

std::vector<int> get_min_verts(int axis, bool flip_axis = false, double tol = 0.001) {
    int dim = axis; // x
    double min_x_val = flip_axis ?  V.col(dim).maxCoeff() : V.col(dim).minCoeff();
    std::vector<int> min_verts;

    for(int ii=0; ii<V.rows(); ++ii) {
        if(fabs(V(ii, dim) - min_x_val) < tol) {
            min_verts.push_back(ii);
        }
    }

    return min_verts;
}

std::vector<int> get_verts_in_sphere(const VectorXd &c, double r, const MatrixXd &verts) {
    std::vector<int> min_verts;

    for(int ii=0; ii<verts.rows(); ++ii) {
        if((verts.row(ii) - c.transpose()).squaredNorm() <= r * r) {
            min_verts.push_back(ii);
        }
    }

    return min_verts;
}

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

void create_or_replace_dir(fs::path dir) {
    if(fs::exists(dir)){
        std::cout << "Are you sure you want to delete " << dir << "? Y/n: ";
        std::string input;
        input = std::cin.get();
        if(input == "Y" || input == "y" || input == "\n") {
            fs::remove_all(dir);
        } else {
            exit(1);
        }
    }
    fs::create_directories(dir);
}

void reset_world (MyWorld &world) {
        auto q = mapStateEigen(world); // TODO is this necessary?
        q.setZero();
}

void exit_gracefully() {
    log_ofstream.seekp(0);
    log_ofstream << std::setw(2) << sim_log;
    log_ofstream.close();

    if(SAVE_TRAINING_DATA) {
        fs::copy_file(log_dir / "sim_stats.json", save_training_data_dir / "sim_stats.json");
    }

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
        m_save_obj_every_iteration = get_json_value(integrator_config, "save_obj_every_iteration", false);
        m_cur_sub_q = sub_dec(cur_z);
        m_prev_sub_q = sub_dec(prev_z);

        m_h = integrator_config["timestep"];
        // Construct mass matrix and external forces
        getMassMatrix(m_M_asm, *m_world);

        m_M = *m_M_asm;

        m_U = m_reduced_space->outer_jacobian();
        std::cout << "Constructing reduced mass matrix..." << std::endl;
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
        int gravity_axis = get_json_value(integrator_config, "gravity_axis", 1);
        for(int i=0; i < g.size(); i += 3) {
            g[i] = 0.0;
            g[i+1] = 0.0;
            g[i+2] = 0.0;
            g[i + gravity_axis] = integrator_config["gravity"];
        }

        m_F_ext = m_M * g;

        m_UT_F_ext = m_U.transpose() * m_F_ext;
        m_interaction_force = SparseVector<double>(m_F_ext.size());

        m_energy_method = energy_method_from_integrator_config(integrator_config);
        m_use_partial_decode =  (m_energy_method == PCR || m_energy_method == AN08 || m_energy_method == AN08_ALL || m_energy_method == NEW_PCR) && get_json_value(integrator_config, "use_partial_decode", true);

        m_do_quasi_static = get_json_value(integrator_config, "quasi_static", false);

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
            std::cout << "Done. Will sample from " << m_cubature_indices.size() << " tets." << std::endl;
        }
        else if(m_energy_method == AN08 || m_energy_method == NEW_PCR) {
            fs::path energy_model_dir;
            
            if(m_energy_method == AN08) {
                energy_model_dir = model_root / ("energy_model/an08/pca_dim_" + std::to_string(m_U.cols()));
                if(!boost::filesystem::exists(energy_model_dir)) { // Old version
                    energy_model_dir = model_root / "energy_model/an08/";
                }
            }
            if(m_energy_method == NEW_PCR) energy_model_dir = model_root / "energy_model/new_pcr/";

            fs::path indices_path = energy_model_dir / "indices.dmat";
            fs::path weights_path = energy_model_dir / "weights.dmat";

            Eigen::VectorXi Is;
            Eigen::VectorXd Ws;
            igl::readDMAT(indices_path.string(), Is); //TODO do I need to sort this?
            igl::readDMAT(weights_path.string(), Ws); 

            m_cubature_weights = Ws;
            m_cubature_indices = Is;
            std::cout << "Done. Will sample from " << m_cubature_indices.size() << " tets." << std::endl;
        } else if(m_energy_method == AN08_ALL) {
            fs::path energy_model_dir = model_root / ("energy_model/an08/pca_dim_" + std::to_string(m_U.cols()));
            if(!boost::filesystem::exists(energy_model_dir)) { // Old version
                energy_model_dir = model_root / "energy_model/an08/";
            }
            load_all_an08_indices_and_weights(energy_model_dir, m_all_cubature_indices, m_all_cubature_weights);

            m_cubature_weights = m_all_cubature_weights.back();
            m_cubature_indices = m_all_cubature_indices.back();
        }

        // Computing the sampled verts
        if (m_energy_method == AN08 || m_energy_method == PCR || m_energy_method == NEW_PCR || m_energy_method == AN08_ALL) {
            initialize_cubature();
        }
    }

    void initialize_cubature() {
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
        std::sort(m_cubature_vert_indices.data(), m_cubature_vert_indices.data()+m_cubature_vert_indices.size());
        MatrixXd denseU = m_U;  // Need to cast to dense to support sparse in full space
        m_U_sampled = igl::slice(denseU, m_cubature_vert_indices, 1);


        // Get the element references ahead of time
        m_cubature_elements.clear();
        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            int tet_index = m_cubature_indices[i];  
            auto elem = m_tets->getImpl().getElement(tet_index);
            m_cubature_elements.push_back(elem);
        }

        // Get the rows of U ahead of time
        m_U_cubature_sampled.resize(n_sample_tets * 12, m_U.cols());
        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            for(int k = 0; k < 4; k++) {
                for(int j = 0; j < 3; j++) {    
                    m_U_cubature_sampled.row(i * 12 + k * 3 + j) = m_cubature_weights(i) * denseU.row(m_cubature_elements[i]->getQDOFList()[k]->getGlobalId() + j);
                }
            }
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



    double parallel_current_reduced_energy_and_forces(double &energy, VectorXd &UT_forces) {
        #pragma omp single 
        {
            UT_forces = VectorXd::Zero(m_U.cols());
            energy = 0;
        }

        const int n_sample_tets = m_cubature_indices.size();
        const int n_force_per_element = T.cols() * 3;
        const int d = m_U.cols();

        VectorXd sampled_force(n_force_per_element); // Holds the force for a single tet
        VectorXd element_reduced_force(d);
        element_reduced_force.setZero();

        double sub_energy = 0.0;

        #pragma omp for nowait //schedule(static)// firstprivate(sampled_force, element_reduced_force) // schedule(static,512)// TODO how to optimize this
        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            int tet_index = m_cubature_indices[i];
            auto elem = m_cubature_elements[i];

            // Energy and forces
            double tet_energy = elem->getStrainEnergy(m_world->getState());
            elem->getInternalForce(sampled_force, m_world->getState());

            element_reduced_force.noalias() += m_U_cubature_sampled.block(i*12, 0, 12, d).transpose() * sampled_force;
            sub_energy += m_cubature_weights(i) * tet_energy;
        }

        #pragma omp critical
        {
            UT_forces += element_reduced_force; //Does this have to do a transpose to assign?
            energy += sub_energy;
        }
    }



    double operator()(const VectorXd& new_z, VectorXd& grad, const int cur_iteration, const int cur_line_search_iteration)
    {

        // Update the tets with candidate configuration
        double obj_start_time = igl::get_seconds();
        VectorXd new_sub_q = sub_dec(new_z);
        double tf_time = igl::get_seconds() - obj_start_time;
        tf_decode_time_tot += tf_time;

        double decode_start_time = igl::get_seconds();
        Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(m_tets->getQ(), *m_world);
        
        // Define up here so they are shared with threads
        double energy;
        VectorXd UT_internal_forces;
        
        #ifdef DO_TIMING
        double energy_forces_start_tim = 0.0;
        double predict_weight_time = 0.0;
        double decode_time;
        #endif

        double energy_forces_start_time = igl::get_seconds();
        if(m_energy_method != FULL) {
            // If we are using cubature then we can leverage parallelism
            #pragma omp parallel num_threads(8) // TODO is it possible to move this out of the optimization loop?
            {
                // Update our current state
                if(m_use_partial_decode) {
                    // We only need to update verts for the tets we actually sample
                    #pragma omp for
                    for(int i = 0; i < m_U_sampled.rows(); i++) {
                        q[m_cubature_vert_indices[i]] = m_U_sampled.row(i) * new_sub_q; //sampled_q[i]; // TODO this could probably be more efficient with a sparse operator?
                    }
                } else {
                    #pragma omp single
                    {
                        q = m_U * new_sub_q;
                    }
                }
                
                #ifdef DO_TIMING
                    #pragma omp single
                    {
                        decode_time = igl::get_seconds() - decode_start_time;
                        cuba_decode_time_tot += decode_time;
                        energy_forces_start_time = igl::get_seconds();
                    }
                #endif
                // Get energy and forces
                parallel_current_reduced_energy_and_forces(energy, UT_internal_forces);
            }
        } else {
            // Update our current state
            q = m_U * new_sub_q;

            #ifdef DO_TIMING
            decode_time = igl::get_seconds() - decode_start_time;
            energy_forces_start_time = igl::get_seconds();
            #endif

            // TWISTING
            bool m_do_full_space_twisting = false;
            if(m_do_full_space_twisting) {
                auto max_verts = get_min_verts(0, false, 0.021);
                Eigen::AngleAxis<double> rot(m_h * current_frame, Eigen::Vector3d(1.0,0.0,0.0));
                for(int i = 0; i < max_verts.size(); i++) {
                    int vi = max_verts[i];
                    Eigen::Vector3d v = V.row(vi);
                    Eigen::Vector3d new_q = rot * v - v;// + Eigen::Vector3d(1.0,0.0,0.0) * current_frame / 300.0;
                    for(int j = 0; j < 3; j++) {
                        q[vi * 3 + j] = new_q[j];
                    }
                }
            }

            // Get energy and forces
            energy = m_tets->getStrainEnergy(m_world->getState());
            getInternalForceVector(m_internal_force_asm, *m_tets, *m_world);
            UT_internal_forces = m_U.transpose() * *m_internal_force_asm;
        }
        

        #ifdef DO_TIMING
        double energy_forces_time = igl::get_seconds() - energy_forces_start_time;
        cuba_eval_time_tot += energy_forces_time;

        double obj_and_grad_time_start = igl::get_seconds();
        #endif

        // const VectorXd UT_F_interaction = m_interaction_force.transpose() * m_U;
        // const VectorXd h_2_UT_external_forces =  m_h * m_h * (UT_F_interaction + m_UT_F_ext);
        const VectorXd h_2_UT_external_forces = m_h * m_h * (m_U.transpose() * m_interaction_force + m_UT_F_ext);
        const VectorXd sub_q_tilde = new_sub_q - 2.0 * m_cur_sub_q + m_prev_sub_q;
        const VectorXd UTMU_sub_q_tilde = m_UTMU * sub_q_tilde;

        double obj_val;
        if(m_do_quasi_static) {
            obj_val = m_h * m_h * energy - sub_q_tilde.transpose() * h_2_UT_external_forces;
            grad = jtvp(new_z, - m_h * m_h * UT_internal_forces - h_2_UT_external_forces);
            
        } else { 
            obj_val = 0.5 * sub_q_tilde.transpose() * UTMU_sub_q_tilde
                            + m_h * m_h * energy
                            - sub_q_tilde.transpose() * h_2_UT_external_forces;
            VectorXd preGrad = UTMU_sub_q_tilde - m_h * m_h * UT_internal_forces - h_2_UT_external_forces;

            obj_grad_time_tot_no_jvp += igl::get_seconds() - obj_and_grad_time_start;
            double vjp_start = igl::get_seconds();
            grad = jtvp(new_z, preGrad);
            tf_vjp_time_tot += igl::get_seconds() - vjp_start;
        }

        #ifdef DO_TIMING
        double obj_and_grad_time = igl::get_seconds() - obj_and_grad_time_start;
        double obj_time = igl::get_seconds() - obj_start_time;
        #endif

        if(LOGGING_ENABLED) {
            timestep_info["iteration_info"]["lbfgs_iteration"].push_back(cur_iteration);
            timestep_info["iteration_info"]["lbfgs_line_iteration"].push_back(cur_line_search_iteration);
            timestep_info["iteration_info"]["lbfgs_obj_vals"].push_back(obj_val);

            #ifdef DO_TIMING
            timestep_info["iteration_info"]["timing"]["tot_obj_time_s"].push_back(obj_time);
            timestep_info["iteration_info"]["timing"]["tf_time_s"].push_back(tf_time);
            timestep_info["iteration_info"]["timing"]["linear_decode_time_s"].push_back(decode_time);
            timestep_info["iteration_info"]["timing"]["energy_forces_time_s"].push_back(energy_forces_time);
            timestep_info["iteration_info"]["timing"]["predict_weight_time"].push_back(predict_weight_time);
            timestep_info["iteration_info"]["timing"]["obj_and_grad_eval_time_s"].push_back(obj_and_grad_time);
            #endif

            if(m_save_obj_every_iteration) {
                VectorXd q = m_reduced_space->decode(new_z);
                Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
                fs::path obj_filename = iteration_obj_dir / (ZeroPadNumber(cur_iteration) + "_" + ZeroPadNumber(cur_line_search_iteration, 2) + ".obj");
                MatrixXd new_verts = V + dV.transpose();
                igl::writeOBJ(obj_filename.string(), new_verts, F);
            }
        }

        total_evals++;
        return obj_val;
    }

    VectorXi get_current_tets() {
        return m_cubature_indices;
    }

    void switch_to_next_tets(int i) { // This is an ugly dirty hack to get extra tets working for an08
        m_cubature_weights = m_all_cubature_weights[m_all_cubature_weights.size() - (i % m_all_cubature_weights.size()) - 1];
        m_cubature_indices = m_all_cubature_indices[m_all_cubature_weights.size() - (i % m_all_cubature_weights.size()) - 1];

        initialize_cubature();

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
    MatrixXd m_U_cubature_sampled;

    AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > m_M_asm;
    AssemblerParallel<double, AssemblerEigenVector<double> > m_internal_force_asm;

    double m_h;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    EnergyMethod m_energy_method;
    ReducedSpaceType *m_reduced_space;
    bool m_use_partial_decode = false;
    bool m_save_obj_every_iteration = false;
    bool m_UTMU_is_identity = false;
    bool m_do_quasi_static = false;

    MatrixXd m_energy_basis;
    VectorXd m_summed_energy_basis;
    MatrixXd m_energy_sampled_basis;
    Eigen::FullPivHouseholderQR<MatrixXd> m_energy_sampled_basis_qr;

    VectorXd m_cubature_weights;
    VectorXi m_cubature_indices;
    VectorXi m_cubature_vert_indices;
    std::vector<NeohookeanHFixedTet<double> *> m_cubature_elements;

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
        m_full_only_rest_prefactor = get_json_value(integrator_config, "full_only_rest_prefactor", false);
        m_is_full_space = integrator_config["reduced_space_type"] == "full";
        m_is_linear_space = integrator_config["reduced_space_type"] == "linear";
        m_h = integrator_config["timestep"];
        m_U = reduced_space->outer_jacobian();

        // // WHY ISNT THIS workWILL IT WORK FOR FULL SPACE?
        // // IS IT BECAUSE I am slice based on vert not dof????
        // VectorXi FIunqiue;
        // igl::unique(F, FIunqiue);
        // m_U_F_only.resize(m_U.rows(), m_U.cols());

        // std::vector< Eigen::Triplet<double> > tripletList;
        // tripletList.reserve(FIunqiue.size() * 3);
        // for (int i = 0; i < FIunqiue.size(); ++i)
        // {
        //     for (int j = 0; j < 3; ++j)
        //     {
        //         int dof_i = FIunqiue[i] * 3 + j;
        //         for(int k = 0; k < m_U.cols(); k++) {
        //             // m_U_F_only.insert(dof_i, m_U.coeff(dof_i, k));
        //             tripletList.push_back(Eigen::Triplet<double>(dof_i, k,  m_U.coeff(dof_i, k)));
        //         }
        //     }
        // }
        // m_U_F_only.setFromTriplets(tripletList.begin(), tripletList.end());


        // Set up lbfgs params
        json lbfgs_config = integrator_config["lbfgs_config"];
        m_lbfgs_param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
        m_lbfgs_param.m = lbfgs_config["lbfgs_m"];
        m_lbfgs_param.epsilon = lbfgs_config["lbfgs_epsilon"]; //1e-8// TODO replace convergence test with abs difference
        m_lbfgs_param.delta = get_json_value(lbfgs_config, "lbfgs_delta", 0.0);
        m_lbfgs_param.past = get_json_value(lbfgs_config, "lbfgs_delta_past", 0);
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
        getMassMatrix(m_M_asm, *m_world);
        getStiffnessMatrix(m_K_asm, *m_world);

        reset_zs();

        // Set up preconditioner
        if(m_use_preconditioner) {
            MatrixType J = reduced_space->inner_jacobian(m_cur_z);
            std::cout << "Constructing reduced mass and stiffness matrix..." << std::endl;
            double start_time = igl::get_seconds();
            m_UTMU = m_U.transpose() * (*m_M_asm) * m_U;
            m_UTKU = m_U.transpose() * (*m_K_asm) * m_U;
            m_H = J.transpose() * m_UTMU * J - m_h * m_h * J.transpose() * m_UTKU * J;

            m_use_pardiso = m_is_full_space && m_H.rows() > PARDISO_DOF_CUTOFF;
            std::cout << "Using pardiso: " << m_use_pardiso << std::endl;
            std::cout << "Rows in m_H: " << m_H.rows() << std::endl;
            if(m_use_pardiso) {
                m_H_solver_pardiso.compute(m_H);
            } else {
                m_H_solver_eigen.compute(m_H);
            }
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
            if(!m_is_full_space && !m_is_linear_space) { // Only update the hessian each time for nonlinear space. 
                MatrixType J = m_reduced_space->inner_jacobian(m_cur_z);
                // std::cout << "Frobenius norm of J: " << J.norm() << std::endl;
                m_H = J.transpose() * m_UTMU * J - m_h * m_h * J.transpose() * m_UTKU * J;
                m_H_solver_eigen.compute(m_H);//m_H.ldlt();
            }
            if(m_is_full_space && !m_full_only_rest_prefactor){ // Currently doing a FULL hessian update for the full space..
                // getMassMatrix(m_M_asm, *m_world);
                getStiffnessMatrix(m_K_asm, *m_world);
                m_UTMU = m_U.transpose() * (*m_M_asm) * m_U;
                m_UTKU = m_U.transpose() * (*m_K_asm) * m_U;
                m_H = m_UTMU - m_h * m_h * m_UTKU;
                if(m_use_pardiso) {
                    m_H_solver_pardiso.compute(m_H);
                } else {
                    m_H_solver_eigen.compute(m_H);
                }
            }
            precondition_compute_time = igl::get_seconds() - precondition_start_time;
            precon_time_tot += precondition_compute_time;
            if(m_use_pardiso) {
                niter = m_solver->minimizeWithPreconditioner(*m_gplc_objective, z_param, min_val_res, m_H_solver_pardiso);   
            } else {
                niter = m_solver->minimizeWithPreconditioner(*m_gplc_objective, z_param, min_val_res, m_H_solver_eigen);   
            }
            bool m_use_pardiso;
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

            // std::cout << "tf_decode_time_tot: " << tf_decode_time_tot / total_evals << std::endl;
            // std::cout << "tf_vjp_time_tot: " << tf_vjp_time_tot / total_evals << std::endl;
            // std::cout << "cuba_decode_time_tot: " << cuba_decode_time_tot / total_evals << std::endl;
            // std::cout << "cuba_eval_time_tot: " << cuba_eval_time_tot / total_evals << std::endl;
            // std::cout << "obj_grad_time_tot_no_jvp: " << obj_grad_time_tot_no_jvp / total_evals << std::endl;
            // std::cout << "precon_time_tot: " << precon_time_tot / total_evals << std::endl;
            std::cout << "Avg evals / frame: " << total_evals / (current_frame + 1.0) << std::endl;
            std::cout << "['cuba_eval_time_tot', 'cuba_decode_time_tot', 'obj_grad_time_tot_no_jvp', 'tf_decode_time_tot', 'tf_vjp_time_tot', 'precon_time_tot']" << std::endl;
            std::cout << "["<<  cuba_eval_time_tot / total_evals << ", " 
                           << cuba_decode_time_tot / total_evals <<  ", "
                           << obj_grad_time_tot_no_jvp / total_evals << ", "
                           << tf_decode_time_tot / total_evals << ", "
                           << tf_vjp_time_tot / total_evals << ", "
                           << precon_time_tot / total_evals << "]" << std::endl;

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
            timestep_info["mouse_info"]["dragged_mesh_pos"] = {dragged_mesh_pos[0], dragged_mesh_pos[1], dragged_mesh_pos[2]};
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

    void get_current_V_faces_only(MatrixXd &newV) {
        // Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
        // return V + dV.transpose();   

        VectorXd sub_q = m_reduced_space->sub_decode(m_cur_z);
        VectorXd q = m_U_F_only * sub_q ;
        Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
        newV = V + dV.transpose();
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
    bool m_full_only_rest_prefactor;
    bool m_is_full_space;
    bool m_is_linear_space;

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
        Eigen::SimplicialLDLT<SparseMatrix<double>> >::type m_H_solver_eigen;
    
    bool m_use_pardiso;
    #ifdef EIGEN_USE_MKL_ALL
    typename std::conditional<
        std::is_same<MatrixType, MatrixXd>::value,
        Eigen::LDLT<MatrixXd>,
        Eigen::PardisoLDLT<SparseMatrix<double>, Eigen::Lower> >::type m_H_solver_pardiso;
    #else
    typename std::conditional<
    typename std::conditional<
        std::is_same<MatrixType, MatrixXd>::value,
        Eigen::LDLT<MatrixXd>,
        Eigen::SimplicialLDLT<SparseMatrix<double>> >::type  m_H_solver_pardiso;
    #endif

    MatrixType m_H_inv;
    MatrixType m_UTKU;
    MatrixType m_UTMU;

    MatrixType m_U;
    SparseMatrix<double> m_U_F_only;

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


template <typename S, typename M>
SparseVector<double> compute_interaction_force(const Vector3d &dragged_pos, const std::vector<int> &dragged_verts, bool is_dragging, double spring_stiffness, GPLCTimeStepper<S,M> &gplc_stepper) {
    // double start = igl::get_seconds();
    SparseVector<double> force(n_dof);// = VectorXd::Zero(n_dof); // TODO make this a sparse vector?
    if(is_dragging) {
        force.reserve(3 * dragged_verts.size());
        for(int vi = 0; vi < dragged_verts.size(); vi++) {
            int dragged_vert_local = dragged_verts[vi];

            // It's because I'm only updating the positions of the sampled verts....
            // Vector3d fem_attached_pos = PosFEM<double>(&tets->getQ()[dragged_vert_local], dragged_vert_local, &tets->getImpl().getV())(world.getState());
            Vector3d fem_attached_pos = gplc_stepper.get_current_vert_pos(dragged_vert_local);
            Vector3d local_force = spring_stiffness * (dragged_pos - fem_attached_pos) / dragged_verts.size();
            // std::cout << (local_force).norm() << std::endl;
            for(int i=0; i < 3; i++) {
                force.insert(dragged_vert_local * 3 + i) = local_force[i];    
            }
        }
    } 
    // std::cout << "compute_interaction_force: " << igl::get_seconds() - start << std::endl;
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

    int png_scale = get_json_value(config, "png_scale", 1);
    bool show_camera_info = false;
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
 
    viewer.core.show_lines = get_json_value(visualization_config, "show_lines", false);
    viewer.core.invert_normals = true;
    if(NO_WAIT) {
        viewer.core.is_animating = true;
    } else {
        viewer.core.is_animating = false;
    }
    viewer.data.face_based = false;
    viewer.core.line_width = get_json_value(visualization_config, "line_width", 2.0);
    viewer.core.animation_max_fps = 1000.0;
    viewer.core.background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
    viewer.core.shininess = 120.0;

    viewer.data.set_colors(Eigen::RowVector3d(0.0, 0.0, 0.0));


    Eigen::RowVector3d free_color = Eigen::RowVector3d(94.0/255.0,185.0/255.0,238.0/255.0);
    Eigen::RowVector3d pinned_color = Eigen::RowVector3d(51/255.0, 61.0/255.0, 123.0/255.0);

    std::string space_type = integrator_config["reduced_space_type"];
    if(space_type == "autoencoder") {
        free_color = Eigen::RowVector3d(86.0/255.0,180.0/255.0,233.0/255.0);
        pinned_color = Eigen::RowVector3d(51/255.0, 61.0/255.0, 123.0/255.0);
    } else if(space_type == "linear") {
        free_color = Eigen::RowVector3d(230.0/255.0,159.0/255.0,0.0/255.0);
        pinned_color = Eigen::RowVector3d(138/255.0, 96.0/255.0, 0.0/255.0);
    } else {
        free_color = Eigen::RowVector3d(209.0/255.0,41.0/255.0,61.0/255.0);
        pinned_color = Eigen::RowVector3d(129/255.0, 25.0/255.0, 38.0/255.0);
    }

    // Set the faces that contain only fixed verts to a different colour
    // if(!do_gpu_decode) {

    //     // Per face
    //     MatrixXd C(F.rows(), 3);
    //     for(int i = 0; i < C.rows(); i++) {
    //         C.row(i) = free_color;
    //     }
    //     std::vector<std::vector<int>> VF;
    //     std::vector<std::vector<int>> VFi;
    //     igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);

    //     std::set<int> fixed_vert_set(fixed_verts.begin(), fixed_verts.end());
    //     for(const auto &fvi : fixed_verts) {
    //         for(const auto &fi : VF[fvi]) {
    //             bool all_in_set = true;
    //             for(int i = 0; i < 3; i++) {
    //                 all_in_set &= (bool)fixed_vert_set.count(F(fi, i));
    //             }
    //             if(all_in_set) {
    //                 C.row(fi) = pinned_color;
    //             }
    //         }
    //     }

    //     viewer.data.set_colors(C);
    // } else {

        // Per vert
   
    MatrixXd C(V.rows(), 3);
    for(int i = 0; i < C.rows(); i++) {
        C.row(i) = Eigen::RowVector3d(0.0, 0.0, 0.0);
    }
     if(!get_json_value(visualization_config, "disable_pinned_color", false)) {
        for(const auto &vi : fixed_verts) {
            C.row(vi) = Eigen::RowVector3d(1.0, 1.0, 1.0);
        }
    }

    viewer.data.set_colors(C);
    // }

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

    viewer.core.camera_zoom = get_json_value(visualization_config, "camera_zoom", 1.0);
    std::vector<float> default_angle(4, 0.0f);
    default_angle[0] = 1.0;
    std::vector<float> trackball_angle = get_json_value(visualization_config, "trackball_angle", default_angle);
    viewer.core.trackball_angle.w() = trackball_angle[0];
    viewer.core.trackball_angle.x() = trackball_angle[1];
    viewer.core.trackball_angle.y() = trackball_angle[2];
    viewer.core.trackball_angle.z() = trackball_angle[3];

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
uniform vec4 free_color;
uniform vec4 pinned_color;
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


float eps = 1e-6;
vec4 KdiNew = free_color * step(Kdi.x, 1.0 - eps) + pinned_color * (1.0 - step(Kdi.x, 1.0 - eps));
vec3 Id = Ld * vec3(KdiNew) * clamped_dot_prod;    // Diffuse intensity

vec3 reflection_eye = reflect (-direction_to_light_eye, normal_eye);
vec3 surface_to_viewer_eye = normalize (-position_eye);
float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
dot_prod_specular = float(abs(dot_prod)==dot_prod) * max (dot_prod_specular, 0.0);
float specular_factor = pow (dot_prod_specular, specular_exponent);
vec3 Kfi = 0.5*vec3(Ksi); // vec3(1.0,1.0,1.0);//
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
  (1.0-lighting_factor) * vec3(KdiNew),(Kai.a+Ksi.a+KdiNew.a)/3);
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

        GLint cyan_diffuse_loc = glGetUniformLocation(prog_id,"free_color");
        glUniform4f(cyan_diffuse_loc, free_color(0), free_color(1), free_color(2), 1.0);
        GLint fast_red_diffuse_loc = glGetUniformLocation(prog_id,"pinned_color");
        glUniform4f(fast_red_diffuse_loc, pinned_color(0), pinned_color(1), pinned_color(2), 1.0);
    }



    double cur_fps = 0.0;
    double tot_time = 0.0; 
    double last_time = igl::get_seconds();
    std::vector<std::shared_future<bool>> VF;
    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
        if(show_camera_info) {
            std::cout << "camera_zoom: " << viewer.core.camera_zoom << std::endl;
            std::cout << "trackball_angle: " << viewer.core.trackball_angle.w() << ", "
                                             << viewer.core.trackball_angle.x() << ", "
                                             << viewer.core.trackball_angle.y() << ", "
                                             << viewer.core.trackball_angle.z() << std::endl;
        }

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

        if(is_dragging) {
            dragged_mesh_pos = gplc_stepper.get_current_vert_pos(vis_vert_id);
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
            P.row(1) = dragged_mesh_pos;
            
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
            if(SAVE_PNGS) {
                    std::string frame_num_string = std::to_string(current_frame);// + starting_frame_num);
                    fs::path  png_file = save_pngs_dir / ("image_" + frame_num_string + ".png");
                    // igl::png::render_to_png_async(png_file.string(), 4000, 4000, true, true);
                    // igl::png::render_to_png(png_file.string(), 640, 480, true, true);
                    // Allocate temporary buffers
                    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(1280*png_scale,800*png_scale);
                    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(1280*png_scale,800*png_scale);
                    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(1280*png_scale,800*png_scale);
                    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(1280*png_scale,800*png_scale);

                    // // Draw the scene in the buffers
                    viewer.core.draw_buffer(viewer.data, viewer.opengl, false,R,G,B,A);
                    // // Save it to a PNG
                    VF.push_back(std::async(std::launch::async, igl::png::writePNG, R,G,B,A,png_file.string()));
                     // igl::png::writePNG(R,G,B,A,png_file.string());
                }

            if(!do_gpu_decode) {
                // auto q = mapDOFEigen(tets->getQ(), world);
                Eigen::MatrixXd newV = gplc_stepper.get_current_V();
                // gplc_stepper.get_current_V_faces_only(newV);

                if(show_tets) {
                    VectorXi tet_indices = gplc_stepper.get_current_tets();
                    T_sampled = igl::slice(T, tet_indices, 1);
                    igl::boundary_facets(T_sampled, F_sampled);

                    viewer.data.clear();
                    viewer.data.set_mesh(newV, F_sampled);
                } else {
                    viewer.data.set_vertices(newV);
                    // viewer.data.set_colors(C);
                }

                // if(per_vertex_normals) {
                //     igl::per_vertex_normals(V,F,N);
                // } else {
                //     igl::per_corner_normals(V,F,40,N);
                // }
                // viewer.data.set_normals(N);

            }

            // Play back mouse interaction from previous sim
            if(PLAYBACK_SIM) {
                // Eigen::MatrixXd newV = gplc_stepper.get_current_V(); // TODO we can avoid doing this
                if(current_frame >= sim_playback_json["timesteps"].size()) {
                    exit_gracefully();
                }

                json current_mouse_info = sim_playback_json["timesteps"][current_frame]["mouse_info"];
                dragged_pos = Eigen::RowVector3d(current_mouse_info["dragged_pos"][0], current_mouse_info["dragged_pos"][1], current_mouse_info["dragged_pos"][2]);
                dragged_mesh_pos = Eigen::RowVector3d(current_mouse_info["dragged_mesh_pos"][0], current_mouse_info["dragged_mesh_pos"][1], current_mouse_info["dragged_mesh_pos"][2]);

                bool was_dragging = is_dragging;
                is_dragging = current_mouse_info["is_dragging"];

                if(is_dragging && !was_dragging) { // Got a click
                    Eigen::MatrixXd curV = gplc_stepper.get_current_V();

                    MatrixXd C(1,3);
                    C.row(0) = dragged_mesh_pos;
                    MatrixXi I;
                    igl::snap_points(C, curV, I);

                    vis_vert_id = I(0,0);
                    
                    dragged_verts = get_verts_in_sphere(curV.row(vis_vert_id), spring_grab_radius, curV);
                    std::cout << "Grabbed " << dragged_verts.size() << std::endl;
                    // vis_vert_id = get closest point on mesh for dragged_mesh_pos using snap to
                    // dragged_face = current_mouse_info["dragged_face"];
                    // dragged_vert = current_mouse_info["dragged_vert"];

                    // MatrixXd P = 
                    // igl::point_mesh_squared_distance(P, V, F, sqrD, I, C);
                    // // Get closest point on mesh
                    

                }
            }
            
            // Do the physics update
            interaction_force = compute_interaction_force(dragged_pos, dragged_verts, is_dragging, spring_stiffness, gplc_stepper);
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

                // Export the pointer line
                MatrixXi pointerF(1, 2);
                pointerF << 0, 1;
                MatrixXd pointerV(2,3);
                if(is_dragging) {

                    pointerV.row(0) = newV.row(vis_vert_id);
                    pointerV.row(1) = dragged_pos;
                } else {
                    pointerV << 10000, 10000, 10000, 10001, 10001, 10001; // Fix this and make it work
                }

                fs::path pointer_filename(int_to_padded_str(current_frame) + "_mouse_pointer.obj");
                igl::writeOBJ((pointer_obj_dir / pointer_filename).string(), pointerV, pointerF);
                }

                if(SAVE_TRAINING_DATA) {
                    std::string frame_num_string = std::to_string(current_frame);// + starting_frame_num);
                    auto q = gplc_stepper.get_current_q();
                    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
                    Eigen::MatrixXd displacements = dV.transpose();

                    fs::path  displacements_file = save_training_data_dir / ("displacements_" + frame_num_string + ".dmat");
                    igl::writeDMAT(displacements_file.string(), displacements, false); // Don't use ascii
                }

                
            }


            if(current_frame % print_freq == 0) {   
                double cur_time = igl::get_seconds();
                if(current_frame >= 1) {
                   tot_time += cur_time - last_time;
                }
                
                cur_fps = print_freq / (cur_time - last_time);
                last_time = cur_time;
                std::cout << "Step time + rendering(Hz): "<< cur_fps << std::endl;
                std::cout << "Avg Step time + rendering(s): "<< tot_time / (current_frame) << std::endl;
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
            
            igl::jet(energy_per_face, false, C);
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
                viewer.data.set_colors(Eigen::RowVector3d(1.0, 1.0, 0.0));
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
                gplc_stepper.switch_to_next_tets(--index_counter);
                break;
            }
            case 'n':
            case 'N':
            {
                per_vertex_normals = !per_vertex_normals;
                break;
            }
            case 'c':
            case 'C':
            {
                show_camera_info = !show_camera_info;
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
            
            int fid;
            Eigen::Vector3f bary;
            // Find closest point on mesh to mouse position
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
                vis_vert_id = F(fid,c);

                dragged_pos = curV.row(vis_vert_id);
                dragged_mesh_pos = dragged_pos; // Just using closest vert for now
                dragged_verts = get_verts_in_sphere(dragged_mesh_pos, spring_grab_radius, curV);

                std::cout << "Grabbed " << dragged_verts.size() << " verts" << std::endl;
                is_dragging = true;

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
        if(std::string(argv[2]) != "--log_name") { // hack to save the log name I want
            fs::path sim_recording_path(argv[2]);
            std::ifstream recording_fin(sim_recording_path.string());

            recording_fin >> sim_playback_json;
            PLAYBACK_SIM = true;
        }
    }


    std::ifstream fin((model_root / sim_config).string());
    json config;
    fin >> config;

    auto integrator_config = config["integrator_config"];
    auto material_config = config["material_config"];
    auto viz_config = config["visualization_config"];
    std::string reduced_space_string = integrator_config["reduced_space_type"];

    print_freq = get_json_value(config["visualization_config"], "print_every_n_frames", print_freq);

    std::string log_string = "";
    if(argc == 4 && std::string(argv[3]) == "--no_wait" ) {
        NO_WAIT = true;
    } else if(argc == 4) {
        log_string = argv[3];
    } else {
        log_string = currentDateTime() + "_" + reduced_space_string;
    }

    LOGGING_ENABLED = config["logging_enabled"];
    if(LOGGING_ENABLED) {
        sim_log["sim_config"] = config;
        sim_log["timesteps"] = {};
        fs::path sim_log_path("simulation_logs/");
        log_dir = model_root / fs::path("./simulation_logs/" + log_string + "/");
        surface_obj_dir = log_dir / fs::path("objs/surface/");
        pointer_obj_dir = log_dir / fs::path("objs/pointer/");
        tet_obj_dir = log_dir / fs::path("objs/sampled_tets/");
        iteration_obj_dir = log_dir / fs::path("objs/evaluations/");
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

        if(!boost::filesystem::exists(iteration_obj_dir) && get_json_value(integrator_config, "save_obj_every_iteration", false)){
            boost::filesystem::create_directories(iteration_obj_dir);
        }

        // boost::filesystem::create_directories(log_dir);
        fs::path log_file("sim_stats.json");

        log_ofstream = std::ofstream((log_dir / log_file).string());
    }

    std::string alternative_mesh_path = get_json_value(config, "alternative_full_space_mesh", std::string(""));

    // Now do the set up for stuff if we are saving training data
    SAVE_TRAINING_DATA = get_json_value(config, "save_training_data", false);
    if(SAVE_TRAINING_DATA) {
        std::string save_training_data_path_string = get_json_value(config, "save_training_data_path", std::string());
        save_training_data_dir = fs::path(save_training_data_path_string);
        create_or_replace_dir(save_training_data_dir);
        std::cout << "SAVING TRAINING DATA TO " << save_training_data_dir.string() << std::endl;

        // Save the training params
        json training_data_params;
        training_data_params["density"] = material_config["density"];
        training_data_params["YM"] = material_config["youngs_modulus"];
        training_data_params["Poisson"] = material_config["poissons_ratio"];
        training_data_params["time_step"] = integrator_config["timestep"];
        training_data_params["spring_strength"] = viz_config["interaction_spring_stiffness"];
        training_data_params["spring_grab_radius"] = viz_config["spring_grab_radius"];
        training_data_params["fixed_axis"] = viz_config["full_space_constrained_axis"];
        training_data_params["constrained_axis_eps"] = viz_config["constrained_axis_eps"];
        training_data_params["flip_fixed_axis"] = viz_config["flip_constrained_axis"];
        training_data_params["fixed_point_constraint"] = viz_config["fixed_point_constraint"];
        training_data_params["fixed_point_radius"] = viz_config["fixed_point_radius"];

        std::ofstream fout((save_training_data_dir / "parameters.json").string());
        fout << training_data_params;
        fout.close();

        if(alternative_mesh_path != "") {
            fs::copy_file(alternative_mesh_path, save_training_data_dir / "tets.mesh" );
        } else {
            fs::copy_file((model_root / "tets.mesh").string(), save_training_data_dir / "tets.mesh" );
        }
    }

    SAVE_PNGS = get_json_value(config, "save_pngs", false);
    if(SAVE_PNGS) {
        // std::string save_pngs_path_string = get_json_value(config, "save_pngs_path", std::string());
        save_pngs_dir =  log_dir / fs::path("pngs/");
        create_or_replace_dir(save_pngs_dir);
    }

    // Load the mesh here
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

    // Hacky way of setting spring grab radius to zero for reduced space
    spring_grab_radius = get_json_value(config["visualization_config"], "spring_grab_radius", 0.03);
    if(reduced_space_string != "full") {
        bool use_spring_grab_radius_for_reduced = get_json_value(config["visualization_config"], "use_spring_grab_radius_for_reduced", false);
        if(!use_spring_grab_radius_for_reduced) {
            spring_grab_radius = get_json_value(config["visualization_config"], "spring_grab_radius", 0.03);
            VectorXd R;
            igl::circumradius(V, F, R);
            spring_grab_radius = R.maxCoeff();//Make it slightly bigger than the biggest triangle
        }
    }

    // Do this out here so we can still visualize the constrained verts
    int fixed_axis = -1;
    try {
        fixed_axis = config["visualization_config"].at("full_space_constrained_axis");
    } 
    catch (nlohmann::detail::out_of_range& e){
        std::cout << "full_space_constrained_axis field not found in visualization_config" << std::endl;
        exit(1);
    }
    bool flip_axis = get_json_value(config["visualization_config"], "flip_constrained_axis", false);
    std::vector<double> fixed_vert_c = get_json_value(config["visualization_config"], "fixed_point_constraint", std::vector<double>());
    double fixed_vert_r = get_json_value(config["visualization_config"], "fixed_point_radius", 0.0);
    double constrained_axis_eps = get_json_value(config["visualization_config"], "constrained_axis_eps", 0.01);

    if(fixed_axis != -1) {
        fixed_verts = get_min_verts(fixed_axis, flip_axis, constrained_axis_eps);
    } else if (fixed_vert_r != -1) {
        fixed_verts = get_verts_in_sphere(Eigen::Map<VectorXd>(fixed_vert_c.data(), 3), fixed_vert_r, V);
    }

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
        

        std::cout << "Constraining " << fixed_verts.size() << " verts" << std::endl;
        // For twisting
        // auto max_verts = get_min_verts(fixed_axis, true);
        // fixed_verts.insert(fixed_verts.end(), max_verts.begin(), max_verts.end());

        SparseMatrix<double> P = construct_constraints_P(V, fixed_verts); // Constrain on X axis
        SparseConstraintSpace reduced_space(P.transpose());

        run_sim<SparseConstraintSpace, SparseMatrix<double>>(&reduced_space, config, model_root);
    }
    else {
        std::cout << "Not yet implemented." << std::endl;
        return 1;
    }
    
    return EXIT_SUCCESS;

}

