#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include <igl/readMESH.h>
#include <igl/get_seconds.h>

#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
#include <ConstraintFixedPoint.h>

#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

#include <omp.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>

#include "GreedyCubop.h"

using json = nlohmann::json;
namespace fs = boost::filesystem;
using namespace std;
using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; 

typedef PhysicalSystemFEM<double, NeohookeanTet> NeohookeanTets;

typedef World<double,
    std::tuple<PhysicalSystemParticleSingle<double> *, NeohookeanTets *>,
    std::tuple<ForceSpringFEMParticle<double> *>, std::tuple<ConstraintFixedPoint<double> *> > MyWorld;


// Globals
Eigen::MatrixXd V; // Verts
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices
fs::path model_root;
int goal_tet_count = 0;
double starting_time;

void eig_to_VEC(const Eigen::VectorXd &eig_vec, VECTOR &vec_vec) {
    vec_vec.resizeAndWipe(eig_vec.size());
    std::memcpy(&vec_vec(0), eig_vec.data(), eig_vec.size() * sizeof(double));
}

Eigen::VectorXd VEC_to_eig(VECTOR &vec) {
    Eigen::VectorXd eig(vec.size());
    std::memcpy(eig.data(), &vec(0), vec.size() * sizeof(double));   
    return eig;
}

std::string ZeroPadNumber(int num)
{
    std::ostringstream ss;
    ss << std::setw( 7 ) << std::setfill( '0' ) << num;
    return ss.str();
}

template<typename T>
T get_json_value(const json &j, const std::string &key, T def) {
    try {
        return j.at(key);
    }
    catch (nlohmann::detail::out_of_range& e){
        return def;
    }
}


class MyGreedyCubop : public GreedyCubop {
public:
    MyGreedyCubop(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, const Eigen::MatrixXd &U, double YM, double poisson, double density, const std::vector<Eigen::VectorXd> &reduced_forces, const std::vector<Eigen::VectorXd> &red_displacements) : GreedyCubop(), m_reduced_forces(reduced_forces), m_red_displacements(red_displacements) {
        m_tets = make_unique<NeohookeanTets>(V,T);
        for(auto element: m_tets->getImpl().getElements()) {
            element->setDensity(density);
            element->setParameters(YM, poisson);
        }
        // m_world.addSystem(m_tets.get());
        // m_world.finalize();

        m_U = U;
    }

protected:
    /**
     * Return the total number of points we have to choose from.
     * For example, for FEM piecewise-linear tetrahedra elements, this is just the number of tets.
     */
    int numTotalPoints() {
        return m_tets->getImpl().getNumElements();
    }



    /**
     * The main method that subclasses must implement. This should evaluate the reduced force density (or just the force, since
     * the volume/area is constant anyway) at the given point.
     *
     *  pointId - an index in [0, getNumTotalPoints()). Implementations must map this index deterministically to a cubature point.
     */
    void evalPointForceDensity( int pointId, VECTOR& q, VECTOR& gOut ) {
        // THIS ISN'T THREAD SAFE WITHOUT COPYING THE WORLD
        // Make a copy of the world every time... Not the most efficient, but still faster than being single threaded
        // double start = igl::get_seconds();

        MyWorld m_world;
        m_world.addSystem(m_tets.get());
        m_world.finalize();

        // update position of only the tet pointId
        Eigen::VectorXd eig_q = VEC_to_eig(q);
        auto gauss_map_q = mapDOFEigen(m_tets->getQ(), m_world);
        const int n_verts = 4;
        std::vector<int> verts(n_verts);
        for(int i = 0; i < n_verts; i++) {
            const int tet_vert = m_tets->getImpl().getElement(pointId)->getQDOFList()[i]->getGlobalId();
            verts[i] = tet_vert;

            for(int j = 0; j < 3; j++) {
                gauss_map_q[tet_vert + j] = m_U.transpose().row(tet_vert + j) * eig_q;
            }
        }
        // gauss_map_q = m_U.transpose() * eig_q;

        // Get the element force
        Eigen::VectorXd sampled_force(12);
        m_tets->getImpl().getElement(pointId)->getInternalForce(sampled_force, m_world.getState());

        // Assemble and project the full force
        Eigen::SparseVector<double> full_force(m_U.cols());
        for(int i = 0; i < n_verts; i++) {
            int vert_index = verts[i];//m_tets->getImpl().getElement(pointId)->getQDOFList()[i]->getGlobalId();
            for(int j = 0; j < 3; j++) {    
                full_force.coeffRef(vert_index + j) = sampled_force[i * 3 + j];
            }
        }
        
        eig_to_VEC(m_U * full_force, gOut);
        // std::cout << "evalPointForceDensity: " << igl::get_seconds() - start << "s" << std::endl;
    }

    /**
     * At the end of each iteration (_not_ a sub-train iteration), when some cubature has been optimized, this will be called.
     * Implementations should probably overwrite this and write the cubature out to file or something.
     */
    void handleCubature( std::vector<int>& selectedPoints, VECTOR& weights, Real relErr ) {
        stringstream my_out;

        my_out << "n_tets: " << selectedPoints.size() << endl;
        my_out << "relErr: " << relErr << endl;
        my_out << "Selected tets: ";
        for (std::vector<int>::iterator i = selectedPoints.begin(); i != selectedPoints.end(); ++i)
        {
            my_out << *i << ", ";
        }
        my_out << endl;

        std::vector<int> nonzero_indices;
        std::vector<double> nonzero_weights;
        int n_nonzero = 0;
        my_out << "Weights: ";
        for (int i = 0; i < weights.size(); ++i)
        {
            if(weights(i) > 0.00001) {
                n_nonzero++;
                nonzero_indices.push_back(selectedPoints[i]);
                nonzero_weights.push_back(weights(i));
            }

            my_out << weights(i) << ", ";
        }
        my_out << endl << "n_nonzero: " << n_nonzero << endl;
        my_out << endl;
        my_out << "Running time so far: " << igl::get_seconds() - starting_time << "s" << std::endl;
        my_out << endl;

        Eigen::VectorXi Is = Eigen::Map<Eigen::VectorXi>(&nonzero_indices[0], nonzero_indices.size());
        Eigen::VectorXd Ws = Eigen::Map<Eigen::VectorXd>(&nonzero_weights[0], nonzero_weights.size());

        fs::path energy_model_dir = model_root / ("energy_model/an08/pca_dim_" + std::to_string(m_U.rows()) + "/");
        fs::path this_iteration_output_dir = energy_model_dir / (ZeroPadNumber(n_nonzero) + "_samples/");
        fs::create_directories(this_iteration_output_dir);

        fs::path indices_path = this_iteration_output_dir / "indices.dmat";
        fs::path weights_path = this_iteration_output_dir / "weights.dmat";
        igl::writeDMAT(indices_path.string(), Is);
        igl::writeDMAT(weights_path.string(), Ws);

        fs::path details_path = this_iteration_output_dir / "details.txt";
        ofstream fout(details_path.string());
        fout << my_out.str();
        fout.close();

        cout << my_out.str();
        cout << "Saved weights and indices to " << this_iteration_output_dir.string() << endl << endl;

        const bool DEBUG = false;
        if(DEBUG) {
            int example_id = 500 % m_reduced_forces.size();//250;
            Eigen::VectorXd actual_g = m_reduced_forces[example_id];
            Eigen::VectorXd pred_g = get_predicted_force(example_id, nonzero_indices, nonzero_weights);

            std::cout << actual_g.transpose() << std::endl;
            std::cout << pred_g.transpose() << std::endl;

            std::cout << (actual_g - pred_g).transpose().cwiseAbs() << std::endl;

            std::cout << (actual_g - pred_g).squaredNorm() << std::endl;
        }


        // output everything every frame
        {
            fs::path indices_path = energy_model_dir / "indices.dmat";
            fs::path weights_path = energy_model_dir / "weights.dmat";
            igl::writeDMAT(indices_path.string(), Is);
            igl::writeDMAT(weights_path.string(), Ws);

            fs::path details_path = energy_model_dir / "details.txt";
            ofstream fout(details_path.string());
            fout << my_out.str();
            fout.close();
        }
        if(n_nonzero >= goal_tet_count) {
            cout << "Reached goal number of tets. Exiting." << endl;
            exit(0);
        }
    }

    Eigen::VectorXd get_predicted_force(int example_id, const std::vector<int> &nonzero_indices, const std::vector<double> &nonzero_weights) {
        Eigen::VectorXd pred_g = Eigen::VectorXd::Zero(m_U.rows());

        MyWorld world;
        world.addSystem(m_tets.get());
        world.finalize();
        auto gauss_map_q = mapDOFEigen(m_tets->getQ(), world);
        gauss_map_q =  m_U.transpose() * m_red_displacements[example_id]; // Update the mesh


        int n_sample_tets = nonzero_indices.size();
        Eigen::SparseMatrix<double> neg_energy_sample_jac(m_U.cols(), T.rows());// m_cubature_indices.size());
        neg_energy_sample_jac.reserve(Eigen::VectorXi::Constant(m_U.cols(), T.cols() * 3));
        
            
        Eigen::VectorXd cubature_weights = Eigen::Map<const Eigen::VectorXd>(&nonzero_weights[0], nonzero_weights.size());
        
        int n_force_per_element = T.cols() * 3;
        Eigen::MatrixXd element_forces(n_sample_tets, n_force_per_element);
        Eigen::VectorXd energy_samp(n_sample_tets);


        for(int i = 0; i < n_sample_tets; i++) { // TODO parallel
            int tet_index = nonzero_indices[i];
            energy_samp[i] = m_tets->getImpl().getElement(tet_index)->getStrainEnergy(world.getState());
            // std::cout << energy_samp[i] << std::endl;
            //Forces
            Eigen::VectorXd sampled_force(n_force_per_element);
            m_tets->getImpl().getElement(tet_index)->getInternalForce(sampled_force, world.getState());
            element_forces.row(i) = sampled_force;
        }

        // std::cout << element_forces << std::endl;

        for(int i = 0; i < n_sample_tets; i++) {
            int tet_index = nonzero_indices[i];
            for(int j = 0; j < 4; j++) {
                int vert_index = m_tets->getImpl().getElement(tet_index)->getQDOFList()[j]->getGlobalId();
                for(int k = 0; k < 3; k++) {
                    neg_energy_sample_jac.insert(vert_index + k, i) = element_forces(i, j*3 + k);
                }
            }
        }
        pred_g = m_U * (neg_energy_sample_jac * cubature_weights);



        return pred_g;
    }

private:
    // MyWorld m_world;
    unique_ptr<NeohookeanTets> m_tets;
    Eigen::MatrixXd m_U;
    const std::vector<Eigen::VectorXd> &m_reduced_forces;
    const std::vector<Eigen::VectorXd> &m_red_displacements;
};


std::vector<int> get_file_numbers_for_prefix(std::string prefix, fs::path dir) {
    std::vector<int> nums;
    for (auto i = fs::directory_iterator(dir); i != fs::directory_iterator(); i++)
    {
        fs::path cand_path = i->path();

        std::string name = cand_path.filename().string(); 
        if(strncmp(name.c_str(), prefix.c_str(), prefix.size()) == 0) { // Is forces
            std::string num_str = name.substr(prefix.size());
            nums.push_back(std::stoi(num_str, nullptr));
        }
    }
    std::sort(nums.begin(), nums.end());
    return nums;
}

std::vector<Eigen::VectorXd> load_forces(fs::path training_data_root, fs::path reduced_basis_path) {
    std::cout << "Loading reduced internal forces..." << std::endl;
    Eigen::MatrixXd U;
    igl::readDMAT(reduced_basis_path.string(), U);
    U.transposeInPlace();

    std::string prefix = "internalForces_";
    std::vector<int> forces_nums = get_file_numbers_for_prefix(prefix, training_data_root);
    std::vector<Eigen::VectorXd> recorded_forces(forces_nums.size());
    
    std::cout << "num threads: " << omp_get_num_threads() << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < forces_nums.size(); i++)
    {//   std::vector<int>::iterator i = forces_nums.begin(); i != forces_nums.end(); ++i
        fs::path dis_path = training_data_root / (prefix + std::to_string(forces_nums[i]) + ".dmat");
        Eigen::VectorXd F;
        igl::readDMAT(dis_path.string(), F);
        recorded_forces[i] = U * F;
    }

    std::cout << "Done." << std::endl;

    return recorded_forces;
}

std::vector<Eigen::VectorXd> load_displacements(fs::path training_data_root, fs::path reduced_basis_path) {
    std::cout << "Loading displacements..." << std::endl;
    Eigen::MatrixXd U;
    igl::readDMAT(reduced_basis_path.string(), U);
    U.transposeInPlace();


    std::string prefix = "displacements_";
    std::vector<int> displacements_nums = get_file_numbers_for_prefix(prefix, training_data_root);
    std::vector<Eigen::VectorXd> recorded_displacements(displacements_nums.size());

    #pragma omp parallel for
    for (int i = 0; i < displacements_nums.size(); i++)
    {   
        fs::path dis_path = training_data_root / (prefix + std::to_string(displacements_nums[i]) + ".dmat");
        Eigen::MatrixXd Q;
        igl::readDMAT(dis_path.string(), Q);
        Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(Q.transpose().data(), Q.rows() * Q.cols());
        recorded_displacements[i]=(U*q);
    }
    std::cout << "Done." << std::endl;

    return recorded_displacements;
}

void progress_bar(double progress, int barWidth = 70) {
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

std::vector<Eigen::VectorXd> get_forces_from_reduced_displacements(const std::vector<Eigen::VectorXd> &red_displacements, const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, const Eigen::MatrixXd &U, double YM, double poisson, double density) {
    std::cout << "Generating forces for reduced poses..." << std::endl;
    unique_ptr<NeohookeanTets> tets = make_unique<NeohookeanTets>(V,T);
    
    for(auto element: tets->getImpl().getElements()) {
        element->setDensity(density);//1000.0);
        element->setParameters(YM, poisson);
    }
    std::vector<Eigen::VectorXd> forces(red_displacements.size());
    int n_done = 0;
    double start = igl::get_seconds();
    #pragma omp parallel
    {
        MyWorld world;
        world.addSystem(tets.get());
        world.finalize();


        #pragma omp for
        for(int i = 0; i < red_displacements.size(); i++) {
            Eigen::Map<Eigen::VectorXd> gauss_map_q = mapDOFEigen(tets->getQ(), world);
            gauss_map_q = U.transpose() * red_displacements[i];

            AssemblerEigenVector<double> internal_force;
            getInternalForceVector(internal_force, *tets, world);
            forces[i] = (U * (*internal_force));
            // std::cout << forces.back() << std::endl;
            #pragma omp atomic
            n_done++;

            if(i % 20 == 0) progress_bar(n_done / (double) red_displacements.size());            
        }
    }
    std::cout << "Generating forces: " << igl::get_seconds() - start << "s" << std::endl; 
    std::cout << std::endl;
    return forces;
}

int main(int argc, char **argv) {
    starting_time = igl::get_seconds();

    if(argc < 3) {
        cout << "Need to pass in a path to model root and a goal number of tets." << endl;
        exit(1);
    }

 
    // ---- Set Up
    model_root = fs::path(argv[1]);
    goal_tet_count = std::stoi(argv[2]);

    fs::path reduced_basis_path = model_root / "pca_results" / "ae_pca_components.dmat";

    int pca_dim;
    if(argc == 4) {
        pca_dim = std::stoi(argv[3]);
        reduced_basis_path = model_root / "pca_results" / ("pca_components_" + std::to_string(pca_dim) + ".dmat");
    }


    fs::path model_config_path = model_root / "model_config.json";
    std::ifstream fin_model(model_config_path.string());
    json model_config;
    fin_model >> model_config;

    std::string tdr = model_config["training_dataset"];
    fs::path training_data_root = tdr;
    std::cout << training_data_root << std::endl;
    fs::path mesh_path = model_root / "tets.mesh";
    fs::path sim_config_path = model_root / "sim_config.json";

    // Load sim config
    std::ifstream fin(sim_config_path.string());
    json sim_config;
    fin >> sim_config;
    double YM = sim_config["material_config"]["youngs_modulus"];
    double poisson = sim_config["material_config"]["poissons_ratio"];
    double density = sim_config["material_config"]["density"];

    // Load data
    igl::readMESH(mesh_path.string(), V, T, F);

    Eigen::MatrixXd U;
    igl::readDMAT(reduced_basis_path.string(), U);
    U.transposeInPlace();
    std::cout << U.rows() << " " << U.cols() << std::endl;

    // std::vector<Eigen::VectorXd> reduced_forces = load_forces(training_data_root, reduced_basis_path); // TODO maybe subsample?
    std::vector<Eigen::VectorXd> red_displacements = load_displacements(training_data_root, reduced_basis_path);
    // std::random_shuffle(red_displacements.begin(), red_displacements.end());
    // std::vector<Eigen::VectorXd> less_displacements(red_displacements.begin(), red_displacements.begin() + 100);
    std::vector<Eigen::VectorXd> reduced_forces = get_forces_from_reduced_displacements(red_displacements, V, T, U, YM, poisson, density);

    // *** TODO ***
    // I should test this by summing up the reduced forces at each tet and making sure they are equal to the full reduced force
    // It's also possible that I should be generating the full reduced forces after projecting into, and out of, the reduced space
    // for each training example...

    // ---- Convert it all to the cubacode format
    int n_poses = reduced_forces.size();
    int r = reduced_forces[0].size();
    cout << "T: " << n_poses << endl;
    cout << "r: " << r << endl;
    cout << "r*T: " << (n_poses*r) << endl;
    cout << "Tets: " << T.rows() << endl;

    // I have to construct the *reduced* forces evaluated at each of the total T tets.
    VECTOR trainingForces(r * n_poses);
    for(int i = 0; i < n_poses; i++) {
        for(int j = 0; j < r; j++) {
            trainingForces(i*r + j) = reduced_forces[i][j];
        }
    }

    //What's this? -> It's all the configurations for each training pose
    std::vector<VECTOR> training_poses(red_displacements.size());
    TrainingSet trainingSet;
    for(int i = 0; i < n_poses; i++) {
        eig_to_VEC(red_displacements[i], training_poses[i]);
        trainingSet.push_back(&training_poses[i]);
    }

    // Set up the optimization
    MyGreedyCubop cubop(V, T, U, YM, poisson, density, reduced_forces, red_displacements);

    // Params 
    Real relErrTol = get_json_value(model_config["learning_config"]["energy_model_config"], "rel_error_tol", 0.05);//0.05; // What's a good val?
    int maxNumPoints = goal_tet_count * 5; // some sane limit, for overnight runs
    int numCandsPerIter = 200;//100;//T.rows() / 100;  //100;// default 100;  // |C|
    int itersPerFullNNLS = r/2; // r/2 in the paper
    int numSamplesPerSubtrain = 50; //training_poses.size() / 4; // default 50;   // T_s
    
    std::cout << "relErrTol: " << relErrTol << std::endl;    
    std::cout << "maxNumPoints: " << maxNumPoints << std::endl;
    std::cout << "numCandsPerIter: " << numCandsPerIter << std::endl;    
    std::cout << "itersPerFullNNLS: " << itersPerFullNNLS << std::endl;
    std::cout << "numSamplesPerSubtrain: " << numSamplesPerSubtrain << std::endl;

    cout << "Working" << endl;

    cubop.run(
        trainingSet,
        trainingForces,
        relErrTol,
        maxNumPoints,
        numCandsPerIter,
        itersPerFullNNLS,
        numSamplesPerSubtrain
    );

    cout << "Didn't reach goal number of tets." << endl;

    return 0;
}
