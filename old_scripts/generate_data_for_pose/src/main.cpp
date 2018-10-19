#include <vector>

//#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>
#include <TimeStepperEulerImplicit.h>
#include <AssemblerParallel.h>

#include <igl/get_seconds.h>
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>

#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <string.h>
#include <json.hpp>
#include <boost/filesystem.hpp>


using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::SparseVector;

namespace fs = boost::filesystem;
using json = nlohmann::json;
using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

typedef PhysicalSystemFEM<double, NeohookeanTet> NeohookeanTets;
typedef World<double,
                        std::tuple<PhysicalSystemParticleSingle<double> *, NeohookeanTets *>,
                        std::tuple<ForceSpringFEMParticle<double> *>,
                        std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
typedef TimeStepperEulerImplicit<double, AssemblerEigenSparseMatrix<double>,
 AssemblerEigenVector<double> > MyTimeStepper;

json sim_params;
MatrixXd V;
MatrixXi F, T;

// Take a start and a range so that we can do this in parallel.
void generateEnergyForPoses(NeohookeanTets *tets, const MatrixXd &displacements, int start, int num, MatrixXd &energy_vec_per_pose) {
    MyWorld world;
    world.addSystem(tets);
    fixDisplacementMin(world, tets, sim_params["displacement_axis"], sim_params["displacement_tol"]);
    world.finalize();

    for(int i = start; i - start < num && i < displacements.rows(); i++) {
        // Update the state
        auto q = mapDOFEigen(tets->getQ(), world);
        q = displacements.row(i);

        // Compute the energy
        energy_vec_per_pose.row(i) = tets->getStrainEnergyPerElement(world.getState());
    }
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

int main(int argc, char **argv) {
    fs::path displacements_path(argv[1]); // Takes in a path to a matrix containing an array of flattened displacements
    fs::path mesh_path(argv[2]); // Also takes in a path to the base mesh file.
    fs::path sim_params_path(argv[3]); // Also takes path to the simulation parameters file
    
    fs::path output_dir_path = displacements_path.parent_path();

    std::cout << "Loading simulation parameters from " << sim_params_path.string() << std::endl;
    std::ifstream fin(sim_params_path.string());
    fin >> sim_params;

    std::cout << "Loading MESH from " << mesh_path.string() << std::endl;
    igl::readMESH(mesh_path.string(), V, T, F);

    std::cout << "Loading displacements from " << displacements_path.string() << std::endl;
    MatrixXd displacements;    
    igl::readDMAT(displacements_path.string(), displacements);

    // Time to set up the Tets    
    NeohookeanTets tets(V,T);
    for(auto element: tets.getImpl().getElements()) {
        element->setDensity(sim_params["density"]);
        element->setParameters(sim_params["YM"], sim_params["Poisson"]);
    }

    // Reserve space for the matrices
    int n_tets = T.rows();
    int n_poses = displacements.rows();    
    MatrixXd energy_vec_per_pose(n_poses, n_tets);
    // MatrixXd force_vec_per_pose(n_poses, n_tets * 4); // Leave this out for now

    // Generate the data
    // TODO parallel?
    std::cout << "Generating energies" << std::endl;
    int n_chunks = 32;
    int num_per_chunk = displacements.rows() / n_chunks;
    int n_finished = 0;
    #pragma omp parallel for
    for(int i = 0; i < n_chunks + 1; i++) {
        generateEnergyForPoses(&tets, displacements, i * num_per_chunk, num_per_chunk, energy_vec_per_pose);
        progress_bar(++n_finished / (double) (n_chunks + 1));
    }
    std::cout << std::endl;

    // Save the data
    fs::path energy_path = output_dir_path / "energies.dmat";
    std::cout << "Saving energy to " << energy_path << std::endl;
    igl::writeDMAT(energy_path.string(), energy_vec_per_pose, false);

    return 0;
}