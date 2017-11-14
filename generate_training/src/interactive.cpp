#include <functional>

//#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>

//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>

#include <igl/writePLY.h>
#include <igl/viewer/Viewer.h>

#include <iostream>
#include <string>
#include <sstream>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

/* Tetrahedral finite elements */

//typedef physical entities I need

//typedef scene
typedef PhysicalSystemFEM<double, NeohookeanTet> FEMLinearTets;

typedef World<double, std::tuple<FEMLinearTets *>,
                        std::tuple<ForceSpring<double> *>,
                        std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
typedef TimeStepperEulerImplictLinear<double, AssemblerEigenSparseMatrix<double>,
AssemblerEigenVector<double> > MyTimeStepper;

//typedef Scene<MyWorld, MyTimeStepper> MyScene;

int frame = 0;
//new code -- load tetgen files
Eigen::MatrixXd V;
Eigen::MatrixXi F;

bool saveFrames = false;
std::string outputDir = "frames/";
void preStepCallback(MyWorld &world) {
    if(saveFrames) {
        Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world); // Get displacements only

        Eigen::MatrixXd displacements = q;
        
        std::stringstream filename;
        filename << outputDir << "displacements_" << frame << ".ply";
        Eigen::MatrixXi no_faces;
        igl::writePLY(filename.str(), displacements, no_faces, false);
        frame++;
    }
}

void saveBase(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    std::stringstream filename;
    filename << outputDir << "base_mesh.ply";
    igl::writePLY(filename.str(), V, F, false);
}

int main(int argc, char **argv) {
    std::cout<<"Test Neohookean FEM \n";
    
    //Setup Physics
    MyWorld world;
    
    readTetgen(V, F, dataDir()+"/meshesTetgen/Beam/Beam.node", dataDir()+"/meshesTetgen/Beam/Beam.ele");


    saveBase(V,F);


    FEMLinearTets *test = new FEMLinearTets(V,F);
    
    world.addSystem(test);
    fixDisplacementMin(world, test);
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    auto q = mapStateEigen(world);
    q.setZero();
    
    MyTimeStepper stepper(0.01);
    
    // //Display
    // QGuiApplication app(argc, argv);`
    
    // MyScene *scene = new MyScene(&world, &stepper, preStepCallback);
    // GAUSSVIEW(scene);
    
    // return app.exec();
    igl::viewer::Viewer viewer;
    const auto & update = [&]()
    {
        // // predefined colors
        // const Eigen::RowVector3d orange(1.0,0.7,0.2);
        // const Eigen::RowVector3d yellow(1.0,0.9,0.2);
        // const Eigen::RowVector3d blue(0.2,0.3,0.8);
        // const Eigen::RowVector3d green(0.2,0.6,0.3);
        // if(s.placing_handles)
        // {
        //     viewer.data.set_vertices(V);
        //     viewer.data.set_colors(blue);
        //     viewer.data.set_points(s.CV,orange);
        // }else
        // {
        //     // SOLVE FOR DEFORMATION
        //     switch(method)
        //     {
	       //      default:
	       //      case BIHARMONIC:
	       //      {
	       //          Eigen::MatrixXd D;
	       //          biharmonic_solve(biharmonic_data,s.CU-s.CV,D);
	       //          U = V+D;
	       //          break;
	       //      }
	       //      case ARAP:
	       //      {
	       //          arap_single_iteration(arap_data,arap_K,s.CU,U);
	       //          break;
        //     	}
        //     }
        //     viewer.data.set_vertices(U);
        //     viewer.data.set_colors(method==BIHARMONIC?orange:yellow);
        //     viewer.data.set_points(s.CU,method==BIHARMONIC?blue:green);
        // }
        // viewer.data.compute_normals();
    };

    viewer.data.set_mesh(V,F);
    viewer.core.show_lines = false;
    viewer.core.is_animating = true;
    viewer.data.face_based = true;
    update();
    viewer.launch();
    return EXIT_SUCCESS;
}

