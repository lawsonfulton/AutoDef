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
        stepper.step(world);

        Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world); // Get displacements only

        Eigen::MatrixXd newV = V + q; // TODO this isn't working

        viewer.data.set_vertices(newV);
   
        // viewer.data.compute_normals();
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
            default:
            return false;
        }
        update();
        return true;
    };

    viewer.data.set_mesh(V,F);
    viewer.core.show_lines = false;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;
    update();
    viewer.launch();
    return EXIT_SUCCESS;
}

