#include <vector>

//#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>

// #include <igl/writePLY.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>

#include <iostream>
#include <string>
#include <sstream>

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

// Mouse state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
bool is_dragging = false;
int dragged_vert = 0;

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

int main(int argc, char **argv) {
    std::cout<<"Test Neohookean FEM \n";
    
    //Setup Physics
    MyWorld world;
    
    igl::readMESH("../mesh/Beam.mesh", V, T, F);
    NeohookeanTets *tets = new NeohookeanTets(V,T);

    // // Pinned particle to attach spring for dragging
    PhysicalSystemParticleSingle<double> *pinned_point = new PhysicalSystemParticleSingle<double>();
    pinned_point->getImpl().setMass(10000000);
    auto fem_attached_pos = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());

    double spring_stiffness = 400.0;
    double spring_rest_length = 0.1;
    ForceSpringFEMParticle<double> *forceSpring = new ForceSpringFEMParticle<double>(fem_attached_pos, // TODO compare getV to V. Get rid of double use of index
                                                                                     PosParticle<double>(&pinned_point->getQ()),
                                                                                     spring_rest_length, 0.0);

    world.addSystem(pinned_point);
    world.addForce(forceSpring);
    world.addSystem(tets);
    fixDisplacementMin(world, tets);
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    reset_world(world);
    
    MyTimeStepper stepper(0.01);
    stepper.step(world);
    

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
            auto q_part = mapDOFEigen(pinned_point->getQ(), world);
            Eigen::MatrixXd part_pos = q_part;
            // std::cout << "q_part " << q_part << std::endl;
            // std::cout << "part_pos " << part_pos << std::endl;
            viewer.data.set_points(part_pos.transpose(), orange);
        } else {
            Eigen::MatrixXd part_pos;
            viewer.data.set_points(part_pos, orange);
        }

        if(viewer.core.is_animating)
        {
            stepper.step(world);

            Eigen::MatrixXd newV = getCurrentVertPositions(world, tets); 
            // std::cout<< newV.block(0,0,10,3) << std::endl;
            viewer.data.set_vertices(newV);
            viewer.data.compute_normals();
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
        std::cout << last_mouse << std::endl;
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
            std::cout << "Dragging vert: " << dragged_vert << std::endl;

            // Update the system
            is_dragging = true;
            forceSpring->getImpl().setStiffness(spring_stiffness);
            auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
            pinned_q = dragged_pos;//(dragged_pos).cast<double>(); // necessary?

            fem_attached_pos = PosFEM<double>(&tets->getQ()[dragged_vert],dragged_vert, &tets->getImpl().getV());

            return true;
        }
        
        return false; // TODO false vs true??
    };

    viewer.callback_mouse_up = [&](igl::viewer::Viewer&, int, int)->bool
    {
        is_dragging = false;
        forceSpring->getImpl().setStiffness(0.0);
        // auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
        // pinned_q = Eigen::RowVector3d(-10000.0,0.0,0.0); // Move the point out of view

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
            // dragged_pos += (drag_mouse-last_mouse).cast<double>();
            last_mouse = drag_mouse;

            // Update the system
            // TODO dedupe this
            auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
            pinned_q = dragged_pos; //(dragged_pos).cast<double>(); // necessary?
        }

        return false;
    };

    viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;

    viewer.launch();
    return EXIT_SUCCESS;
}

// bool saveFrames = false;
// std::string outputDir = "frames/";
// int frame = 0;
// void preStepCallback(MyWorld &world) {
//     // if(saveFrames) {
//     //     Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world); // Get displacements only

//     //     Eigen::MatrixXd displacements = q;
        
//     //     std::stringstream filename;
//     //     filename << outputDir << "displacements_" << frame << ".ply";
//     //     Eigen::MatrixXi no_faces;
//     //     igl::writePLY(filename.str(), displacements, no_faces, false);
//     //     frame++;
//     // }
// }

// // void saveBase(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
// //     std::stringstream filename;
// //     filename << outputDir << "base_mesh.ply";
// //     igl::writePLY(filename.str(), V, F, false);
// }