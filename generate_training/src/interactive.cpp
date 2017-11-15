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
                        std::tuple<ForceSpring<double> *>,
                        std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
typedef TimeStepperEulerImplictLinear<double, AssemblerEigenSparseMatrix<double>,
AssemblerEigenVector<double> > MyTimeStepper;




Eigen::MatrixXd V; // Verts
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices

bool saveFrames = false;
std::string outputDir = "frames/";
int frame = 0;
void preStepCallback(MyWorld &world) {
    // if(saveFrames) {
    //     Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world); // Get displacements only

    //     Eigen::MatrixXd displacements = q;
        
    //     std::stringstream filename;
    //     filename << outputDir << "displacements_" << frame << ".ply";
    //     Eigen::MatrixXi no_faces;
    //     igl::writePLY(filename.str(), displacements, no_faces, false);
    //     frame++;
    // }
}

// void saveBase(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
//     std::stringstream filename;
//     filename << outputDir << "base_mesh.ply";
//     igl::writePLY(filename.str(), V, F, false);
// }

// Todo put this in utilities
Eigen::MatrixXd getCurrentVertPositions(MyWorld &world, NeohookeanTets *tets) {
    // Eigen::Map<Eigen::MatrixXd> q(mapStateEigen<0>(world).data(), V.cols(), V.rows()); // Get displacements only
    auto q = mapDOFEigen(tets->getQ(), world);
    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only

    return V + dV.transpose(); 
}


int main(int argc, char **argv) {
    std::cout<<"Test Neohookean FEM \n";
    
    //Setup Physics
    MyWorld world;
    
    igl::readMESH("../mesh/Beam.mesh", V, T, F);
    NeohookeanTets *tets = new NeohookeanTets(V,T);
    
    world.addSystem(tets);
    fixDisplacementMin(world, tets);

    // Pinned particle to attach spring for dragging
    PhysicalSystemParticleSingle<double> *pinned_point = new PhysicalSystemParticleSingle<double>();
    // TODO make a set position helper
    Eigen::Vector3d pinned_pos(5.0,5.0,5.0); 

    ForceSpring<double> *forceSpring = new ForceSpring<double>(&pinned_point->getQ(),  &tets->getQ()[100], 1, 8.0);

    // TODO make this work
    world.addSystem(pinned_point);
    world.addForce(forceSpring);

    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    

    auto reset_world = [&]() {
        auto q = mapStateEigen(world); // TODO is this necessary?
        q.setZero();

        auto q_part = mapDOFEigen(pinned_point->getQ(), world);
        q_part = pinned_pos;
    };

    reset_world();
    
    MyTimeStepper stepper(0.01);
    stepper.step(world);
    


    /** libigl display stuff **/
    std::vector<int> dragged_verts;
    igl::viewer::Viewer viewer;

    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
        // predefined colors
        const Eigen::RowVector3d orange(1.0,0.7,0.2);
        const Eigen::RowVector3d yellow(1.0,0.9,0.2);
        const Eigen::RowVector3d blue(0.2,0.3,0.8);
        const Eigen::RowVector3d green(0.2,0.6,0.3);

        auto q_part = mapDOFEigen(pinned_point->getQ(), world);
        Eigen::MatrixXd part_pos = q_part;
        std::cout << "q_part " << q_part << std::endl;
        std::cout << "part_pos " << part_pos << std::endl;
        viewer.data.set_points(part_pos.transpose(), orange);

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
                reset_world();
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
        Eigen::RowVector3f last_mouse = Eigen::RowVector3f(viewer.current_mouse_x,viewer.core.viewport(3)-viewer.current_mouse_y,0);

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
            Eigen::RowVector3d new_c = curV.row(F(fid,c));
            if(dragged_verts.size() != 0) {
                dragged_verts.pop_back();
            }
            dragged_verts.push_back(F(fid,c));
            std::cout << dragged_verts[0] << std::endl;
        }
        
        return false;
    };

    // viewer.callback_mouse_move = [&](igl::viewer::Viewer &, int,int)->bool
    // {
    // if(sel!=-1)
    // {
    //     Eigen::RowVector3f drag_mouse(
    //     viewer.current_mouse_x,
    //     viewer.core.viewport(3) - viewer.current_mouse_y,
    //     last_mouse(2));
    //     Eigen::RowVector3f drag_scene,last_scene;
    //     igl::unproject(
    //     drag_mouse,
    //     viewer.core.view*viewer.core.model,
    //     viewer.core.proj,
    //     viewer.core.viewport,
    //     drag_scene);
    //     igl::unproject(
    //     last_mouse,
    //     viewer.core.view*viewer.core.model,
    //     viewer.core.proj,
    //     viewer.core.viewport,
    //     last_scene);
    //     s.CU.row(sel) += (drag_scene-last_scene).cast<double>();
    //     last_mouse = drag_mouse;
    //     update();
    //     return true;
    // }
    // return false;
    // };
    // viewer.callback_mouse_up = [&](igl::viewer::Viewer&, int, int)->bool
    // {
    // sel = -1;
    // return false;
    // };

    viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;

    viewer.launch();
    return EXIT_SUCCESS;
}

