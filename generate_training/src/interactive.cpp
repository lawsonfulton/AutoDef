#include <vector>
#include <boost/filesystem.hpp>
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
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>

#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>

#include <json.hpp>

namespace fs = boost::filesystem;
using json = nlohmann::json;
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

typedef AssemblerParallel<double, AssemblerEigenVector<double> > MyAssembler;
typedef AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > MyAssemblerSparse;

typedef TimeStepperEulerImplicit<double, MyAssemblerSparse, MyAssembler > ImplicitStepper;
typedef TimeStepperEulerImplicitLinear<double, MyAssemblerSparse, MyAssembler > LinearImplicitStepper;

// Mesh
Eigen::MatrixXd V; // Verts
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices

std::vector<ConstraintFixedPoint<double> *> movingConstraints;
Eigen::VectorXi movingVerts;

// Mouse/Viewer state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
bool is_dragging = false;
int dragged_vert = 0;
int current_frame = 0;
int starting_frame_num = 0;

// Parameters
bool saving_training_data = true;
json mouse_json;

std::string ZeroPadNumber(int num)
{
    std::ostringstream ss;
    ss << std::setw( 7 ) << std::setfill( '0' ) << num;
    return ss.str();
}

void save_displacements_DMAT_and_energy(int current_frame, MyWorld &world, NeohookeanTets *tets, json mouse_json, fs::path output_dir) { // TODO: Add mouse position data to ouput
    std::string frame_num_string = std::to_string(current_frame + starting_frame_num);
    auto q = mapDOFEigen(tets->getQ(), world);
    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
    Eigen::MatrixXd displacements = dV.transpose();

    fs::path  displacements_file = output_dir / ("displacements_" + frame_num_string + ".dmat");
    igl::writeDMAT(displacements_file.string(), displacements, false); // Don't use ascii

    MyAssembler internal_force; //maybe?
    getInternalForceVector(internal_force, *tets, world);
    fs::path force_file = output_dir / ("internalForces_" + frame_num_string + ".dmat");
    std::cout << (*internal_force).size() << " " << q.size() << std::endl;
    igl::writeDMAT(force_file.string(), *internal_force, false);


    fs::path energy_file = output_dir / ("energy_" + frame_num_string + ".dmat");
    Eigen::MatrixXd energy_vec = tets->getStrainEnergyPerElement(world.getState());
    igl::writeDMAT(energy_file.string(), energy_vec, false); // Don't use ascii

    std::ofstream fout((output_dir / "mouse.json").string());
    fout << mouse_json;
    fout.close();

    std::cout << "Saved " << displacements_file.string() << std::endl;
    std::cout << "Saved " << energy_file.string() << std::endl;
    std::cout << "Saved " << force_file.string() << std::endl;
}

void save_base_configurations_DMAT(Eigen::MatrixXd &V, Eigen::MatrixXi &F, fs::path output_dir) {
    igl::writeDMAT((output_dir / "base_verts.dmat").string(), V, false); // Don't use ascii
    igl::writeDMAT((output_dir / "base_faces.dmat").string(), F, false); // Don't use ascii
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
    fs::create_directory(dir);
}



void update_twist_constraints(MyWorld &world, double timestep) {
    double rot_t = -2.0 * current_frame * timestep * 0.0; 
    double rot_t_2 = -1.0 * current_frame * timestep * 0.0; 
    double offset_t = -0.0001 * current_frame * timestep * 0.0;
    
    Eigen::AngleAxis<double> rot(rot_t, Eigen::Vector3d(1.0,0.0,0.0));
    Eigen::AngleAxis<double> rot2(-rot_t_2 * 0.3, Eigen::Vector3d(0.0,1.0,1.0));
    Eigen::Vector3d offset = offset_t * Eigen::Vector3d(-0.7,-0.5,1.5);

    for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
        Eigen::Vector3d current_u = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
        Eigen::Vector3d v = V.row(movingVerts[jj]);
        Eigen::Vector3d new_v = rot2 * rot * v + offset;

        Eigen::Vector3d delta_u = (new_v - v) - current_u;
        movingConstraints[jj]->getImpl().setFixedPoint(new_v, delta_u/timestep);
    }
}



int main(int argc, char **argv) {
    MyWorld world;

    // --- Read in config
    fs::path config_path(argv[1]);
    fs::path config_dir(config_path.parent_path());
    std::ifstream fin(config_path.string());
    json config;
    fin >> config;

    std::string output_dir_string = config["output_dir"];
    std::string mesh_path_string = config["mesh_path"];
    fs::path output_dir = config_dir / output_dir_string;
    fs::path mesh_path = config_dir / mesh_path_string;

    create_or_replace_dir(output_dir);
    fs::copy_file(config_path, output_dir/ "parameters.json" );
    fs::copy_file(mesh_path, output_dir/ "tets.mesh" );
    // ----


    // --- Config vars
    bool do_twist = config["do_twist"];
    bool full_implicit = config["full_implicit"];
    int fixed_axis = config["fixed_axis"];
    bool flip_fixed_axis = config["flip_fixed_axis"];
    int its = config["implicit_its"];
    int max_frames = config["max_frames"];
    starting_frame_num = config["starting_frame"];
    double timestep = config["time_step"];
    double vert_select_eps = 1e-4;
    saving_training_data = config["save_training_data"];
    // ---


    // --- Setup tet mesh
    igl::readMESH(mesh_path.string(), V, T, F);
    NeohookeanTets *tets = new NeohookeanTets(V,T);
    for(auto element: tets->getImpl().getElements()) {
        element->setDensity(config["density"]);//1000.0);
        element->setParameters(config["YM"], config["Poisson"]);
    }
    world.addSystem(tets);
    if(flip_fixed_axis) {
        fixDisplacementMax(world, tets, fixed_axis, vert_select_eps);
    } else {
        fixDisplacementMin(world, tets, fixed_axis, vert_select_eps);
    }

    if(do_twist) {
        movingVerts = maxVertices(tets, fixed_axis); //indices for moving parts
        for(unsigned int ii=0; ii<movingVerts.rows(); ++ii) {
            movingConstraints.push_back(new ConstraintFixedPoint<double>(&tets->getQ()[movingVerts[ii]], Eigen::Vector3d(0,0,0)));
            world.addConstraint(movingConstraints[ii]);
        }
    }
    // ---


    // --- Set up mouse interaction
    PhysicalSystemParticleSingle<double> *pinned_point = new PhysicalSystemParticleSingle<double>();
    pinned_point->getImpl().setMass(100000000); //10000000
    auto fem_attached_pos = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    double spring_stiffness = config["spring_strength"];
    double spring_rest_length = 0.0001;
    ForceSpringFEMParticle<double> *forceSpring = new ForceSpringFEMParticle<double>(fem_attached_pos, // TODO compare getV to V. Get rid of double use of index
                                                                                     PosParticle<double>(&pinned_point->getQ()),
                                                                                     spring_rest_length, 0.0);
    world.addSystem(pinned_point);
    world.addForce(forceSpring);
    // --- 


    // --- Finalize world and time stepper
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    reset_world(world);

    ImplicitStepper implicit_stepper(timestep, its);
    LinearImplicitStepper linear_stepper(timestep, its);
    // --- 


    // This is where we update state
    auto do_timestep = [&] () {
        if(do_twist) {
            update_twist_constraints(world, timestep);
        }
        
        if(full_implicit) {
            implicit_stepper.step(world);
        } else {
            linear_stepper.step(world);
        }

        current_frame++;
        if(max_frames != 0 and current_frame >= max_frames) {
            exit(0);
        }
    };


    if(saving_training_data) {
        save_base_configurations_DMAT(V, F, output_dir.string());
    }

    mouse_json["mouse_state_per_frame"] = json::array();

    /** libigl display stuff **/
    igl::viewer::Viewer viewer;
    double tot_time = 0.0;
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

            viewer.data.set_points(part_pos.transpose(), orange);
        } else {
            Eigen::MatrixXd part_pos;
            viewer.data.set_points(part_pos, orange);
        }

        if(viewer.core.is_animating)
        {
            // Save Current configuration

            json mouse_state;
            mouse_state["dragged_pos"] = {dragged_pos[0], dragged_pos[1], dragged_pos[2]};
            mouse_state["dragged_vert"] = dragged_vert;
            mouse_state["is_dragging"] = is_dragging;
            mouse_json["mouse_state_per_frame"].push_back(mouse_state);
            // if(mouse_json["potential_energy_per_frame"].size() == current_frame) {
            //     std::cout << "Index mismatch!" << std::endl;
            // }
            if(saving_training_data) {
                save_displacements_DMAT_and_energy(current_frame, world, tets, mouse_json, output_dir.string());
            }
            double start = igl::get_seconds();
            do_timestep();
            tot_time += igl::get_seconds() - start;



            Eigen::MatrixXd newV = getCurrentVertPositions(world, tets);
            // std::cout<< newV.block(0,0,10,3) << std::endl;
            viewer.data.set_vertices(newV);
            viewer.data.compute_normals();
            std::cout << "Average timestep: " << (tot_time / current_frame) << std::endl;
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
            forceSpring->getImpl().setPosition0(fem_attached_pos);

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