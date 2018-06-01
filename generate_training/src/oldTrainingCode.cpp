#include <vector>
#include <iostream>

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

#include <igl/writeDMAT.h>
#include <igl/writeOFF.h>
#include <igl/writeOBJ.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/boundary_facets.h>
#include <igl/material_colors.h>

#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>

#include <json.hpp>

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
typedef TimeStepperEulerImplicit<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>,

 AssemblerParallel<double, AssemblerEigenVector<double> >> MyTimeStepper;

// Mesh
Eigen::MatrixXd V; // Verts
Eigen::MatrixXi T; // Tet indices
Eigen::MatrixXi F; // Face indices

// Mouse/Viewer state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
bool is_dragging = false;
bool is_adding_old_mouse_forces = false;
int dragged_vert = 0;
int current_frame = 0;

// Parameters
bool saving_training_data = true;
std::string output_dir;// = "output/";
std::string input_dir;// = "input/";
std::string mesh_path;
Eigen::RowVector3d mesh_pos;

json read_parameters_json;
json read_mouse_input;

json mouse_json;
json parameters_json;

void save_displacements_DMAT_and_energy(int current_frame, MyWorld &world, NeohookeanTets *tets, json mouse_json) { // TODO: Add mouse position data to ouput
    auto q = mapDOFEigen(tets->getQ(), world);
    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
    Eigen::MatrixXd displacements = dV.transpose();

    std::stringstream displacements_filename;
    displacements_filename << output_dir << "displacements_" << current_frame << ".dmat";
    igl::writeDMAT(displacements_filename.str(), displacements, false); // Don't use ascii

    AssemblerEigenVector<double> internal_force; //maybe?
    getInternalForceVector(internal_force, *tets, world);

    std::stringstream force_filename;
    force_filename<<output_dir<<"internalForces_"<<current_frame<<".dmat";
    std::cout << (*internal_force).size() << " " << q.size() << std::endl;
    igl::writeDMAT(force_filename.str(), *internal_force, false);


    std::stringstream energy_filename;
    energy_filename << output_dir << "energy_" << current_frame << ".dmat";
    Eigen::MatrixXd energy_vec = tets->getStrainEnergyPerElement(world.getState());
    std::cout<<"here"<<std::endl;
    igl::writeDMAT(energy_filename.str(), energy_vec, false); // Don't use ascii
    // std::cout<<energy_vec<<std::endl;
    std::ofstream fout(output_dir + "mouse.json");
    fout << mouse_json;
    fout.close();

    std::cout << "Saved " << displacements_filename.str() << std::endl;
    std::cout << "Saved " << energy_filename.str() << std::endl;
    std::cout << "Saved " << force_filename.str() << std::endl;
}

void save_base_configurations_DMAT(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    std::stringstream verts_filename, faces_filename;
    verts_filename << output_dir << "base_verts.dmat";
    faces_filename << output_dir << "base_faces.dmat";

    igl::writeDMAT(verts_filename.str(), V, false); // Don't use ascii
    igl::writeDMAT(faces_filename.str(), F, false); // Don't use ascii

    //TODO: Check the other todo. Also save timestep and Density and YM, Poisson, fixDisplacementMin tolerance
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

int main(int argc, char **argv) {
    if(argc<7 || argc>9){
      std::cout<<"Run like this: ./prog -m mesh_file.mesh -o output_folder/ -i input_folder/"<<std::endl;
      exit(0);
    } else {
      for(int i=1; i<argc; ++i){
        if(std::string(argv[i])=="-m"){
          mesh_path= std::string(argv[i+1]);
        }else if(std::string(argv[i])=="-o"){
          output_dir = std::string(argv[i+1]);
        }else if(std::string(argv[i])=="-i"){
          input_dir= std::string(argv[i+1]);
        }else if(std::string(argv[i])=="-t"){
          saving_training_data = false;
        }else{
          std::cout<<"Crap, Try again."<<std::endl;
          exit(0);
        }
        i++;
      }
      if((output_dir.at(output_dir.length() -1)!='/') ||(input_dir.at(input_dir.length() -1)!='/')){
        std::cout<<"Try again: folders end in / "<<std::endl;
        exit(0);
      }
    }

    double m_Density;
    double m_Poisson;
    double m_YM;
    double p_Mass;
    double p_Stiffness;
    double p_RestLength;

    int solverIts;
    double timestep;
    int displacementAxis;
    double displacementTol;

    double input_paramfile_timestep = 0;

    try{
      std::ifstream input_mouse_json_file(input_dir+"mouse.json");
      input_mouse_json_file >> read_mouse_input;

      {
        json in_parameters_json;
        std::ifstream input_parameters_json_file(input_dir+"parameters.json");
        input_parameters_json_file >> in_parameters_json;
        input_paramfile_timestep = in_parameters_json["timestep"];
        input_parameters_json_file.close();
      }

    }
    catch(...){
      output_dir = input_dir;
      input_dir = std::string("");
      std::cout<<"Mouse Data WARNING: Input data not found in input dir. Writing to *input* file."<<std::endl;
      std::cout<<"Enter to continue..."<<std::endl;
      std::cin.get();
      // system("rm "+input_dir+"*.dmat")
    }

    //parameters for simulation that is being inputted
    //pretty much the only thing that should not be read into
    //GAUSS is time step (maybe iteration count too)
    try{
      std::cout<<output_dir<<std::endl;
      std::ifstream output_parameters_json_file(output_dir+"parameters.json");
      output_parameters_json_file >> read_parameters_json;

      m_Density = static_cast<double>(read_parameters_json["density"]);
      m_YM = static_cast<double>(read_parameters_json["YM"]);
      m_Poisson = static_cast<double>(read_parameters_json["Poisson"]);
      p_Mass = static_cast<double>(read_parameters_json["spring_point_Mass"]);
      p_Stiffness = static_cast<double>(read_parameters_json["spring_Stiffness"]);
      p_RestLength = static_cast<double>(read_parameters_json["spring_RestLength"]);
      displacementTol = static_cast<double>(read_parameters_json["displacement_tol"]);
      displacementAxis = static_cast<int>(read_parameters_json["displacement_axis"]);
      solverIts = static_cast<int>(read_parameters_json["solver_Iterations"]);
      timestep = static_cast<double>(read_parameters_json["timestep"]);

    }
    catch(...){
      std::cout<<"Parameters WARNING: Parameter file not found in output dir. Using *default* parameters."<<std::endl;
      std::cout<<"Enter to continue..."<<std::endl;
      std::cin.get();
      m_Density = 1000.0;
      m_YM = 1e6;
      m_Poisson = 0.45;
      p_Mass = 100000000;
      p_Stiffness = 150.0;
      p_RestLength = 0.1;
      displacementAxis = 1;
      displacementTol = 1e-2;
      solverIts = 1;
      timestep = .1;

      //Parameters for *this* current simulation
      std::ofstream jout(output_dir + "parameters.json");
      parameters_json["density"] = m_Density;
      parameters_json["YM"] = m_YM;
      parameters_json["Poisson"] = m_Poisson;
      parameters_json["spring_point_Mass"] = p_Mass;
      parameters_json["spring_Stiffness"] = p_Stiffness;
      parameters_json["spring_RestLength"] = p_RestLength;
      parameters_json["displacement_axis"] = displacementAxis;
      parameters_json["displacement_tol"] = displacementTol;
      parameters_json["timestep"] = timestep;
      parameters_json["solver_Iterations"] = solverIts;
      jout << parameters_json;
      jout.close();
      //----------------
    }

    std::cout<<"Neohookean FEM \n";

    igl::readMESH(mesh_path, V, T, F);

    igl::boundary_facets(T, F);

    //Setup Physics
    MyWorld world;

    NeohookeanTets *tets = new NeohookeanTets(V,T);
    int countElements = 0;
    for(auto element: tets->getImpl().getElements()) {
        element->setDensity(m_Density);//1000.0);
        element->setParameters(m_YM, m_Poisson);
        countElements += 1;
    }
    std::cout<<"Vertices: "<<tets->getImpl().getV().rows()<<std::endl;
    std::cout<<"Elements: "<<countElements<<std::endl;
    // exit(0);
    mouse_json["mouse_state_per_frame"] = json::array();
    // // Pinned particle to attach spring for dragging
    PhysicalSystemParticleSingle<double> *pinned_point = new PhysicalSystemParticleSingle<double>();
    pinned_point->getImpl().setMass(p_Mass); //10000000
    auto fem_attached_pos1 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos2 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos3 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos4 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos5 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos6 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos7 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos8 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos9 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());
    auto fem_attached_pos10 = PosFEM<double>(&tets->getQ()[0],0, &tets->getImpl().getV());

    std::vector<ForceSpringFEMParticle<double>*> springs;

    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos1, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos2, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos3, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos4, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos5, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos6, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos7, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos8, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos9, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));
    springs.push_back( new ForceSpringFEMParticle<double>(fem_attached_pos10, PosParticle<double>(&pinned_point->getQ()), p_RestLength, 0.0));




    world.addSystem(pinned_point);
    for(int s=0; s<springs.size(); s++)
    {
      world.addForce(springs[s]);
    }
    world.addSystem(tets);
    fixDisplacementMin(world, tets, displacementAxis, 0.015);

    int click_count =0;
    // Eigen::VectorXi movingVerts = minVertices(tets, displacementAxis);//indices for moving parts
    // std::cout<<movingVerts<<std::endl;
    std::vector<ConstraintFixedPoint<double> *> movingConstraints;
    // auto q = mapDOFEigen(tets->getQ(), world);
    // for(unsigned int ii=0; ii<movingVerts.rows(); ++ii) {
    //     movingConstraints.push_back(new ConstraintFixedPoint<double>(&tets->getQ()[movingVerts[ii]], Eigen::Vector3d(0,0,0)));
    //     //world.addConstraint(movingConstraints[ii]);
    // }

    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)

    reset_world(world);

    MyTimeStepper stepper(timestep, solverIts);
    stepper.step(world);

    if(saving_training_data) {
        save_base_configurations_DMAT(V, F);
    }

    // std::stringstream volume_file;
    // volume_file << output_dir << "energy_" << current_frame << ".dmat";
    // Eigen::MatrixXd vol_vec = tets->getStrainEnergyPerElement(world.getState());
    // std::cout<<"here"<<std::endl;
    // igl::writeDMAT(volume_file.str(), energy_vec, false); // Don't use ascii
    //
    // for(auto elem: tets->getImpl().getElements())
    // {
    //
    // }
    // exit(0);



    double findNewVert = true;
    int normIndex = 0;
    /** libigl display stuff **/
    igl::viewer::Viewer viewer;

    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
        // predefined colors
        const Eigen::RowVector3d orange(1.0,0.7,0.2);
        const Eigen::RowVector3d yellow(1.0,0.9,0.2);
        const Eigen::RowVector3d blue(0.2,0.3,0.8);
        const Eigen::RowVector3d green(0.2,0.6,0.3);
        const Eigen::RowVector3d sea_green(229./255.,211./255.,91./255.);

        if(is_dragging) {
            // Eigen::MatrixXd part_pos(1, 3);
            // part_pos(0,0) = dragged_pos[0]; // TODO why is eigen so confusing.. I just want to make a matrix from vec
            // part_pos(0,1) = dragged_pos[1];
            // part_pos(0,2) = dragged_pos[2];

            // viewer.data.set_points(part_pos, orange);

            Eigen::MatrixXi E(1,2);
            E(0,0) = 0;
            E(0,1) = 1;
            Eigen::MatrixXd P(2,3);
            P.row(0) = dragged_pos;
            P.row(1) = mesh_pos;

            viewer.data.set_edges(P, E, sea_green);
        } else {
            Eigen::MatrixXd part_pos = Eigen::MatrixXd::Zero(1,3);
            part_pos(0,0)=100000.0;
            // viewer.data.set_points(part_pos, sea_green);

            Eigen::MatrixXi E(1,2);
            E(0,0) = 0;
            E(0,1) = 1;
            Eigen::MatrixXd P(2,3);
            P.row(0) = Eigen::RowVector3d(1000.0,1000.0, 1000.0);
            P.row(1) = Eigen::RowVector3d(1000.0,1000.0, 1000.0);;
            viewer.data.set_edges(P, E, sea_green);
}


        if(viewer.core.is_animating)
        {
              if(!input_dir.empty())
              {

                  //inputting forces based on mouse drags from previously run simulation
                  json saved_mouse_state = read_mouse_input["mouse_state_per_frame"][(int) current_frame*timestep/input_paramfile_timestep];
                  if(saved_mouse_state == NULL){
                    std::cout<<"Successfully(?) completed the simulation."<<std::endl;
                    viewer.core.is_animating = false;
                    return false;
                  }
                  if(saved_mouse_state["is_dragging"])
                  {
                    auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
                    pinned_q = Eigen::Vector3d(saved_mouse_state["dragged_pos"][0], saved_mouse_state["dragged_pos"][1], saved_mouse_state["dragged_pos"][2] );//(dragged_pos).cast<double>(); // necessary?
                    std::cout<<"NEW: "<<pinned_q<<std::endl;
                    auto dragged_vertex_pos = Eigen::RowVector3d(saved_mouse_state["dragged_vert_pos"][0], saved_mouse_state["dragged_vert_pos"][1], saved_mouse_state["dragged_vert_pos"][2]);
                    std::cout<<"Dragging to: "<<dragged_vertex_pos<<std::endl;
                    std::cout<<saved_mouse_state["dragged_vert"]<<std::endl;


                    if(findNewVert){

                        double minSqNorm = (dragged_vertex_pos - tets->getImpl().getV().row(0)).norm();
                        normIndex = 0;
                        for(unsigned int pvi = 1; pvi<tets->getImpl().getV().rows(); ++pvi)
                        {
                          if((dragged_vertex_pos - tets->getImpl().getV().row(pvi)).norm() < minSqNorm){
                            minSqNorm = (dragged_vertex_pos - tets->getImpl().getV().row(pvi)).norm();
                            normIndex = pvi;
                          }
                        }

                        int s = 0;
                        double tol = 0.20;
                        //Use a Q, not a double for loop
                        for(int v=0; v<V.rows(); v++)
                        {   double dist = (V.row(normIndex) - V.row(v)).norm();
                            if(dist<tol)
                            {
                              if(s>springs.size()-1){
                                break;
                              }
                              std::cout<<"s "<<s<<std::endl;
                              springs[s]->getImpl().setStiffness(p_Stiffness/springs.size());
                              auto mesh_attached = PosFEM<double>(&tets->getQ()[v],v, &tets->getImpl().getV());
                              springs[s]->getImpl().setPosition0(mesh_attached);
                              s++;
                            }
                        }

                      std::cout<<"Saved Mouse State" <<saved_mouse_state<<std::endl;
                      std::cout<<"Selected vert for dragging "<<tets->getImpl().getV().row(normIndex)<<std::endl;
                    }

                    findNewVert = false;
                  }
                  else{
                    findNewVert = true;
                    for(int s = 0; s<springs.size(); s++)
                    {
                      springs[s]->getImpl().setStiffness(0.0);
                    }
                  }

              }
              // std::cout<<"input dir empty "<<input_dir.empty()<<std::endl;
              // Save Current configuration every 0.5 seconds
              if(current_frame%1 == 0 || input_dir.empty()){

                if(saving_training_data) {
                  json mouse_state;
                  mouse_state["dragged_pos"] = {dragged_pos[0], dragged_pos[1], dragged_pos[2]};
                  mouse_state["dragged_vert"] = dragged_vert;
                  mouse_state["dragged_vert_pos"] = {tets->getImpl().getV().row(dragged_vert)[0], tets->getImpl().getV().row(dragged_vert)[1], tets->getImpl().getV().row(dragged_vert)[2]};
                  mouse_state["is_dragging"] = is_dragging;
                  mouse_json["mouse_state_per_frame"].push_back(mouse_state);
                  // if(mouse_json["potential_energy_per_frame"].size() == current_frame) {
                  //     std::cout << "Index mismatch!" << std::endl;
                  // }
                    std::cout<<"Time of Print: "<<current_frame*timestep<<std::endl;
                    save_displacements_DMAT_and_energy(current_frame, world, tets, mouse_json);
                    std::stringstream off_file;
                    off_file<< output_dir<<current_frame<<".obj";
                    igl::writeOBJ(off_file.str(), viewer.data.V, viewer.data.F);
                }
              }


              stepper.step(world);
              Eigen::MatrixXd newV = getCurrentVertPositions(world, tets);
              mesh_pos = newV.row(dragged_vert);
              if(current_frame%1==0){
                viewer.data.set_vertices(newV);
                viewer.data.compute_normals();
              }
              current_frame++;
              std::cout<<"Time Elapsed: "<<current_frame*timestep<<std::endl;
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
                if(input_dir.empty())
                  reset_world(world);
                break;
            }
            default:
            return false;
        }
        return true;
    };

    viewer.callback_mouse_down = [&](igl::viewer::Viewer&, int mb, int)->bool
    {
        last_mouse = Eigen::RowVector3f(viewer.current_mouse_x,viewer.core.viewport(3)-viewer.current_mouse_y,0);

        std::cout<<"Mouse down at: "<<last_mouse<<std::endl;
        if(!input_dir.empty())
        {
          return false;
        }
        Eigen::MatrixXd curV = getCurrentVertPositions(world, tets);
        // Find closest point on mesh to mouse position
        int fid;
        Eigen::Vector3f bary;
        // std::cout<<"Here"<<std::endl;
        if(igl::unproject_onto_mesh(
          last_mouse.head(2),
          viewer.core.view * viewer.core.model,
          viewer.core.proj,
          viewer.core.viewport,
          curV, F,
          fid, bary))
          {
              //Left mouse click

              long c;
              bary.maxCoeff(&c);
              dragged_pos = curV.row(F(fid,c)); //Epsilon offset so we don't div by 0
              dragged_vert = F(fid,c);
              std::cout << "clicked vert: " << dragged_vert << std::endl;

              if(mb==0){
                  // Update the system
                  //This is some weird ass roundabout way to set the new
                  //pointer end of the spring
                  auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
                  pinned_q = dragged_pos;//(dragged_pos).cast<double>(); // necessary?
                  std::cout<<"Pinned q "<<pinned_q<<std::endl;

                  if(is_dragging==false){
                    int s = 0;
                    double tol = 0.20;
                    //Use a Q, not a double for loop
                    for(int v=0; v<V.rows(); v++)
                    {   double dist = (V.row(dragged_vert) - V.row(v)).norm();
                        if(dist<tol)
                        {
                          if(s>springs.size()-1){
                            break;
                          }
                          std::cout<<"s "<<s<<std::endl;
                          springs[s]->getImpl().setStiffness(p_Stiffness/springs.size());
                          auto mesh_attached = PosFEM<double>(&tets->getQ()[v],v, &tets->getImpl().getV());
                          springs[s]->getImpl().setPosition0(mesh_attached);
                          s++;
                        }
                    }

                  // std::cout<<"fem attached pos " << dragged_pos<<", "<<tets->getImpl().getV().row(dragged_vert)<<std::endl;
                  // std::cout<<"dragged vertpos "<<tets->getImpl().getV().row(dragged_vert)<<std::endl;

                }
                is_dragging = true;
              }


              if(mb==2)
              {
                click_count +=1;
                for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
                    double r = (dragged_pos - movingConstraints[jj]->getImpl().getFixedPoint().transpose()).norm();
                    std::cout<<r<<std::endl;
                    movingConstraints[jj]->getImpl().setFixedPoint(movingConstraints[jj]->getImpl().getFixedPoint() + Eigen::Vector3d(r*std::sin(0.3),0,r*std::cos(0.3)));

                }
                  // long c;
                  // bary.maxCoeff(&c);
                  // auto fix_pos = curV.row(F(fid,c)); //Epsilon offset so we don't div by 0
                  // auto fix_vert = F(fid,c);
                  // std::cout << "Fixing vert: " << fix_vert <<"@"<<fix_pos<<std::endl;
                  //
                  // // Update the system
                  // world.addConstraint(new ConstraintFixedPoint<double>(&tets->getQ()[fix_vert], Eigen::Vector3d(0,0,0)));
                  // world.updateInequalityConstraints();
              }

            return true;
        }

        return false; // TODO false vs true??
    };

    viewer.callback_mouse_up = [&](igl::viewer::Viewer&, int, int)->bool
    {
        if(input_dir.empty())
        {
          is_dragging = false;
          for(int s = 0; s<springs.size(); s++)
          {
            springs[s]->getImpl().setStiffness(0.0);
          }
          // auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
          // pinned_q = Eigen::RowVector3d(-10000.0,0.0,0.0); // Move the point out of view
        }
        return false;
    };

    viewer.callback_mouse_move = [&](igl::viewer::Viewer &, int,int)->bool
    {
        if(is_dragging && input_dir.empty()) {
            Eigen::RowVector3f drag_mouse(viewer.current_mouse_x, viewer.core.viewport(3) - viewer.current_mouse_y, last_mouse(2));

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
    viewer.data.face_based = false;
    viewer.core.line_width = 2;

    viewer.core.background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
    viewer.core.shininess = 120.0;
    viewer.data.set_colors(Eigen::RowVector3d(igl::CYAN_DIFFUSE[0], igl::CYAN_DIFFUSE[1], igl::CYAN_DIFFUSE[2]));
    // viewer.data.uniform_colors(
    //   Eigen::Vector3d(igl::CYAN_AMBIENT[0], igl::CYAN_AMBIENT[1], igl::CYAN_AMBIENT[2]),
    //   Eigen::Vector3d(igl::CYAN_DIFFUSE[0], igl::CYAN_DIFFUSE[1], igl::CYAN_DIFFUSE[2]),
    //   Eigen::Vector3d(igl::CYAN_SPECULAR[0], igl::CYAN_SPECULAR[1], igl::CYAN_SPECULAR[2]));

    // viewer.launch();

    viewer.launch_init(true, false);
      viewer.opengl.shader_mesh.free();

  {
    std::string mesh_vertex_shader_string =
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
void main()
{
  position_eye = vec3 (view * model * vec4 (position, 1.0));
  normal_eye = vec3 (view * model * vec4 (normal, 0.0));
  normal_eye = normalize(normal_eye);
  gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * model * vec4(position, 1.0);
  Kai = Ka;
  Kdi = Kd;
  Ksi = Ks;
  texcoordi = texcoord;
})";

    std::string mesh_fragment_shader_string =
R"(#version 150
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 fixed_color;
in vec3 position_eye;
in vec3 normal_eye;
uniform vec3 light_position_world;
vec3 Ls = vec3 (1, 1, 1);
vec3 Ld = vec3 (1, 1, 1);
vec3 La = vec3 (1, 1, 1);
in vec4 Ksi;
in vec4 Kdi;
in vec4 Kai;
in vec2 texcoordi;
uniform sampler2D tex;
uniform float specular_exponent;
uniform float lighting_factor;
uniform float texture_factor;
out vec4 outColor;
void main()
{
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
outColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;
if (fixed_color != vec4(0.0)) outColor = fixed_color;
})";

  viewer.opengl.shader_mesh.init(
      mesh_vertex_shader_string,
      mesh_fragment_shader_string,
      "outColor");
  }


  viewer.launch_rendering(true);
  viewer.launch_shut();


    return EXIT_SUCCESS;
}