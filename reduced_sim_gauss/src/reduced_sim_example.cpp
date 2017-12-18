#include <vector>

//#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <ForceSpring.h>
#include <FEMIncludes.h>
#include <PhysicalSystemParticles.h>
//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>

#include <igl/writeDMAT.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>

#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>

// Optimization
#include <LBFGS.h>

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

// Mouse/Viewer state
Eigen::RowVector3f last_mouse;
Eigen::RowVector3d dragged_pos;
bool is_dragging = false;
int dragged_vert = 0;
int current_frame = 0;

// Parameters
bool saving_training_data = false;
std::string output_dir = "output_data/";


void save_displacements_DMAT(const std::string path, MyWorld &world, NeohookeanTets *tets) { // TODO: Add mouse position data to ouput
    auto q = mapDOFEigen(tets->getQ(), world);
    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only
    Eigen::MatrixXd displacements = dV.transpose();

    igl::writeDMAT(path, displacements, false); // Don't use ascii
}

void save_base_configurations_DMAT(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    std::stringstream verts_filename, faces_filename;
    verts_filename << output_dir << "base_verts.dmat";
    faces_filename << output_dir << "base_faces.dmat";
    
    igl::writeDMAT(verts_filename.str(), V, false); // Don't use ascii
    igl::writeDMAT(faces_filename.str(), F, false); // Don't use ascii
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
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using namespace LBFGSpp;


// void test_opt(MyWorld &world, NeohookeanTets *tets) {

//     // AssemblerEigenVector<double> forces; //maybe?
//     // getForceVector(forces, *world);

//     //test_gradient(world, tets);


//     const int n = 10;
//     // Set up parameters
//     LBFGSParam<double> param;
//     param.epsilon = 1e-3;
//     param.max_iterations = 100;

//     // Create solver and function object
//     LBFGSSolver<double> solver(param);

//     auto q = mapDOFEigen(tets->getQ(), world);

//     GPLC fun(q, q, 0.001, &world, tets);

//     // Initial guess
//     VectorXd x = q;//VectorXd::Zero(n);
//     // x will be overwritten to be the best point found
//     double fx;
//     int niter = solver.minimize(fun, x, fx);

//     std::cout << niter << " iterations" << std::endl;
//     std::cout << "x = \n" << x.transpose() << std::endl;
//     std::cout << "f(x) = " << fx << std::endl;

//     return;
// }

class GPLC
{
private:
    VectorXd m_cur_Pq;
    VectorXd m_prev_Pq;
    VectorXd F_ext;
    SparseMatrix<double> M; // mass matrix
    SparseMatrix<double> P;
    //SparseMatrix<double> P; // Constraint projection n x m, n=total dof m=unconstrained dof
    // TODO should I use a sparse matrix for P? M? Any special considerations?

    double h;

    MyWorld *world;
    NeohookeanTets *tets;
public:
    GPLC(VectorXd cur_Pq,
        VectorXd prev_Pq,
        double h,
        MyWorld *world,
        NeohookeanTets *tets,
        SparseMatrix<double> &constraint_matrix_P) : m_cur_Pq(cur_Pq), m_prev_Pq(prev_Pq), h(h), world(world), tets(tets), P(constraint_matrix_P)
    {

        std::cout << "test6" << std::endl;
        // Construct mass matrix and external forces
        AssemblerEigenSparseMatrix<double> M_asm;
        getMassMatrix(M_asm, *world);
        std::cout << "test7" << std::endl;
        // tets->getMassMatrix(M_asm, world->getState()); // TODO FIX THIS WITH THE PROPER INIT
        M = P * *M_asm * P.transpose();
        std::cout << "test8" << std::endl;
        VectorXd g(cur_Pq.size());
        for(int i=0; i < g.size(); i += 3) {
            g[i] = 0.0;
            g[i+1] = -9.8;
            g[i+2] = 0.0;
        }
        std::cout << "test9" << std::endl;

        F_ext = M * g;
        std::cout << "test10" << std::endl;
    }

    void set_prev_Pqs(const VectorXd &cur_Pq, const VectorXd &prev_Pq) {
        m_cur_Pq = cur_Pq;
        m_prev_Pq = prev_Pq;
    }

    double objective(const VectorXd& new_Pq) {
         // Update the tets with candidate configuration
        Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(tets->getQ(), *world);
        VectorXd expanded_q = P.transpose() * new_Pq;
        for(int i=0; i < q.size(); i++) {
            q[i] = expanded_q[i]; // TODO is this the fastest way to do this?
        }

        // Compute GPLC objective
        double obj_val = 0.0;
        VectorXd A = new_Pq - 2.0 * m_cur_Pq + m_prev_Pq;
        double energy = tets->getPotentialEnergy(world->getState());
        // std::cout << "energy: " << energy << std::endl;
        obj_val = 0.5 * A.transpose() * M * A + h * h * energy - h * h * A.transpose() * F_ext;
        // Gravity only
        // obj_val = 0.5 * A.transpose() * M * A - h*h * A.dot(F_ext);

        return obj_val;
    }
    
    double operator()(const VectorXd& new_Pq, VectorXd& grad)
    {
        double obj_val = objective(new_Pq);

        // Compute gradient
        VectorXd A = new_Pq - 2.0 * m_cur_Pq + m_prev_Pq;
        AssemblerEigenVector<double> internal_force; //maybe?
        // getForceVector(internal_force, *world);
        ASSEMBLEVECINIT(internal_force, P.cols());
        tets->getInternalForce(internal_force, world->getState());
        ASSEMBLEEND(internal_force);
        // // tets->getForce(internal_force, world->getState());  // THIS NEEDS PROPER INIT
        grad = M * A - h * h * P * (*internal_force) - h * h * F_ext;

        // external Forces only
        // grad = M * A - h * h * F_ext;

        // // // Finite differences instead
        // double t = 0.000001;
        // for(int i = 0; i < new_Pq.size(); i++) {
        //     VectorXd dq_pos(new_Pq);
        //     VectorXd dq_neg(new_Pq);
        //     dq_pos[i] += t;
        //     dq_neg[i] -= t;

        //     grad[i] = (objective(dq_pos) - objective(dq_neg)) / (2.0 * t);
        // }

        /////

        // std::cout << "Objective: " << obj_val << std::endl;
        return obj_val;
    }
};

class GPLCTimeStepper {
public:
    GPLCTimeStepper(MyWorld *world, NeohookeanTets *tets, double timestep) : m_world(world), m_tets(tets) {
        std::cout << "test1" << std::endl;
        // Set up lbfgs params
        m_lbfgs_param.epsilon = 1e-3; //1e-8// TODO replace convergence test with abs difference
        //m_lbfgs_param.delta = 1e-3;
        m_lbfgs_param.max_iterations = 10000;
        m_solver = new LBFGSSolver<double>(m_lbfgs_param);

        std::cout << "test2" << std::endl;
        construct_constraints_P();
        std::cout << "test3" << std::endl;
        Eigen::Map<Eigen::VectorXd> gauss_map_q = mapDOFEigen(tets->getQ(), *world);
        m_prev_Pq = m_P * gauss_map_q;
        std::cout << "test4" << std::endl;
        m_cur_Pq = m_P * gauss_map_q;

        std::cout << "test5" << std::endl;
        m_gplc_objective = new GPLC(m_cur_Pq, m_prev_Pq, timestep, world, tets, m_P);
        std::cout << "testend" << std::endl;
    }

    ~GPLCTimeStepper() {
        delete m_solver;
        delete m_gplc_objective;
    }

    void step() {
        VectorXd Pq_param = m_cur_Pq; // Stores both the first guess and the final result
        double min_val_res;

        m_gplc_objective->set_prev_Pqs(m_cur_Pq, m_prev_Pq);
        int niter = m_solver->minimize(*m_gplc_objective, Pq_param, min_val_res);

        std::cout << niter << " iterations" << std::endl;
        std::cout << "objective val = " << min_val_res << std::endl;

        m_prev_Pq = m_cur_Pq;
        m_cur_Pq = Pq_param; // TODO: Use pointers to avoid copies

        update_world_with_current_configuration();
        return;
    }

    void update_world_with_current_configuration() {
        // TODO: is this the fastest way to do this?
        Eigen::Map<Eigen::VectorXd> gauss_map_q = mapDOFEigen(m_tets->getQ(), *m_world);
        VectorXd expanded_q = m_P.transpose() * m_cur_Pq;
        for(int i=0; i < expanded_q.size(); i++) {
            gauss_map_q[i] = expanded_q[i];
        }
    }

    void reset() {
        Eigen::Map<Eigen::VectorXd> gauss_map_q = mapDOFEigen(m_tets->getQ(), *m_world);
        m_prev_Pq = m_P * gauss_map_q;
        m_cur_Pq = m_P * gauss_map_q;
    }

    void construct_constraints_P() {
        // Construct constraint projection matrix
        //find all vertices with minimum x coordinate and fix DOF associated with them
        int dim = 0; // x
        auto min_x_val = m_tets->getImpl().getV()(0,dim);
        std::vector<unsigned int> min_verts;

        for(unsigned int ii=0; ii<m_tets->getImpl().getV().rows(); ++ii) {
            if(m_tets->getImpl().getV()(ii,dim) < min_x_val) {
                min_x_val = m_tets->getImpl().getV()(ii,dim);
                min_verts.clear();
                min_verts.push_back(ii);
            } else if(fabs(m_tets->getImpl().getV()(ii,dim) - min_x_val) < 1e-5) {
                min_verts.push_back(ii);
            }
        }

        std::sort(min_verts.begin(), min_verts.end());
        int n = m_tets->getImpl().getV().rows() * 3;
        int m = n - min_verts.size()*3;
        m_P = SparseMatrix<double>(m, n);
        m_P.reserve(VectorXi::Constant(n, 1)); // Reserve enough space for 1 non-zero per column
        int min_vert_i = 0;
        int cur_col = 0;
        for(int i = 0; i < m; i+=3){
            while(min_verts[min_vert_i] * 3 == cur_col) { // Note * is for vert index -> flattened index
                cur_col += 3;
                min_vert_i++;
            }
            m_P.insert(i, cur_col) = 1.0;
            m_P.insert(i+1, cur_col+1) = 1.0;
            m_P.insert(i+2, cur_col+2) = 1.0;
            cur_col += 3;
        }
        m_P.makeCompressed();
        // std::cout << m_P << std::endl;
        // -- Done constructing P
    }

    void test_gradient() {
        auto q_map = mapDOFEigen(m_tets->getQ(), *m_world);
        VectorXd q(m_P * q_map);

        GPLC fun(q, q, 0.001, m_world, m_tets, m_P);
        
        VectorXd fun_grad(q.size());
        double fun_val = fun(q, fun_grad);

        VectorXd finite_diff_grad(q.size());
        VectorXd empty(q.size());
        double t = 0.000001;
        for(int i = 0; i < q.size(); i++) {
            VectorXd dq_pos(q);
            VectorXd dq_neg(q);
            dq_pos[i] += t;
            dq_neg[i] -= t;

            finite_diff_grad[i] = (fun(dq_pos, empty) - fun(dq_neg, empty)) / (2.0 * t);
        }

        VectorXd diff = fun_grad - finite_diff_grad;
        std::cout << "q size: " << q.size() << std::endl;
        std::cout << "Function val: " << fun_val << std::endl;
        std::cout << "Gradient diff: " << diff.norm() << std::endl;
        assert(diff.norm() < 1e-4);
    }

private:
    LBFGSParam<double> m_lbfgs_param;
    LBFGSSolver<double> *m_solver;
    GPLC *m_gplc_objective;

    SparseMatrix<double> m_P; // TODO sparse?
    VectorXd m_prev_Pq;
    VectorXd m_cur_Pq;

    MyWorld *m_world;
    NeohookeanTets *m_tets;
};



int main(int argc, char **argv) {
    std::cout<<"Testt Neohookean FEM \n";
    //Setup Physics
    MyWorld world;
    std::cout<<"defined world\n";
    igl::readMESH(argv[1], V, T, F);
    NeohookeanTets *tets = new NeohookeanTets(V,T);

    world.addSystem(tets);
    fixDisplacementMin(world, tets);
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    reset_world(world);

    // MyTimeStepper stepper(0.05);
    // stepper.step(world);
    GPLCTimeStepper gplc_stepper(&world, tets, 0.05);

    //gplc_stepper.test_gradient();

    /** libigl display stuff **/
    igl::viewer::Viewer viewer;

    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
        // predefined colors
        const Eigen::RowVector3d orange(1.0,0.7,0.2);
        const Eigen::RowVector3d yellow(1.0,0.9,0.2);
        const Eigen::RowVector3d blue(0.2,0.3,0.8);
        const Eigen::RowVector3d green(0.2,0.6,0.3);

        if(viewer.core.is_animating)
        {   
            // Save Current configuration
            std::stringstream filename;
            filename << output_dir << "displacements_" << current_frame << ".dmat";
            if(saving_training_data) {
                save_displacements_DMAT(filename.str(), world, tets);
            }

            // stepper.step(world);
            auto q = mapDOFEigen(tets->getQ(), world);
            std::cout << "Potential = " << tets->getPotentialEnergy(world.getState()) << std::endl;
            q *= 1.01;

            Eigen::MatrixXd newV = getCurrentVertPositions(world, tets); 
            // std::cout<< newV.block(0,0,10,3) << std::endl;
            viewer.data.set_vertices(newV);
            viewer.data.compute_normals();
            current_frame++;

            gplc_stepper.step();
            // test_gradient(world, tets);
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
                gplc_stepper.reset();
                break;
            }
            default:
            return false;
        }
        return true;
    };

    // viewer.callback_mouse_down = [&](igl::viewer::Viewer&, int, int)->bool
    // {   
    //     Eigen::MatrixXd curV = getCurrentVertPositions(world, tets); 
    //     last_mouse = Eigen::RowVector3f(viewer.current_mouse_x,viewer.core.viewport(3)-viewer.current_mouse_y,0);
    //     std::cout << last_mouse << std::endl;
    //     // Find closest point on mesh to mouse position
    //     int fid;
    //     Eigen::Vector3f bary;
    //     if(igl::unproject_onto_mesh(
    //         last_mouse.head(2),
    //         viewer.core.view * viewer.core.model,
    //         viewer.core.proj, 
    //         viewer.core.viewport, 
    //         curV, F, 
    //         fid, bary))
    //     {
    //         long c;
    //         bary.maxCoeff(&c);
    //         dragged_pos = curV.row(F(fid,c)) + Eigen::RowVector3d(0.001,0.0,0.0); //Epsilon offset so we don't div by 0
    //         dragged_vert = F(fid,c);
    //         std::cout << "Dragging vert: " << dragged_vert << std::endl;

    //         // Update the system
    //         is_dragging = true;
    //         forceSpring->getImpl().setStiffness(spring_stiffness);
    //         auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
    //         pinned_q = dragged_pos;//(dragged_pos).cast<double>(); // necessary?

    //         fem_attached_pos = PosFEM<double>(&tets->getQ()[dragged_vert],dragged_vert, &tets->getImpl().getV());
    //         forceSpring->getImpl().setPosition0(fem_attached_pos);

    //         return true;
    //     }
        
    //     return false; // TODO false vs true??
    // };

    // viewer.callback_mouse_up = [&](igl::viewer::Viewer&, int, int)->bool
    // {
    //     is_dragging = false;
    //     forceSpring->getImpl().setStiffness(0.0);
    //     // auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
    //     // pinned_q = Eigen::RowVector3d(-10000.0,0.0,0.0); // Move the point out of view

    //     return false;
    // };

    // viewer.callback_mouse_move = [&](igl::viewer::Viewer &, int,int)->bool
    // {
    //     if(is_dragging) {
    //         Eigen::RowVector3f drag_mouse(
    //             viewer.current_mouse_x,
    //             viewer.core.viewport(3) - viewer.current_mouse_y,
    //             last_mouse(2));

    //         Eigen::RowVector3f drag_scene,last_scene;

    //         igl::unproject(
    //             drag_mouse,
    //             viewer.core.view*viewer.core.model,
    //             viewer.core.proj,
    //             viewer.core.viewport,
    //             drag_scene);
    //         igl::unproject(
    //             last_mouse,
    //             viewer.core.view*viewer.core.model,
    //             viewer.core.proj,
    //             viewer.core.viewport,
    //             last_scene);

    //         dragged_pos += ((drag_scene-last_scene)*4.5).cast<double>(); //TODO why do I need to fine tune this
    //         // dragged_pos += (drag_mouse-last_mouse).cast<double>();
    //         last_mouse = drag_mouse;

    //         // Update the system
    //         // TODO dedupe this
    //         auto pinned_q = mapDOFEigen(pinned_point->getQ(), world);
    //         pinned_q = dragged_pos; //(dragged_pos).cast<double>(); // necessary?
    //     }

    //     return false;
    // };

    viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;

    viewer.launch();
    return EXIT_SUCCESS;
}

