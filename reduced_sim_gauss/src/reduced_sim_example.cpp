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
#include <igl/readDMAT.h>
#include <igl/viewer/Viewer.h>
#include <igl/readMESH.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/get_seconds.h>

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


// -- My integrator

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using namespace LBFGSpp;


template <typename ReducedSpaceImpl>
class ReducedSpace
{
public:
    template<typename ...Params> // TODO necessary?
    ReducedSpace(Params ...params) : m_impl(params...) {}
    ReducedSpace() : m_impl() {}

    inline VectorXd encode(const VectorXd &q) {
        return m_impl.encode(q);
    }

    inline VectorXd decode(const VectorXd &z) {
        return m_impl.decode(z);
    }

    inline Eigen::Transpose<MatrixXd> jacobian(const VectorXd &z) { // todo should be const auto?
        // d decode / d z
        return m_impl.jacobian(z);
    }

private:
    ReducedSpaceImpl m_impl;
};

class IdentitySpaceImpl
{
public:
    IdentitySpaceImpl(int n) {
        m_I.resize(n,n);
        m_I.setIdentity();
    }

    inline VectorXd encode(const VectorXd &q) {return q;}
    inline VectorXd decode(const VectorXd &z) {return z;}
    inline SparseMatrix<double> jacobian(const VectorXd &z) {return m_I;}

private:
    SparseMatrix<double> m_I;
};

class LinearSpaceImpl
{
public:
    LinearSpaceImpl(const MatrixXd &U) : m_U(U) {
        std::cout<<"U rows: " << U.rows() << std::endl;
        std::cout<<"U cols: " << U.cols() << std::endl;
    }

    inline VectorXd encode(const VectorXd &q) {
        return m_U * q;
    }

    inline VectorXd decode(const VectorXd &z) {
        return m_U.transpose() * z;
    }

    inline Eigen::Transpose<MatrixXd> jacobian(const VectorXd &z) { // TODO: do this without copy?
        return m_U.transpose();
    }

private:
    MatrixXd m_U;
};

template <typename ReducedSpaceType>
class GPLC
{
public:
    GPLC(VectorXd cur_z,
        VectorXd prev_z,
        double h,
        MyWorld *world,
        NeohookeanTets *tets,
        ReducedSpaceType *reduced_space ) : 
            m_cur_z(cur_z),
            m_prev_z(prev_z),
            m_h(h),
            m_world(world),
            m_tets(tets),
            m_reduced_space(reduced_space)
    {
        // Construct mass matrix and external forces
        AssemblerEigenSparseMatrix<double> M_asm;
        getMassMatrix(M_asm, *m_world);

        m_M = *M_asm;
        VectorXd g(m_M.cols());
        for(int i=0; i < g.size(); i += 3) {
            g[i] = 0.0;
            g[i+1] = -9.8;
            g[i+2] = 0.0;
        }

        m_F_ext = m_M * g;
    }

    void set_prev_zs(const VectorXd &cur_z, const VectorXd &prev_z) {
        m_cur_z = cur_z;
        m_prev_z = prev_z;
    }

    // Just short helpers
    inline VectorXd dec(const VectorXd &z) { return m_reduced_space->decode(z); }
    inline VectorXd enc(const VectorXd &q) { return m_reduced_space->encode(q); }
    inline Eigen::Transpose<MatrixXd> jac(const VectorXd &z) { return m_reduced_space->jacobian(z); }

    double operator()(const VectorXd& new_z, VectorXd& grad)
    {
        // Update the tets with candidate configuration
        Eigen::Map<Eigen::VectorXd> q = mapDOFEigen(m_tets->getQ(), *m_world);
        VectorXd new_q = dec(new_z);
        for(int i=0; i < q.size(); i++) {
            q[i] = new_q[i]; // TODO is this the fastest way to do this?
        }

        // -- Compute GPLC objective
        double obj_val = 0.0;
        VectorXd A = new_q - 2.0 * dec(m_cur_z) + dec(m_prev_z); // TODO: avoid decodes here by saving the qs aswell
        double energy = m_tets->getPotentialEnergy(m_world->getState());

        obj_val = 0.5 * A.transpose() * m_M * A + m_h * m_h * energy - m_h * m_h * A.transpose() * m_F_ext;

        // -- Compute gradient
        AssemblerEigenVector<double> internal_force; //maybe?
        ASSEMBLEVECINIT(internal_force, new_q.size());
        m_tets->getInternalForce(internal_force, m_world->getState());
        ASSEMBLEEND(internal_force);

        Eigen::Transpose<Eigen::Transpose<MatrixXd>> jac_z_T = jac(new_z).transpose();
        grad = jac_z_T * (m_M * A - m_h * m_h * (*internal_force) - m_h * m_h * m_F_ext);

        // Finite differences gradient
        // double t = 0.000001;
        // for(int i = 0; i < new_z.size(); i++) {
        //     VectorXd dq_pos(new_z);
        //     VectorXd dq_neg(new_z);
        //     dq_pos[i] += t;
        //     dq_neg[i] -= t;
        //     grad[i] = (objective(dq_pos) - objective(dq_neg)) / (2.0 * t);
        // }

        // std::cout << "Objective: " << obj_val << std::endl;
        return obj_val;
    }
private:
    VectorXd m_cur_z;
    VectorXd m_prev_z;
    VectorXd m_F_ext;
    SparseMatrix<double> m_M; // mass matrix

    double m_h;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    ReducedSpaceType *m_reduced_space;
};

template <typename ReducedSpaceType>
class GPLCTimeStepper {
public:
    GPLCTimeStepper(MyWorld *world, NeohookeanTets *tets, double timestep, ReducedSpaceType *reduced_space) :
        m_world(world), m_tets(tets), m_reduced_space(reduced_space) {
        // Set up lbfgs params
        m_lbfgs_param.epsilon = 1e-3; //1e-8// TODO replace convergence test with abs difference
        //m_lbfgs_param.delta = 1e-3;
        m_lbfgs_param.max_iterations = 10000;
        m_solver = new LBFGSSolver<double>(m_lbfgs_param);

        reset_zs_to_current_world();
        m_gplc_objective = new GPLC<ReducedSpaceType>(m_cur_z, m_prev_z, timestep, world, tets, reduced_space);
    }

    ~GPLCTimeStepper() {
        delete m_solver;
        delete m_gplc_objective;
    }

    void step() {
        double start_time = igl::get_seconds();

        VectorXd z_param = m_cur_z; // Stores both the first guess and the final result
        double min_val_res;

        m_gplc_objective->set_prev_zs(m_cur_z, m_prev_z);
        int niter = m_solver->minimize(*m_gplc_objective, z_param, min_val_res);

        std::cout << niter << " iterations" << std::endl;
        std::cout << "objective val = " << min_val_res << std::endl;

        m_prev_z = m_cur_z;
        m_cur_z = z_param; // TODO: Use pointers to avoid copies

        update_world_with_current_configuration();

        std::cout << "Timestep took: " << igl::get_seconds() - start_time << "s" << std::endl;
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

    void reset_zs_to_current_world() {
        Eigen::Map<Eigen::VectorXd> gauss_map_q = mapDOFEigen(m_tets->getQ(), *m_world);
        m_prev_z = m_reduced_space->encode(gauss_map_q);
        m_cur_z = m_reduced_space->encode(gauss_map_q);
    }

    void test_gradient() {
        auto q_map = mapDOFEigen(m_tets->getQ(), *m_world);
        VectorXd q(m_reduced_space->encode(q_map));

        GPLC<ReducedSpaceType> fun(q, q, 0.001, m_world, m_tets);
        
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
    GPLC<ReducedSpaceType> *m_gplc_objective;

    VectorXd m_prev_z;
    VectorXd m_cur_z;

    MyWorld *m_world;
    NeohookeanTets *m_tets;

    ReducedSpaceType *m_reduced_space;
};

SparseMatrix<double> construct_constraints_P(NeohookeanTets *tets) {
    // Construct constraint projection matrix
    // Fixes all vertices with minimum x coordinate to 0
    int dim = 0; // x
    auto min_x_val = tets->getImpl().getV()(0,dim);
    std::vector<unsigned int> min_verts;

    for(unsigned int ii=0; ii<tets->getImpl().getV().rows(); ++ii) {
        if(tets->getImpl().getV()(ii,dim) < min_x_val) {
            min_x_val = tets->getImpl().getV()(ii,dim);
            min_verts.clear();
            min_verts.push_back(ii);
        } else if(fabs(tets->getImpl().getV()(ii,dim) - min_x_val) < 1e-5) {
            min_verts.push_back(ii);
        }
    }

    std::sort(min_verts.begin(), min_verts.end());
    int n = tets->getImpl().getV().rows() * 3;
    int m = n - min_verts.size()*3;
    SparseMatrix<double> P(m, n);
    P.reserve(VectorXi::Constant(n, 1)); // Reserve enough space for 1 non-zero per column
    int min_vert_i = 0;
    int cur_col = 0;
    for(int i = 0; i < m; i+=3){
        while(min_verts[min_vert_i] * 3 == cur_col) { // Note * is for vert index -> flattened index
            cur_col += 3;
            min_vert_i++;
        }
        P.insert(i, cur_col) = 1.0;
        P.insert(i+1, cur_col+1) = 1.0;
        P.insert(i+2, cur_col+2) = 1.0;
        cur_col += 3;
    }
    P.makeCompressed();
    // std::cout << P << std::endl;
    // -- Done constructing P
    return P;
}

typedef ReducedSpace<IdentitySpaceImpl> IdentitySpace;
typedef ReducedSpace<LinearSpaceImpl> LinearSpace;

int main(int argc, char **argv) {
    std::cout<<"Testt Neohookean FEM \n";
    //Setup Physics
    MyWorld world;
    std::cout<<"defined world\n";
    igl::readMESH(argv[1], V, T, F);
    NeohookeanTets *tets = new NeohookeanTets(V,T);
    for(auto element: tets->getImpl().getElements()) {
        element->setDensity(1000.0);//1000.0);
        element->setParameters(300000, 0.45);
    }

    world.addSystem(tets);
    fixDisplacementMin(world, tets);
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    reset_world(world);

    // SparseMatrix<double> P = construct_constraints_P(tets);
    //IdentitySpace reduced_space(tets->getImpl().getV().rows() * 3);
    MatrixXd U;
    igl::readDMAT("../../training_data/fixed_material_model/pca_components.dmat", U);
    LinearSpace reduced_space(U);
    GPLCTimeStepper<LinearSpace> gplc_stepper(&world, tets, 0.05, &reduced_space);
    // gplc_stepper.test_gradient();

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
                gplc_stepper.reset_zs_to_current_world();
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

