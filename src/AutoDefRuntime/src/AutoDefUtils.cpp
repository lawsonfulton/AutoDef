#include <AutoDefUtils.h>

EnergyMethod energy_method_from_integrator_config(const json &integrator_config) {
    try {
        if(!integrator_config.at("use_reduced_energy")) {
            return FULL;
        }

        std::string energy_method = integrator_config.at("reduced_energy_method");    

        if(energy_method == "full") return FULL;
        if(energy_method == "pcr") return PCR;
        if(energy_method == "an08") return AN08;
        if(energy_method == "pred_weights_l1") return PRED_WEIGHTS_L1;
        if(energy_method == "pred_energy_direct") return PRED_DIRECT;
        if(energy_method == "new_pcr") return NEW_PCR;
        if(energy_method == "an08_all") return AN08_ALL;
    } 
    catch (nlohmann::detail::out_of_range& e){} // Didn't exist
    std::cout << "Unkown energy method." << std::endl;
    exit(1);
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

std::string ZeroPadNumber(int num, int N = 5)
{
    std::ostringstream ss;
    ss << std::setw( N ) << std::setfill( '0' ) << num;
    return ss.str();
}

double approxRollingAverage (double avg, double new_sample, int N=20) {
    avg -= avg / N;
    avg += new_sample / N;

    return avg;
};

void load_all_an08_indices_and_weights(fs::path energy_model_dir, std::vector<VectorXi> &all_indices, std::vector<VectorXd> &all_weights) {
    std::vector<std::string> paths;
    for(auto & p : boost::filesystem::directory_iterator( energy_model_dir )) {
        if(fs::is_directory(p)) {
            paths.push_back(p.path().string());
        }
    }

    std::sort(paths.begin(), paths.end());

    for(auto &p : paths) {
        std::string indices_path = p + "/indices.dmat";
        std::string weights_path = p + "/weights.dmat";
        
        Eigen::VectorXi Is;
        Eigen::VectorXd Ws;
        igl::readDMAT(indices_path, Is); //TODO do I need to sort this?
        igl::readDMAT(weights_path, Ws); 

        all_indices.push_back(Is);
        all_weights.push_back(Ws);

        std::cout << "Loaded tets from " << p << std::endl;
    }
}


// VectorXd tf_JTJ(const VectorXd &z, const VectorXd &z_v, tf::Session* m_decoder_JTJ_session) {
//     tf::Tensor z_tensor(tf_dtype, tf::TensorShape({1, z_v.size()})); 
//     auto z_tensor_mapped = z_tensor.tensor<tf_dtype_type, 2>();
//     for(int i =0; i < z.size(); i++) {
//         z_tensor_mapped(0, i) = z[i];
//     } // TODO map with proper function

//     tf::Tensor z_v_tensor(tf_dtype, tf::TensorShape({1, z_v.size()}));  
//     auto z_v_tensor_mapped = z_v_tensor.tensor<tf_dtype_type, 2>();
//     for(int i =0; i < z_v.size(); i++) {
//         z_v_tensor_mapped(0, i) = z_v[i];
//     } // TODO map with proper function

//     std::vector<tf::Tensor> JTJ_outputs; 
//     tf::Status status = m_decoder_JTJ_session->Run({{"decoder_input:0", z_tensor}, {"input_z_v:0", z_v_tensor}},
//                                {"JTJ/dense_decode_layer_0/MatMul_grad/MatMul:0"}, {}, &JTJ_outputs); // TODO get better names
    
//     auto JTJ_tensor_mapped = JTJ_outputs[0].tensor<tf_dtype_type, 2>();
    
//     VectorXd res(z_v.size()); // TODO generalize
//     for(int i = 0; i < res.rows(); i++) {
//         res[i] = JTJ_tensor_mapped(0,i);    
//     }
//     return res;
// }

// void compare_jac_speeds(AutoencoderSpace &reduced_space) {
//     fs::path tf_models_root("../../models/x-final/tf_models/");
//     fs::path decoder_JTJ_path = tf_models_root / "decoder_JTJ.pb";
//     tf::Session* m_decoder_JTJ_session;
//     tf::NewSession(tf::SessionOptions(), &m_decoder_JTJ_session);
    
//     tf::GraphDef decoder_JTJ_graph_def;
//     ReadBinaryProto(tf::Env::Default(), decoder_JTJ_path.string(), &decoder_JTJ_graph_def);
    
//     m_decoder_JTJ_session->Create(decoder_JTJ_graph_def);


//     for(int j = 100; j < 10001; j *= 10) {
//         int n_its = j;
        
//         VectorXd q = VectorXd::Zero(reduced_space.outer_jacobian().rows());
//         VectorXd z = reduced_space.encode(q);
//         VectorXd vq = MatrixXd::Ones(reduced_space.outer_jacobian().cols(), 1);
//         VectorXd vz = MatrixXd::Ones(z.size(), 1);
    
        
//         double start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             MatrixXd full_jac = reduced_space.inner_jacobian(z);
//             VectorXd Jv = full_jac * vz;    
//         }
//         std::cout << n_its << " its of fd jac product took " << igl::get_seconds() - start << std::endl;

//         start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             VectorXd Jv = reduced_space.jacobian_vector_product(z, vz);    
//         }
//         std::cout << n_its << " its of jvp took " << igl::get_seconds() - start << std::endl;

//         start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             MatrixXd full_jac = reduced_space.inner_jacobian(z);
//             VectorXd Jv = full_jac.transpose() * vq;    
//         }
//         std::cout << n_its << " its of fd jacT product took " << igl::get_seconds() - start << std::endl;

//         start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             VectorXd Jv = reduced_space.jacobian_transpose_vector_product(z, vq);    
//         }
//         std::cout << n_its << " its of vjp took " << igl::get_seconds() - start << std::endl;

//         MatrixXd U = reduced_space.outer_jacobian();
//         MatrixXd UTU = U.transpose() * U;
//         start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             MatrixXd full_jac = reduced_space.inner_jacobian(z);
//             VectorXd JUTUv = full_jac.transpose() * UTU * full_jac * vz;    
//         }
//         std::cout << n_its << " its of fd JUTUJv product took " << igl::get_seconds() - start << std::endl;

//         start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             VectorXd JUTUv = reduced_space.jacobian_transpose_vector_product(z, UTU * reduced_space.jacobian_vector_product(z, vz));    
//         }
//         std::cout << n_its << " its of vjp JUTUJv took " << igl::get_seconds() - start << std::endl;

//         start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             VectorXd JUTUv = reduced_space.jacobian_transpose_vector_product(z, reduced_space.jacobian_vector_product(z, vz));    
//         }
//         std::cout << n_its << " its of vjp JTJv took " << igl::get_seconds() - start << std::endl;

//         start = igl::get_seconds();
//         for(int i = 0; i < n_its; i++) {
//             VectorXd JUTUv = tf_JTJ(z, vz, m_decoder_JTJ_session);    
//         }
//         std::cout << n_its << " its of Tensorflow JTJv took " << igl::get_seconds() - start << std::endl;

//     }

// }