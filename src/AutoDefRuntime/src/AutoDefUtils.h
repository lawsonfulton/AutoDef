#ifndef AutoDefUtils_H
#define AutoDefUtils_H

#include "TypeDefs.h"
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <iomanip>


#include <igl/readDMAT.h>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = boost::filesystem;

EnergyMethod energy_method_from_integrator_config(const json &integrator_config);

template<typename T>
T get_json_value(const json &j, const std::string &key, T def) {
    try {
        return j.at(key);
    }
    catch (nlohmann::detail::out_of_range& e){
        return def;
    }
}


std::string ZeroPadNumber(int num, int N = 5);

double approxRollingAverage (double avg, double new_sample, int N=20);

void load_all_an08_indices_and_weights(fs::path energy_model_dir, std::vector<VectorXi> &all_indices, std::vector<VectorXd> &all_weights);


#endif
