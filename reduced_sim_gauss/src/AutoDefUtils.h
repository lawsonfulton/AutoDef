#ifndef AutoDefUtils_H
#define AutoDefUtils_H

#include "TypeDefs.h"

#include <iostream>

#include <json.hpp>

using json = nlohmann::json;

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
    } 
    catch (nlohmann::detail::out_of_range& e){} // Didn't exist
    std::cout << "Unkown energy method." << std::endl;
    exit(1);
}

#endif
