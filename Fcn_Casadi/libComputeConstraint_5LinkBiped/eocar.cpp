// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/compute_constraint_5link_biped.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function f = casadi::external(libcompute_constraint_5link_biped_name());
    
    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0.5, 0, 0, 0, 0})), 5, 1));
    casadi::DM QdotMinus(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0})), 5, 1));

    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {Q, QdotMinus};
    std::vector<casadi::DM> fRes = f(arg);
    std::cout << "Q QdotPlus F: " << fRes.at(0) << std::endl;
    //std::cout << "F: " << fRes.at(1) << std::endl;
    return 0;
}
