// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/forward_dynamics_contact.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function f = casadi::external(libforward_dynamics_contact_name());
    
    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0.5, 0, 0, 0, 0})), 5, 1));
    casadi::DM Qdot(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0})), 5, 1));
    casadi::DM Tau(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0})), 5, 1));

    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {Q, Qdot, Tau};
    std::vector<casadi::DM> fRes = f(arg);
    std::cout << "Xp et F: " << fRes.at(0) << std::endl;
    //std::cout << "F: " << fRes.at(1) << std::endl;
    return 0;
}
