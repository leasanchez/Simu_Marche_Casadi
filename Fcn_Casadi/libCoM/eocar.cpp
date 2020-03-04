// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/CoM.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function CoM = casadi::external(libCoM_name());
    
    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0})), 5, 1));
    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {Q};
    std::vector<casadi::DM> CoMRes = CoM(arg);
    std::cout << "CoM" << CoMRes.at(0) << std::endl;

    return 0;
}
