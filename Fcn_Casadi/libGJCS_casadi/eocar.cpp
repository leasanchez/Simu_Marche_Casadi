// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/GJCS_casadi.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function f = casadi::external(libGJCS_casadi_name());
    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0})), 5, 1));
    casadi::DM BodyId(reshape(casadi::DM(std::vector<double>({2})), 1, 1));
    casadi::DM BodyPoint(reshape(casadi::DM(std::vector<double>({0,0,0})), 3, 1));
    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {Q,BodyId,BodyPoint};
    std::vector<casadi::DM> fRes = f(arg);
    std::cout << "result: " << fRes.at(0) << std::endl; 
    return 0;
}
