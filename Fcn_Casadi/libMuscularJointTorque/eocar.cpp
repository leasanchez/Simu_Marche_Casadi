// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/muscular_joint_torque.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function f = casadi::external(libmuscular_joint_torque_name());
    
    casadi::DM activations(reshape(casadi::DM(std::vector<double>({0.5, 0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0})), m.nbMuscleTotal(), 1));
    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0, 0})), m.nbQ(), 1));
    casadi::DM Qdot(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0, 0})), m.nbQ(), 1));

    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {activations, Q, Qdot};
    std::vector<casadi::DM> fRes = f(arg);
    std::cout << "torque: " << fRes.at(0) << std::endl;

    return 0;
}
