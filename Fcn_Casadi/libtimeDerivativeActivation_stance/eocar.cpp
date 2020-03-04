// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/time_Derivative_Activation_stance.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function f = casadi::external(libtime_Derivative_Activation_stance_name());
    
    casadi::DM emg_Excitation(reshape(casadi::DM(std::vector<double>({0.5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3})), m.nbMuscleTotal(), 1));
     casadi::DM emg_Activation(reshape(casadi::DM(std::vector<double>({1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3,1e-3, 0.5, 1e-3, 1e-3, 1e-3, 1e-3})), m.nbMuscleTotal(), 1));

    // Use like any other CasADi function
    
    std::vector<casadi::DM> arg = {emg_Excitation, emg_Activation};
    std::vector<casadi::DM> fRes = f(arg);
    std::cout << "activationdot: " << fRes.at(0) << std::endl;

    return 0;
}
