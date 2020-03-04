// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/marche_Florent_markers_swing.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function marche_Florent_markers_swing = casadi::external(libmarche_Florent_markers_swing_name());
    
    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0})), 5, 1));
    bool bool1(1);
    bool bool2(1);
    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {Q, bool1, bool2};
    std::vector<casadi::DM> markersRes = marche_Florent_markers_swing(arg);
    std::cout << "markers_swing" << markersRes.at(0) << std::endl;

    return 0;
}
