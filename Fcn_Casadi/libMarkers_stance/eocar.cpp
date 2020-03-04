// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/marche_Florent_markers_stance.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function marche_Florent_markers_stance = casadi::external(libmarche_Florent_markers_stance_name());
    
    casadi::DM Q(reshape(casadi::DM(std::vector<double>({0, 0, 0, 0, 0})), 5, 1));
    bool bool1(1);
    bool bool2(1);
    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {Q, bool1, bool2};
    std::vector<casadi::DM> marche_Florent_markers_stanceRes = marche_Florent_markers_stance(arg);
    std::cout << "marche_Florent_markers_stance" << marche_Florent_markers_stanceRes.at(0) << std::endl;

    return 0;
}
