// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "include/force_iso_max.h"

int main(){
    // Use CasADi's "external" to load the compiled function
    casadi::Function f = casadi::external(libforce_iso_max_name());
    int nGrp(1); 
    int nMus(1); 
    casadi::DM val(20);

    std::cout << "out func avant: " << m.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax() << std::endl;
    // Use like any other CasADi function
    std::vector<casadi::DM> arg = {nGrp, nMus, val};
    f(arg);
    f(arg);

    return 0;
}
