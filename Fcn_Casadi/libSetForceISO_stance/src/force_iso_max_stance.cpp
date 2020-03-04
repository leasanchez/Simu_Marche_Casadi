#include "force_iso_max_stance.h"
#include "Utils/Vector.h"
#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int forceISO_sparsity[3]  = {m.nbMuscleTotal(), 1, 1};
static casadi_int dummy_sparsity[3]  = {1, 1, 1};

const char* libforce_iso_max_stance_name(void){
    return "libforce_iso_max_stance";
}

int libforce_iso_max_stance(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
                             
     biorbd::utils::Vector  forceISO(m.nbMuscleTotal()); 

    // Dispatch data
    for (unsigned int i = 0; i < m.nbMuscleTotal(); ++i){
         forceISO[i] = arg[0][i];
    }

    // Change the isometric force 
    int n_muscle = 0; 
    for (unsigned int nGrp = 0; nGrp < m.nbMuscleGroups(); ++nGrp){
         for (unsigned int nMus = 0; nMus < m.muscleGroup(nGrp).nbMuscles(); ++nMus){
         double val = forceISO[n_muscle]; 
         m.muscleGroup(nGrp).muscle(nMus).setForceIsoMax(val); 
         ++ n_muscle; 
         }
     }
    return 0;
}

// IN
casadi_int libforce_iso_max_stance_n_in(void){
    return 1;
}
const char* libforce_iso_max_stance_name_in(casadi_int i){
    switch (i) {
    case 0: return "foceISO";
    default: return nullptr;
    }
}
const casadi_int* libforce_iso_max_stance_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return forceISO_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libforce_iso_max_stance_n_out(void){
    return 1;
}
const char* libforce_iso_max_stance_name_out(casadi_int i){
   switch (i) {
    case 0: return "dummy";
    default: return nullptr;
    }
}
const casadi_int* libforce_iso_max_stance_sparsity_out(casadi_int i) {
   switch (i) {
    case 0: return dummy_sparsity;
    default: return nullptr;
    }
}

int libforce_iso_max_stance_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = 3;
    if (sz_res) *sz_res = 1;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libforce_iso_max_stance_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
