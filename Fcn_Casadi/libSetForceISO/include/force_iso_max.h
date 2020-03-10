#include <math.h>
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "Muscles/Muscle.h" 
#include "Muscles/Muscles.h" 
#include "Muscles/MuscleGroup.h" 
#include "Muscles/Characteristics.h" 

static biorbd::Model m("/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod");

#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

bool  libforce_iso_max_has_derivative(void);
const char* libforce_iso_max_name(void);

int libforce_iso_max(const casadi_real** arg,
                                                   double** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
casadi_int libforce_iso_max_n_in(void);
const char* libforce_iso_max_name_in(casadi_int i);
const casadi_int* libforce_iso_max_sparsity_in(casadi_int i);

// OUT
casadi_int libforce_iso_max_n_out(void);
const char* libforce_iso_max_name_out(casadi_int i);
const casadi_int* libforce_iso_max_sparsity_out(casadi_int i);

int libforce_iso_max_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);
// BIORBD_API int jaco_libforce_iso_max(const casadi_real** arg,
//                                                   double** res);

#ifdef __cplusplus
} /* extern "C" */
#endif

